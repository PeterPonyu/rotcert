#!/usr/bin/env python3
"""Fresh, time-boxed real-input S4 work; never a formal migration approval.

v3 (user-authorized final window, derived from the v2 candidate and NOT independently reviewed): adds the RTMDet .6
cell, up to 20 compute workers, a configurable number of verify workers and base-first deep ordering. Science,
RNG, B=5000 and every guard are unchanged.

Keep the frozen B=5000, source-draw RNG, prepared provenance and numerical
kernel. A new attempt owns new checkpoints; --reuse-window only reads older
windows made by this script. All attempts have immutable identities/receipts.
The foreground command must additionally run under budget_run.py's guardian.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import ctypes
from datetime import datetime, timezone
from functools import wraps
import importlib
import json
import math
import multiprocessing as mp
from multiprocessing.reduction import recv_handle, send_handle
import os
from pathlib import Path
import resource
import signal
import socket
import sys
import time
import traceback

import s4_multicore as core
import s4_multicore_full as control

PRIOR_CELLS = (("dior-s2anet-s0", .5), ("dior-s2anet-s0", .6),
               ("dior-s2anet-s0", .7), ("dior-rtmdet-s0", .7))   # the cells of the first two-hour window
CELLS = PRIOR_CELLS[:3] + (("dior-rtmdet-s0", .6), PRIOR_CELLS[3])
MAX_WORKERS = 20
DEEP_DRAWS = (-1, 0, 1, 2499, 4999)
SCHEMA = "S4-fresh-two-hour-window-v1"
STATE = None
DATA = None
PIPE = None
ROLES = None
INPUT = None
DATA_GUARDS = {}
LOCK_FD = None
SLOT = None
READY_BARRIER = None


class WindowStop(RuntimeError):
    pass


class WorkerTaskError(RuntimeError):
    """Picklable pre-redacted worker failure; no raw remote traceback payload."""
    def __init__(self, record):
        self.record = record
        super().__init__(record)


def affinity_plan(workers):
    """core.affinity_plan with the 16-worker ceiling raised to MAX_WORKERS; identical rules otherwise."""
    if workers <= 16:
        return core.affinity_plan(workers)
    core.require(type(workers) is int and workers <= MAX_WORKERS, "Workers must be 1..%d" % MAX_WORKERS)
    physical, selected = set(), []
    for cpu in sorted(os.sched_getaffinity(0)):
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            identity = ((base / "physical_package_id").read_text().strip(), (base / "core_id").read_text().strip())
        except FileNotFoundError:
            identity = ("unreported", str(cpu))
        if identity not in physical:
            physical.add(identity)
            selected.append(cpu)
    core.require(len(selected) >= workers, "Fewer physical cores available than requested")
    core.require(len(selected) - workers >= 4, "At least four physical cores must remain for the OS/other work")
    sys.path.insert(0, str(core.REPO))
    from rotcert.matching_parallel import _cgroup_cpu_quota
    quota = _cgroup_cpu_quota()
    if quota is not None:
        core.require(quota >= workers, "Cgroup CPU quota below requested worker count")
    return dict(worker_cpus=selected[:workers], available_physical_cpus=selected,
                reserved_physical_cpus=selected[workers:], cpu_quota=quota,
                quota_scope="readable cgroup v1/v2 hierarchy; null is unavailable, not unlimited")


def base_first(tasks):
    """The slow base-draw (-1) oracles first: the shortest critical path for the deep pool."""
    return sorted(tasks, key=lambda task: task[2] != -1)


def key(model, tau):
    return f"{model}/iou{tau:.1f}"


def pin_files():
    frozen = core.checked_json(core.KERNEL / "FROZEN.json", core.SOURCE_HASH)
    paths = [Path(__file__), Path(core.__file__), Path(control.__file__), core.INDEX,
             core.KERNEL / "FROZEN.json", core.LEGACY / "INPUTS.json"]
    paths.extend(core.KERNEL / name for name in frozen["files"])
    for model, tau in CELLS:
        for path in core.prepared_paths(model, tau):
            paths.extend((path, path.with_suffix(".json")))
    return {str(path): {"sha256": core.sha(path), "stat": list(core.stat_id(path))}
            for path in dict.fromkeys(paths)}


def pins_same(pins, hashes=False):
    for path, identity in pins.items():
        core.require(list(core.stat_id(path)) == identity["stat"], "Pinned input/source stat changed")
        if hashes:
            core.require(core.sha(path) == identity["sha256"], "Pinned input/source bytes changed")


def cancel_window(state, *_):
    """User/guardian cancellation is irrevocable, unlike a phase boundary."""
    state["cancel"].set()
    state["stop"].set()


def begin_phase(state, phase):
    if state["cancel"].is_set() or Path(state["stop_file"]).exists():
        raise WindowStop("window cancelled; next phase refused")
    state["phase"] = phase
    state["stop"].clear()


def safe_failure(error, phase):
    """Code locations and hashed messages, never arbitrary exception payloads."""
    frames = [{"file": Path(row.filename).name, "line": row.lineno, "function": row.name}
              for row in traceback.extract_tb(error.__traceback__)]
    report = dict(error_type=type(error).__name__, phase=phase, frames=frames,
                message_sha256=core.hashlib.sha256(str(error).encode()).hexdigest(),
                message_length=len(str(error)),
                cause_type=type(error.__cause__).__name__ if error.__cause__ else None,
                raw_exception_payload_retained=False)
    if isinstance(error, WorkerTaskError):
        report["worker_failure"] = error.record
    return report


def redacted_task(function):
    @wraps(function)
    def wrapped(task):
        try:
            return function(task)
        except WindowStop:
            raise
        except BaseException as error:
            phase = STATE["phase"] if STATE is not None else "control-test"
            report = safe_failure(error, phase)
            report["task"] = {"model": task[0], "iou": task[1], "draw": task[2]}
            # Scrub in the worker BEFORE ProcessPoolExecutor serializes it.
            raise WorkerTaskError(report) from None
    return wrapped


class WindowGuard:
    """Complete frozen resources interface, including ticks inside the kernel."""
    def __init__(self, state, parent=False):
        self.state, self.parent = state, parent
        self.last = -math.inf

    def check(self, stage=None, allocation=0):
        state = self.state
        clock = time.monotonic()
        if state["cancel"].is_set() or state["stop"].is_set() or Path(state["stop_file"]).exists():
            raise WindowStop("requested stop")
        until = state["compute_until"] if state["phase"] == "compute" else state["verify_until"]
        if clock >= until or time.time() >= state["deadline"]:
            raise WindowStop("window deadline")
        if not self.parent:
            parent = control.process_identity(state["owner"]["pid"])
            if (parent is None or parent["start_ticks"] != state["owner"]["start_ticks"]
                    or parent["state"] == "Z"):
                raise WindowStop("owner disappeared")
            if clock - state["heartbeat"].value > 15:
                raise WindowStop("supervisor sample stale")
        if not allocation and clock - self.last < .5:
            return
        self.last = clock
        pins_same(state["pins"])
        if core.available_memory() < state["memory_reserve"]:
            cancel_window(state)
            raise WindowStop("memory reserve")
        if core.shutil.disk_usage(state["out"]).free < state["disk_reserve"] + allocation:
            cancel_window(state)
            raise WindowStop("disk reserve")
        if state["charged_bytes"].value + allocation > state["max_bytes"]:
            raise WindowStop("output reservation cap")

    def before_write(self, path, size):
        raise ValueError("Frozen output writes are forbidden; this attempt uses reserved immutable IO")

    def after_write(self, path):
        core.require(Path(path).resolve().is_relative_to(Path(self.state["out"]).resolve()),
                     "Output escapes attempt")
        self.check()


def worker_init(cpus, slots, state, receiver, barrier):
    global STATE, INPUT, LOCK_FD, SLOT, READY_BARRIER
    with slots.get_lock():
        SLOT = slots.value
        slots.value += 1
    core.require(SLOT < len(cpus), "Unexpected worker replacement")
    STATE = state
    READY_BARRIER = barrier
    os.sched_setaffinity(0, {cpus[SLOT]})
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (min(4 * core.GIB, hard) if hard >= 0 else 4 * core.GIB, hard))
    LOCK_FD = recv_handle(receiver)
    receiver.close()
    # The inherited shared flock stays held even after sudden supervisor death.
    parent = control.process_identity(state["owner"]["pid"])
    core.require(parent is not None and parent["start_ticks"] == state["owner"]["start_ticks"],
                 "Parent identity changed during spawn")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot install parent-death signal")
    if os.getppid() != state["owner"]["pid"]:
        raise WindowStop("Parent disappeared while installing death signal")
    signal.signal(signal.SIGTERM, lambda *_: cancel_window(state))
    core.require(all(os.environ.get(name) == "1" for name in core.THREAD_KEYS), "Native thread drift")
    core.modules()
    importlib.import_module("resources").CURRENT = WindowGuard(state)
    core.verify_source()
    binding = core.validate_binding()
    ids, _ = core.source_roles(binding)
    core.require(ids == state["ids"], "Source roles changed")
    INPUT = type("BoundInput", (), {"ids": ids})()
    barrier.wait(timeout=90)


def ready(_):
    READY_BARRIER.wait(timeout=90)
    return {"health": core.health()}


def dataset(model, tau):
    global DATA, PIPE, ROLES, DATA_GUARDS
    tag = (model, tau)
    if STATE.get("cell") != tag:
        DATA, PIPE = None, None
        DATA, DATA_GUARDS = core.load_data(model, tau)
        _, ROLES = core.source_roles(core.validate_binding())
        PIPE = core.modules()["statistics_core"].ModelPipeline(DATA, ROLES)
        STATE["cell"] = tag
    core.require(all(core.stat_id(path) == before for path, before in DATA_GUARDS.items()),
                 "Prepared data changed")
    return DATA


def checkpoint_paths(out, model, tau, draw):
    path = Path(out) / key(model, tau) / (core.checkpoint_name(draw) + ".npz")
    return path, path.with_suffix(".json"), path.with_suffix(".checked.json")


def charge(size):
    with STATE["charged_bytes"].get_lock():
        if STATE["charged_bytes"].value + size > STATE["max_bytes"]:
            raise WindowStop("checkpoint reservation refused")
        STATE["charged_bytes"].value += size
    # Reservations remain conservatively charged through interruption/failure.


@redacted_task
def compute(task):
    model, tau, draw = task
    guard = importlib.import_module("resources").CURRENT
    started_cpu, started_wall = time.process_time(), time.monotonic()
    try:
        guard.check()
        dataset(model, tau)
        mult = core.modules()["statistics_core"].multiplicities(ROLES, draw)
        result = PIPE.run(mult)
        result["multiplicity"] = mult
        result["pipeline_cpu_seconds"] = core.modules()["np"].asarray(time.process_time() - started_cpu)
        core.modules()["run_s4"].validate_saved(result, ROLES, draw)
        check = core.modules()["verify_s4"].verify_one(result, INPUT, DATA, draw, False)
        guard.check()
        path, side, receipt = checkpoint_paths(STATE["out"], model, tau, draw)
        estimate = 2 * sum(array.nbytes for array in result.values()) + 131072
        charge(estimate)
        core.save_arrays(path, result, dict(core.checkpoint_metadata(model, tau, draw),
                                           execution_contract=SCHEMA,
                                           window_identity_sha256=STATE["identity_sha256"]))
        readback = core.modules()["io_s4"].load_npz(path, core.checkpoint_metadata(model, tau, draw))
        core.exact_compare(result, readback)
        record = dict(model=model, iou=tau, draw=draw, checkpoint_sha256=core.sha(path),
                      sidecar_sha256=core.sha(side), science_sha256=core.array_digest(readback),
                      light_verification=check, serialized_exact_readback=True,
                      window_identity_sha256=STATE["identity_sha256"],
                      cpu_seconds=time.process_time() - started_cpu,
                      wall_seconds=time.monotonic() - started_wall, health=core.health())
        core.write_json(receipt, record)
        return {"status": "SAVED", "cell": key(model, tau), "draw": draw, "receipt": str(receipt),
                "cpu_seconds": record["cpu_seconds"], "health": record["health"]}
    except WindowStop as error:
        return {"status": "INTERRUPTED_DRAW", "cell": key(model, tau), "draw": draw,
                "reason": str(error), "cpu_seconds": time.process_time() - started_cpu,
                "health": core.health()}


@redacted_task
def verify_task(task):
    model, tau, draw, path, deep = task
    guard = importlib.import_module("resources").CURRENT
    start_cpu, start_wall = time.process_time(), time.monotonic()
    try:
        guard.check()
        data = dataset(model, tau)
        before = {str(p): core.stat_id(p) for p in (Path(path), Path(path).with_suffix(".json"))}
        arrays = core.modules()["io_s4"].load_npz(path, core.checkpoint_metadata(model, tau, draw))
        core.require(all(core.stat_id(p) == stamp for p, stamp in before.items()),
                     "Checkpoint changed during verification read")
        core.modules()["run_s4"].validate_saved(arrays, ROLES, draw)
        checked = core.modules()["verify_s4"].verify_one(arrays, INPUT, data, draw, deep)
        return dict(status="VERIFIED", cell=key(model, tau), draw=draw, deep=deep,
                    checkpoint_sha256=core.sha(path), science_sha256=core.array_digest(arrays),
                    verification=checked, cpu_seconds=time.process_time() - start_cpu,
                    wall_seconds=time.monotonic() - start_wall, health=core.health())
    except WindowStop:
        return dict(status="INTERRUPTED_VERIFICATION", cell=key(model, tau), draw=draw, deep=deep,
                    cpu_seconds=time.process_time() - start_cpu, health=core.health())


def process_cpu(start_cpu):
    """Only this fresh attempt: parent + waited children + live descendants.

    Historical costs are deliberately separate, never admitted as zero. Retry
    racing reap snapshots, using the proven second rusage snapshot just once.
    """
    for _ in range(5):
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        prior = before.ru_utime + before.ru_stime
        rows = {}
        for path in Path("/proc").iterdir():
            if path.name.isdigit():
                row = control.process_identity(int(path.name))
                if row is not None:
                    rows[row["pid"]] = row
        parents, found = {os.getpid()}, []
        while True:
            children = [row for pid, row in rows.items() if pid not in parents and row["ppid"] in parents]
            if not children:
                break
            found.extend(children)
            parents.update(row["pid"] for row in children)
        live = 0.
        for row in found:
            current = control.process_identity(row["pid"])
            if current is not None:
                core.require(current["start_ticks"] == row["start_ticks"], "PID recycled during accounting")
                live += current["cpu_seconds"]
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        reaped = after.ru_utime + after.ru_stime
        if prior == reaped:
            own = resource.getrusage(resource.RUSAGE_SELF)
            return own.ru_utime + own.ru_stime + reaped - start_cpu + live
    raise WindowStop("Cannot obtain consistent cumulative CPU sample")


def strict_size(root):
    return control.retained_bytes(root)


def inventory(out, identity_hash, allow_incomplete=False):
    """Identity/hash/readback candidates, not full scientific acceptance."""
    found, pending = {}, []
    allowed = {key(*cell) for cell in CELLS}
    for path in sorted(Path(out).rglob("*.npz")):
        relative = path.relative_to(out)
        core.require(not path.is_symlink() and len(relative.parts) == 3 and
                     "/".join(relative.parts[:2]) in allowed, "Checkpoint path escape/unplanned cell")
        model, tau = relative.parts[0], float(relative.parts[1][3:])
        stem = path.stem
        draw = -1 if stem == "base" else int(stem.removeprefix("draw-"))
        core.require(-1 <= draw < 5000 and path.name == core.checkpoint_name(draw) + ".npz",
                     "Unplanned draw")
        _, side, receipt = checkpoint_paths(out, model, tau, draw)
        if not side.is_file():
            pending.append(str(path))
            continue
        meta = core.checked_json(side)
        expected = core.checkpoint_metadata(model, tau, draw)
        core.require(all(meta.get(name) == value for name, value in expected.items()) and
                     meta.get("execution_contract") == SCHEMA and
                     meta.get("window_identity_sha256") == identity_hash and
                     meta.get("sha256") == core.sha(path) and meta.get("file") == path.name,
                     "Checkpoint identity/hash mismatch")
        if not receipt.is_file():
            pending.append(str(path))
            continue
        checked = core.checked_json(receipt)
        core.require(checked.get("window_identity_sha256") == identity_hash and
                     checked.get("serialized_exact_readback") is True and
                     checked.get("checkpoint_sha256") == meta["sha256"] and
                     checked.get("sidecar_sha256") == core.sha(side) and
                     (checked.get("model"), checked.get("iou"), checked.get("draw")) == (model, tau, draw),
                     "Checkpoint checked receipt mismatch")
        found[(model, tau, draw)] = {"path": str(path), "sha256": meta["sha256"],
                                   "science_sha256": checked["science_sha256"],
                                   "receipt_sha256": core.sha(receipt)}
    # Sidecar-only / temporary evidence is retained and excluded, never erased.
    for path in Path(out).rglob("*"):
        core.require(not path.is_symlink(), "Symlink in attempt")
        if path.is_file() and ".tmp-" in path.name:
            pending.append(str(path))
        elif path.is_file() and path.suffix == ".json" and not path.name.endswith(".checked.json") and (
                path.stem == "base" or path.stem.startswith("draw-")) and not path.with_suffix(".npz").exists():
            pending.append(str(path))
    if not allow_incomplete:
        core.require(not pending, "Incomplete checkpoint pair/receipt: retained, not reused")
    return found, sorted(set(pending))


def previous_window(root):
    root = core.owned_output(root)
    core.require(not root.is_symlink(), "Reuse symlink")
    identity = core.checked_json(root / "IDENTITY.json")
    core.require(identity["schema"] == SCHEMA and identity["source_sha256"] == core.SOURCE_HASH and
                 identity["input_binding_sha256"] == core.INPUT_HASH and
                 identity["cells"] in ([[model, tau] for model, tau in PRIOR_CELLS],
                                       [[model, tau] for model, tau in CELLS]) and identity["B"] == 5000,
                 "Reuse window science contract differs")
    owner = identity["owner"]
    current = control.process_identity(owner["pid"])
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    core.require(boot != identity["boot_id"] or current is None or
                 current["start_ticks"] != owner["start_ticks"] or current["state"] == "Z",
                 "Reuse window still has an active owner")
    return inventory(root / "checkpoints", core.sha(root / "IDENTITY.json"), allow_incomplete=True)


def run_phase(tasks, function, cpus, state, update):
    ctx = mp.get_context("spawn")
    slots = ctx.Value("i", 0)
    barrier = ctx.Barrier(len(cpus))
    sender, receiver = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    pool, records, actual = None, [], {}
    try:
        for _ in cpus:
            send_handle(sender, control._CLAIM_FD, os.getpid())
        with core.single_thread_spawn():
            pool = ProcessPoolExecutor(max_workers=len(cpus), mp_context=ctx,
                                       initializer=worker_init, initargs=(cpus, slots, state, receiver, barrier))
            startup = [pool.submit(ready, None) for _ in cpus]
            pending = set(startup)
            while pending:
                update(state, records, "starting")
                done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    records.append(future.result())
            actual = control.assert_worker_health(records, cpus)
            core.write_json(Path(state["out"]).parent / f"WORKERS-{state['phase']}.json",
                            dict(phase=state["phase"], actual_workers=actual, receipts=records, utc=core.utc()))
            records = []
            iterator = iter(tasks)
            exhausted = object()
            pending = set()
            for _ in range(2 * len(cpus)):
                task = next(iterator, exhausted)
                if task is exhausted:
                    break
                pending.add(pool.submit(function, task))
            while pending:
                update(state, records, "running")
                done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    record = future.result()
                    core.require(record["health"]["pid"] in actual, "Unobserved worker")
                    records.append(record)
                    if record["status"].startswith("INTERRUPTED"):
                        state["stop"].set()
                    if not state["stop"].is_set():
                        task = next(iterator, exhausted)
                        if task is not exhausted:
                            pending.add(pool.submit(function, task))
            return records, actual
    finally:
        state["stop"].set()
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        sender.close()
        receiver.close()


def run(args):
    started = time.monotonic()
    start_cpu = core.cpu_total()
    out = core.owned_output(args.out)
    core.require(not out.exists(), "Use a fresh attempt; resume writes a NEW directory")
    core.require(0 < args.seconds <= 7200 and 0 < args.verify_seconds < args.seconds - 10,
                 "Finite <=2-hour window with closeout reserve required")
    core.require(args.workers in (1, 12, 16, 20), "Workers must be 1/12/16/20")
    verify_workers = args.workers if args.verify_workers is None else args.verify_workers
    core.require(1 <= verify_workers <= args.workers, "Verify workers must be 1..workers")
    core.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    with control.atomic_execution_claim(out) as owner:
        core.verify_source()
        affinity = affinity_plan(args.workers)
        core.require(core.available_memory() >= args.workers * 2 * core.GIB + 8 * core.GIB,
                     "Insufficient worker admission memory")
        core.require(core.shutil.disk_usage(out).free >= 16 * core.GIB, "Insufficient disk reserve")
        # Keep supervisor CPU on a reserved core, not a compute core.
        if affinity["reserved_physical_cpus"]:
            os.sched_setaffinity(0, {affinity["reserved_physical_cpus"][0]})
        pins = pin_files()
        binding = core.validate_binding()
        ids, _ = core.source_roles(binding)
        ctx = mp.get_context("spawn")
        state = dict(out=str(out / "checkpoints"), pins=pins, ids=ids, owner=owner,
                     phase="compute", stop=ctx.Event(), cancel=ctx.Event(),
                     heartbeat=ctx.Value("d", time.monotonic()),
                     stop_file=str(out / "STOP-REQUEST.json"),
                     charged_bytes=ctx.Value("q", 4 * 1024 ** 2), max_bytes=args.max_bytes,
                     memory_reserve=8 * core.GIB, disk_reserve=8 * core.GIB,
                     compute_until=started + args.seconds - args.verify_seconds - 10,
                     verify_until=started + args.seconds - 10, deadline=time.time() + args.seconds)
        state["cpu_limit"] = args.cpu_limit
        identity = dict(schema=SCHEMA, utc=core.utc(), source_sha256=core.SOURCE_HASH,
                        input_binding_sha256=core.INPUT_HASH, index_sha256=core.INDEX_HASH,
                        runner_sha256=core.sha(__file__), cells=[[model, tau] for model, tau in CELLS],
                        B=5000, RNG=core.RNG, science_semantics_unchanged=True, scientific_acceptance=False,
                        owner=owner, boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                        affinity=affinity, workers=args.workers, verify_workers=verify_workers, runner_version="v3-final-window",
                        prior_cells=[[model, tau] for model, tau in PRIOR_CELLS],
                        max_window_seconds=args.seconds, compute_stop_monotonic=state["compute_until"],
                        started_monotonic=started, deadline_utc=datetime.fromtimestamp(
                            state["deadline"], timezone.utc).isoformat(),
                        max_new_aggregate_cpu_seconds=args.cpu_limit, max_new_output_bytes=args.max_bytes,
                        cpu_limit_semantics="sampled own fresh attempt; guardian overhead recorded separately",
                        byte_limit_semantics="conservative per-checkpoint reservations plus polled actual bytes; not quota",
                        historical_costs_reset=False, old_migration_excluded=True,
                        historical_accounting="saved floor and unresolved tail/control remain in original records; "
                                              "new independent authorized window cost must be reconciled before formal merge",
                        reuse_windows=[str(core.owned_output(path)) for path in args.reuse_window], pins=pins)
        core.write_json(out / "IDENTITY.json", identity)
        state["identity_sha256"] = core.sha(out / "IDENTITY.json")
        (out / "checkpoints").mkdir()
        signal.signal(signal.SIGTERM, lambda *_: cancel_window(state))
        signal.signal(signal.SIGINT, lambda *_: cancel_window(state))
        events_path = out / "PROGRESS.jsonl"
        last = [-math.inf]

        def update(current, records, activity):
            clock = time.monotonic()
            current["heartbeat"].value = clock
            used = process_cpu(start_cpu)
            until = current["compute_until"] if current["phase"] == "compute" else current["verify_until"]
            if clock >= until or used >= current["cpu_limit"] - 100:
                current["stop"].set()
            if core.available_memory() < current["memory_reserve"] or core.shutil.disk_usage(out).free < current["disk_reserve"]:
                cancel_window(current)
            if clock - last[0] >= 30:
                used_bytes = strict_size(out)
                if used_bytes >= args.max_bytes:
                    cancel_window(current)
                row = dict(utc=core.utc(), phase=current["phase"], activity=activity,
                           elapsed_seconds=clock - started, aggregate_cpu_seconds=used,
                           phase_finished_tasks=len(records), retained_bytes=used_bytes,
                           reserved_bytes=current["charged_bytes"].value, stop=current["stop"].is_set(),
                           cancelled=current["cancel"].is_set(),
                           available_memory_bytes=core.available_memory(),
                           free_disk_bytes=core.shutil.disk_usage(out).free)
                with events_path.open("a") as stream:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                print(json.dumps(row), flush=True)
                last[0] = clock
            pins_same(pins)

        resources = importlib.import_module("resources")
        resources.CURRENT = WindowGuard(state, parent=True)
        computed, verified, reuse, pending, failure = [], [], {}, [], None
        reuse_pins = {}
        try:
            # Bound archive hashes first; complete semantic InputSet acceptance
            # remains separately labelled and is not fabricated by this window.
            files = binding.get("files", {})
            core.require(files, "Original bound archive inventory absent")
            for path, digest in files.items():
                update(state, [], "archive_hash_admission")
                resources.CURRENT.check()
                core.require(core.sha(path) == digest, "Bound archive drift")
            core.write_json(out / "ADMISSION.json", dict(status="PASS_HASH_BOUND_PREPARED_REAL_INPUTS",
                            files=len(files), sources=len(ids), prepared_provenance_frozen=True,
                            full_inputset_semantic_replay=False, scientific_acceptance=False, utc=core.utc()))
            for prior in args.reuse_window:
                candidates, incomplete = previous_window(prior)
                for token, record in candidates.items():
                    core.require(token not in reuse or reuse[token]["science_sha256"] == record["science_sha256"],
                                 "Prior window science conflict")
                    reuse[token] = record
                    path = Path(record["path"])
                    for pinned in (path, path.with_suffix(".json"), path.with_suffix(".checked.json"),
                                   Path(prior) / "IDENTITY.json"):
                        reuse_pins[str(pinned)] = {"sha256": core.sha(pinned), "stat": list(core.stat_id(pinned))}
                pending.extend(incomplete)
            if reuse_pins:
                # Full immutable evidence, but do not stat tens of thousands of
                # reused files inside every hot geometry tick. Each loaded pair
                # has its own read guard; this full set is checked at boundaries.
                pins_same(reuse_pins, hashes=True)
                core.write_json(out / "REUSE-PINS.json", reuse_pins)
            # Every reused checkpoint gets an independent light replay before
            # omission from the work queue; no old formal-root adoption here.
            if reuse:
                begin_phase(state, "reuse-light")
                tasks = [(model, tau, draw, row["path"], False) for (model, tau, draw), row in reuse.items()]
                adopted, _ = run_phase(tasks, verify_task, affinity["worker_cpus"][:verify_workers], state, update)
                core.require(len(adopted) == len(reuse) and all(row["status"] == "VERIFIED" for row in adopted),
                             "Prior window light replay incomplete")
                core.require(all(row["science_sha256"] == reuse[(row["cell"].split("/")[0],
                    float(row["cell"].split("iou")[1]), row["draw"])]["science_sha256"] for row in adopted),
                    "Reused science digest changed")
                pins_same(reuse_pins, hashes=True)
                core.write_json(out / "REUSED.json", dict(records=adopted, incomplete_excluded=pending, utc=core.utc()))
            begin_phase(state, "compute")
            resources.CURRENT = WindowGuard(state, parent=True)
            tasks = ((model, tau, draw) for model, tau in CELLS for draw in range(-1, 5000)
                     if (model, tau, draw) not in reuse)
            computed, actual = run_phase(tasks, compute, affinity["worker_cpus"], state, update)
            resources.CURRENT = None
            if reuse_pins:
                pins_same(reuse_pins, hashes=True)
            candidates, incomplete = inventory(out / "checkpoints", state["identity_sha256"], allow_incomplete=True)
            pending.extend(incomplete)
            core.write_json(out / "COMPUTED.json", dict(records=computed, actual_workers=actual,
                            checked_candidates=len(candidates), reused=len(reuse), incomplete_excluded=pending,
                            scientific_acceptance=False, utc=core.utc()))
            # Exact startup proof for the next pool happens only after compute
            # workers have completely shut down. Deep scope is never shortened
            # and mislabelled: selected full-oracle draws may remain incomplete.
            if (not state["cancel"].is_set() and not Path(state["stop_file"]).exists() and
                    time.monotonic() < state["verify_until"] - 5 and process_cpu(start_cpu) < args.cpu_limit - 100):
                begin_phase(state, "deep")
                state["heartbeat"].value = time.monotonic()
                all_candidates = dict(reuse)
                all_candidates.update(candidates)
                tasks = [(model, tau, draw, all_candidates[(model, tau, draw)]["path"], True)
                         for model, tau in CELLS for draw in DEEP_DRAWS
                         if (model, tau, draw) in all_candidates]
                tasks = base_first(tasks)
                verified, actual_verify = run_phase(tasks, verify_task,
                    affinity["worker_cpus"][:verify_workers], state, update)
                core.write_json(out / "DEEP.json", dict(records=verified, actual_workers=actual_verify,
                                scientific_acceptance=False, utc=core.utc()))
        except BaseException as error:
            failure = safe_failure(error, state["phase"])
            state["stop"].set()
            core.write_json(out / "FAILURE.json", failure)
        finally:
            resources.CURRENT = None
            state["stop"].set()
        # Interrupted publications are preserved, explicitly excluded from the
        # immutable resume list. Only complete checked pairs are reusable.
        candidates, incomplete = inventory(out / "checkpoints", state["identity_sha256"], allow_incomplete=True)
        pending.extend(incomplete)
        pins_same(pins, hashes=True)
        if reuse_pins:
            pins_same(reuse_pins, hashes=True)
        counts = {key(*cell): sum(token[:2] == cell for token in candidates) for cell in CELLS}
        result = dict(schema=SCHEMA, status="FAILED_PRESERVED" if failure else "WINDOW_CLOSED_PARTIAL_S4",
                      failure=failure, cancelled=state["cancel"].is_set(),
                      utc=core.utc(), elapsed_seconds=time.monotonic() - started,
                      aggregate_cpu_seconds=process_cpu(start_cpu), new_checked_checkpoints=len(candidates),
                      new_checkpoints_per_cell=counts, reused_checkpoints=len(reuse),
                      interrupted_artifacts=sorted(set(pending)),
                      completed_deep_checks=sum(row.get("status") == "VERIFIED" for row in verified),
                      identity_sha256=state["identity_sha256"], runner_sha256=core.sha(__file__),
                      retained_bytes=strict_size(out), scientific_acceptance=False,
                      formal_B5000_completion=False, legacy_output_modified=False,
                      resume_policy="new attempt + --reuse-window; validate identity/hash/light replay; no overwrite",
                      checkpoints=[dict(model=token[0], iou=token[1], draw=token[2], **row)
                                   for token, row in candidates.items()])
        core.write_json(out / "RESULT.json", result)
        print(json.dumps({name: value for name, value in result.items() if name != "checkpoints"}), flush=True)
        return 1 if failure else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--verify-workers", type=int, default=None, help="default: same as --workers")
    parser.add_argument("--seconds", type=float, default=7100)
    parser.add_argument("--verify-seconds", type=float, default=180)
    parser.add_argument("--cpu-limit", type=float, default=110000)
    parser.add_argument("--max-bytes", type=int, default=12 * core.GIB)
    parser.add_argument("--reuse-window", type=Path, action="append", default=[])
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())

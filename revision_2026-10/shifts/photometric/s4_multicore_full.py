#!/usr/bin/env python3
"""Read-only legacy checkpoint overlay and reviewed multicore S4 completion.

plan is read-only except its new manifest. execute requires a hash-pinned review
and finite aggregate budgets. Old ACTIVE/locks/results are never rewritten.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
import fcntl
import math
import multiprocessing as mp
import os
from pathlib import Path
import resource
import socket
import sys
import time
from datetime import datetime

_ENTRY_WALL = time.monotonic()
_ENTRY_UTC = time.time()

import s4_multicore as core

_PLAN = None
_INPUTS = None
_LIMITS = None
_PINS = None
_START_CPU = None
_START_WALL = None
_PIN_STATS = None
_LAST_PIN_HASH = None
_HISTORY = None
_RETAINED_ROOTS = None
_LAST_SAMPLE = None
_PARENT_ID = None
_CLAIM_FD = None
_COOPERATIVE_HEADROOM = None
_TERMINAL_RESERVE = None
_READY_BARRIER = None
_SAMPLING = False
_LAST_RETAINED_SCAN = None
_STOP_EVENT = None
_LAST_HOST_CHECK = None
SCHEDULE_HASH = "2d1daced4992342ddb8bd181a79cefef8d5c9ad8d0e9d5a15144aacf9dacdcda"
SAVED_LIFECYCLE_HASH = "27eee9d47837f810c5aa8c6ad02964d130d42be57ea43a2297e0892488a17ccf"
# Workers refuse a supervisor sample older than this. 5 s stopped two formal attempts on transient parent stalls
# (2026-10-04; swap full on the host). Unsupervised work is bounded by 60 s x workers, at most 960 CPU s and 60 s
# wall, inside the reviewed cooperative headroom (4,000 CPU s, 3,600 s) the parent keeps below every limit.
SAMPLE_MAX_AGE = 60.0
SLOW_SAMPLE_SECONDS = 3.0
LEDGER_ALLOWANCE = 65536
COMPONENTS = ("input_admission", "missing_draw_compute", "all_draw_light_verification",
              "sixty_full_deep_checks", "all_endpoint_interval_reduction", "closeout")


def runner_hashes():
    return {"s4_multicore.py": core.sha(core.__file__),
            "s4_multicore_full.py": core.sha(__file__),
            "s4_window_v3.py": core.sha(Path(__file__).with_name("s4_window_v3.py")),
            "matching_parallel.py": core.sha(core.REPO / "rotcert/matching_parallel.py")}


def retained_bytes(out):
    total = 0
    # Test evidence deliberately retains bad links and pytest aliases. Charge
    # each link inode's own logical bytes, never its target or linked directory.
    # Scientific roots and this execution's output remain strictly link-free.
    controls = [core.REPO / "var/planc-local-closure-20261002/s4-admission",
                core.REPO / "var/s4-high-risk-fix-20261002"]
    for path in Path(out).rglob("*"):
        try:
            if path.is_symlink():
                core.require(any(path.is_relative_to(root) for root in controls), "Symlink in retained output")
                total += path.lstat().st_size
            elif path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            # Only normal disappearance during atomic publication/scratch cleanup.
            continue
    return total


def original_authority():
    """A review may tighten the original contract, never replace its authority."""
    path = core.KERNEL / "PARENT-SCHEDULE.json"
    value = core.checked_json(path, SCHEDULE_HASH)
    core.require(value["source_sha256"] == core.SOURCE_HASH and
                 value["index_sha256"] == core.INDEX_HASH and
                 value["input_binding_sha256"] == core.INPUT_HASH and value["B"] == 5000,
                 "Original schedule scientific identity drift")
    return dict(path=str(path), sha256=SCHEDULE_HASH), dict(
        max_aggregate_cpu_seconds=value["max_cpu_seconds"],
        max_wall_seconds=value["max_wall_seconds"], max_output_bytes=value["max_output_bytes"],
        minimum_free_disk_bytes=value["minimum_free_disk_bytes"],
        minimum_available_memory_bytes=value["minimum_available_memory_bytes"],
        deadline_unix_seconds=datetime.fromisoformat(value["deadline_utc"]).timestamp())


def required_retained_roots():
    parent = core.KERNEL.parents[3]
    return [core.LEGACY, parent / "recovery-host-execution-v1",
            parent / "recovery-host-execution-v2", core.OUTPUT_ROOT,
            core.REPO / "var/planc-local-closure-20261002/s4-admission",
            core.REPO / "var/s4-high-risk-fix-20261002"]


def validate_accounting(binding, entry_utc):
    """Require an append-only, evidence-bound reconciliation, not a saved floor PASS.

    Existing prior_charged_debit is ALREADY inside the saved cumulative values.
    Reconciliation entries are incremental only; old receipts are never changed.
    """
    path = Path(binding["path"]).resolve()
    core.require(path.is_relative_to(core.REPO) and not path.is_relative_to(core.KERNEL),
                 "Reconciliation must be a new repo-local artifact, not a legacy overwrite")
    value = core.checked_json(path, binding["sha256"])
    saved_path = core.LEGACY / "attempts/recovery-2/LIFECYCLE.json"
    saved = core.checked_json(saved_path, SAVED_LIFECYCLE_HASH)
    core.require(value.get("schema") == "S4-cumulative-reconciliation-v1" and
                 value.get("saved_lifecycle") == dict(path=str(saved_path), sha256=SAVED_LIFECYCLE_HASH) and
                 value.get("schedule_sha256") == SCHEDULE_HASH,
                 "Reconciliation original authority/lineage drift")
    core.require(saved["identity"]["schedule_sha256"] == SCHEDULE_HASH and saved["attempt"] == 2,
                 "Saved cumulative lineage drift")
    entries = value.get("incremental_entries", [])
    core.require(entries and len({entry["id"] for entry in entries}) == len(entries) and
                 {entry["scope"] for entry in entries} >= {"unclean_recovery2_tail", "post_recovery_control"},
                 "Unclean tail and all later control costs require distinct incremental reconciliation")
    totals = dict(cpu_seconds=saved["cpu_seconds"], wall_seconds=saved["wall_seconds"])
    covered_work = set()
    for entry in entries:
        core.require(entry.get("incremental_only") is True and bool(entry.get("basis")),
                     "Historical debit must not be re-added or reset")
        work = entry.get("coverage_ids", [])
        core.require(work and len(set(work)) == len(work) and not covered_work.intersection(work),
                     "Explicit unique work coverage required; no overlapping historical debit")
        covered_work.update(work)
        for metric in totals:
            cost = entry.get(metric)
            core.require(type(cost) in (int, float) and math.isfinite(cost) and cost >= 0,
                         "Invalid incremental historical cost: " + metric)
            totals[metric] += cost
        core.require(core.sha(entry["evidence_path"]) == entry["evidence_sha256"],
                     "Reconciliation evidence drift")
    core.require(value.get("cumulative_debit") == totals, "Cumulative history must equal saved floor plus increments exactly once")
    coverage = value.get("control_covered_until_unix_seconds")
    core.require(type(coverage) in (int, float) and math.isfinite(coverage) and coverage >= entry_utc,
                 "Reconciliation does not cover control/rework through this launch")
    core.require(value.get("measurement_exact") is False and value.get("physical_upper_proven") is False,
                 "Conditional historical debit cannot be relabelled exact or a physical upper")
    core.require(value.get("coverage_complete") is True and
                 value.get("wall_accounting") == "parent/control coverage; no parallel-worker wall sum" and
                 value.get("reservation_overlap_reviewed") is True,
                 "Incomplete control/reservation coverage requires reconciliation review")
    roots = [Path(root).resolve() for root in value.get("retained_roots", [])]
    core.require(roots and len(set(roots)) == len(roots) and
                 all(root.is_relative_to(core.REPO) and root != core.REPO and root.is_dir() for root in roots),
                 "Exact existing repo-local retained roots required")
    core.require(not any(a != b and a.is_relative_to(b) for a in roots for b in roots),
                 "Overlapping retained roots would double-charge bytes")
    core.require(all(any(required.resolve().is_relative_to(root) for root in roots)
                     for required in required_retained_roots()), "Historical/control retained output omitted")
    return value, totals, roots


def aggregate_retained_bytes():
    core.require(_RETAINED_ROOTS is not None, "Historical retained-byte scope is not admitted")
    return sum(retained_bytes(root) for root in _RETAINED_ROOTS)


def _ledger_write(path, state):
    payload = json.dumps(state, sort_keys=True, allow_nan=False).encode()
    core.require(2 * len(payload) <= LEDGER_ALLOWANCE, "Reservation ledger exceeds its reserved overhead")
    temp = path.with_name(path.name + ".tmp-" + core.uuid.uuid4().hex)
    try:
        with temp.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temp.exists():
            temp.unlink()


def initialize_reservations():
    global _LAST_RETAINED_SCAN
    out = Path(_PLAN["output"])
    ledger = out / "RESOURCE-RESERVATIONS.json"
    core.require(not ledger.exists(), "Attempt capacity ledger exists; no implicit restart")
    current = retained_bytes(out)
    old = aggregate_retained_bytes() - current
    _ledger_write(ledger, dict(claims={}, old_retained_bytes=old, current_bytes=current))
    _LAST_RETAINED_SCAN = time.monotonic()


def reserved_retained_bytes(force_scan=False):
    """Shared current-output accounting; periodic old-root growth reconciliation.

    This is logical-byte accounting, not a filesystem quota. External writers to
    historical/control roots are forbidden by host admission and remain a scoped
    assumption between scans; keep their worst-case headroom in the review.
    """
    global _LAST_RETAINED_SCAN
    out = Path(_PLAN["output"])
    ledger = out / "RESOURCE-RESERVATIONS.json"
    with (out / "resource-reservations.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(ledger.read_text())
        clock = time.monotonic()
        due = force_scan or _LAST_RETAINED_SCAN is None or clock - _LAST_RETAINED_SCAN >= 30
        core.require(not force_scan or not state["claims"], "Final retained scan requires all writers closed")
        if due and not state["claims"]:
            # With the same lock, no normal writer can publish while scanning.
            current = retained_bytes(out)
            state["old_retained_bytes"] = aggregate_retained_bytes() - current
            state["current_bytes"] = current
            _ledger_write(ledger, state)
            _LAST_RETAINED_SCAN = clock
        return state["old_retained_bytes"] + state["current_bytes"] + LEDGER_ALLOWANCE + sum(
            item["bytes"] for item in state["claims"].values())


def capacity_projection(document):
    capacity = core.importlib.import_module("capacity").shape_plan(11738)
    return dict(new_checkpoint_upper_bytes=document["new_checkpoints"] * (
                    capacity["per_checkpoint_uncompressed_upper_bytes"] + 16384),
                all_interval_control_upper_bytes=capacity["full_intervals_upper_bytes"] + 256 * 1024**2,
                concurrent_verifier_scratch_bytes=4 * capacity["per_cell_scratch_bytes"],
                atomic_publication_extra_bytes=capacity["full_intervals_upper_bytes"] // 12 +
                    16 * capacity["per_checkpoint_uncompressed_upper_bytes"])


@contextmanager
def write_reservation(path, size, additional_paths=()):
    """Shared locked capacity claims, including parallel temporary publication."""
    path = Path(path).resolve()
    out = Path(_PLAN["output"]).resolve()
    core.require(path.is_relative_to(out) and path != out, "Write escapes this run")
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger = out / "RESOURCE-RESERVATIONS.json"
    claim = f"{os.getpid()}:{core.uuid.uuid4().hex}"
    lock_path = out / "resource-reservations.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        core.require(type(size) is int and size >= 0, "Invalid write reservation")
        core.require(ledger.is_file() and not ledger.is_symlink(), "Initialized capacity ledger missing; no reset/rebuild")
        state = json.loads(ledger.read_text())
        reserved = sum(item["bytes"] for item in state["claims"].values())
        core.require(_COOPERATIVE_HEADROOM is not None, "Reviewed cooperative write headroom missing")
        emergency = 0 if path.name in ("STOP-REQUEST.json", "FAILURE.json", "LIFECYCLE.json") else (
            int(_COOPERATIVE_HEADROOM["bytes"]))
        core.require(state["old_retained_bytes"] + state["current_bytes"] + LEDGER_ALLOWANCE + reserved + size
                     + emergency <= _LIMITS["max_output_bytes"],
                     "Concurrent output capacity claim refused")
        core.require(core.shutil.disk_usage(out).free >= reserved + size + _LIMITS["minimum_free_disk_bytes"],
                     "Concurrent disk reservation refused")
        paths = [path] + [Path(extra).resolve() for extra in additional_paths]
        core.require(all(p.is_relative_to(out) and not p.is_symlink() for p in paths), "Reservation companion path escape")
        before = {str(p): p.stat().st_size if p.exists() else 0 for p in paths}
        state["claims"][claim] = dict(bytes=size, before=before)
        _ledger_write(ledger, state)
    try:
        yield
    finally:
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(ledger.read_text())
            record = state["claims"].pop(claim)
            for name, old_size in record["before"].items():
                p = Path(name)
                core.require(not p.is_symlink(), "Reservation output symlink drift")
                state["current_bytes"] += (p.stat().st_size if p.exists() else 0) - old_size
            _ledger_write(ledger, state)


def replace_control(path, value):
    """Only this attempt's mutable monitor/claim ledger; never scientific receipts."""
    core.require(Path(path).name == "BUDGET-SAMPLE.json", "Unexpected replaceable control path")
    with write_reservation(path, len(json.dumps(value).encode()) + 1024):
        _ledger_write(Path(path), value)


def publish_json(path, value):
    size = len(json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()) + 1024
    with write_reservation(path, 2 * size):
        core.write_json(path, value)


def pins_unchanged(force_hash=False):
    """Cheap stat guard at hot ticks; full hashes at stages and every60s."""
    global _PIN_STATS, _LAST_PIN_HASH
    paths = [Path(core.__file__), Path(__file__), Path(__file__).with_name("s4_window_v3.py"),
             core.REPO / "rotcert/matching_parallel.py", Path(_PINS["plan_path"]), Path(_PINS["review_path"])]
    paths.extend(Path(item["path"]) for item in _PINS.get("extra_files", []))
    current_stats = {str(path): core.stat_id(path) for path in paths}
    if _PIN_STATS is not None:
        core.require(current_stats == _PIN_STATS, "Immutable plan/review/runner stat drift")
    clock = time.monotonic()
    if force_hash or _LAST_PIN_HASH is None or clock - _LAST_PIN_HASH >= 60:
        core.require(runner_hashes() == _PINS["runner_files"], "Reviewed runner drift during full lifecycle")
        for key in ("plan", "review"):
            core.require(core.sha(_PINS[key + "_path"]) == _PINS[key + "_sha256"],
                         "Immutable " + key + " changed during execution")
        for item in _PINS.get("extra_files", []):
            core.require(core.sha(item["path"]) == item["sha256"], "Schedule/history/evidence pin drift")
        core.require({str(path): core.stat_id(path) for path in paths} == current_stats,
                     "Pinned files changed during hash")
        _LAST_PIN_HASH = clock
    _PIN_STATS = current_stats


def allocate_scratch(path, dtype, shape):
    np = core.modules()["np"]
    size = math.prod(shape) * np.dtype(dtype).itemsize + 1024
    with write_reservation(path, size):
        with Path(path).open("xb") as stream:
            header = dict(descr=np.lib.format.dtype_to_descr(np.dtype(dtype)), fortran_order=False, shape=shape)
            np.lib.format.write_array_header_2_0(stream, header)
            required = stream.tell() + math.prod(shape) * np.dtype(dtype).itemsize
            core.require(hasattr(os, "posix_fallocate"), "Actual scratch disk preallocation unavailable")
            os.posix_fallocate(stream.fileno(), 0, required)
            stream.flush()
            os.fsync(stream.fileno())
        return np.lib.format.open_memmap(path, mode="r+")


def cell_key(model, tau):
    return f"{model}/iou{tau:.1f}"


# 2026-10-03 completion amendment (authorized by the author: "complete S4 per protocol"). The legacy root lacks
# 24,197 draws; the two user-authorized fresh windows of 2026-10-02 computed exactly those draws with the same
# frozen kernel, RNG, B and per-draw identity contract. They are a READ-ONLY overlay: every reused window
# checkpoint must pass the window runner's own identity/hash/exact-readback receipt check, its window must have
# no live owner, and resolve_checkpoint re-hashes each file before loading. Legacy draws always take precedence.
WINDOW_OVERLAY_ROOTS = (core.OUTPUT_ROOT / "two-hour-fresh-20261002T094200Z/attempt",
                        core.OUTPUT_ROOT / "final-20core-20261002T152132Z/attempt")
_WINDOW_FOUND = {}


def window_found(root):
    root = Path(root)
    core.require(root in WINDOW_OVERLAY_ROOTS, "Undeclared window overlay root")
    if root not in _WINDOW_FOUND:
        window = core.importlib.import_module("s4_window_v3")
        found, pending = window.previous_window(root)
        core.require(not pending, "Window overlay has incomplete checkpoint pairs or receipts")
        _WINDOW_FOUND[root] = (found, core.sha(root / "IDENTITY.json"))
    return _WINDOW_FOUND[root]


def window_overlay_records(model, tau, present):
    records = {}
    for root in WINDOW_OVERLAY_ROOTS:
        found, identity_sha = window_found(root)
        for (m, t, draw), rec in sorted(found.items(), key=lambda item: item[0][2]):
            if m != model or t != tau or str(draw) in present or str(draw) in records:
                continue
            path = Path(rec["path"])
            core.require(path == root / "checkpoints" / cell_key(model, tau) / (core.checkpoint_name(draw) + ".npz"),
                         "Window overlay path escape")
            before = {str(p): core.stat_id(p) for p in (path, path.with_suffix(".json"), path.with_suffix(".checked.json"))}
            core.require(core.sha(path) == rec["sha256"], "Window overlay checkpoint drift")
            core.require(all(core.stat_id(p) == value for p, value in before.items()),
                         "Window overlay checkpoint changed during inventory")
            records[str(draw)] = dict(path=str(path), sha256=rec["sha256"],
                                      sidecar_sha256=core.sha(path.with_suffix(".json")),
                                      receipt_sha256=rec["receipt_sha256"], science_sha256=rec["science_sha256"],
                                      window_root=str(root), window_identity_sha256=identity_sha,
                                      stat={name: list(value) for name, value in before.items()})
    return records


def checkpoint_inventory(model, tau):
    directory = core.LEGACY / cell_key(model, tau)
    records = {}
    # Retain gap/orphan checks rather than assume every filename is valid evidence.
    present = core.modules()["run_s4"].existing_prefix(directory, 5000)
    for draw in present:
        path = directory / (core.checkpoint_name(draw) + ".npz")
        before = {str(p): core.stat_id(p) for p in (path, path.with_suffix(".json"))}
        metadata = core.checked_json(path.with_suffix(".json"))
        expected = core.checkpoint_metadata(model, tau, draw)
        core.require(all(metadata.get(key) == value for key, value in expected.items()),
                     "Legacy checkpoint identity drift")
        core.require(metadata["file"] == path.name and core.sha(path) == metadata["sha256"],
                     "Legacy checkpoint hash drift")
        core.require(all(core.stat_id(p) == value for p, value in before.items()),
                     "Legacy checkpoint changed during inventory")
        records[str(draw)] = dict(path=str(path), sha256=metadata["sha256"],
                                 sidecar_sha256=core.sha(path.with_suffix(".json")),
                                 stat={name: list(value) for name, value in before.items()})
    records.update(window_overlay_records(model, tau, records))
    return records


def plan(args):
    core.verify_source()
    benchmark = core.checked_json(args.benchmark)
    core.require(benchmark["status"] == "PASS_EXACT_SERIAL_PARALLEL_BOUNDED_REAL_INPUTS" and
                 benchmark["runner_sha256"] == core.sha(core.__file__) and
                 benchmark["source_sha256"] == core.SOURCE_HASH and
                 benchmark["input_binding_sha256"] == core.INPUT_HASH,
                 "Current runner must pass real-input serial/parallel benchmark")
    workers = benchmark["selected_workers"]
    core.require(workers in (12, 16), "Measured multicore recommendation required")
    out = core.owned_output(args.out)
    core.require(not out.exists(), "New migration output required")
    out.mkdir(parents=True)
    binding = core.validate_binding()
    ids, roles = core.source_roles(binding)
    prepared = {}
    cells = []
    for model in core.modules()["config"].MODELS:
        for tau in core.modules()["config"].TAUS:
            records = checkpoint_inventory(model, tau)
            missing = [draw for draw in [-1] + list(range(5000)) if str(draw) not in records]
            for path in core.prepared_paths(model, tau):
                for file in (path, path.with_suffix(".json")):
                    prepared[str(file)] = core.sha(file)
            cells.append(dict(model=model, iou=tau, key=cell_key(model, tau),
                              reuse=records, missing_draws=missing))
            print(json.dumps(dict(stage="inventory", cell=cell_key(model, tau),
                                  reuse=len(records), missing=len(missing))), flush=True)
    core.verify_source()
    document = dict(schema="S4-multicore-completion-plan-v1", utc=core.utc(), B=5000,
                    source_sha256=core.SOURCE_HASH, index_sha256=core.INDEX_HASH,
                    input_binding_sha256=core.INPUT_HASH, workers=workers,
                    runner_files=runner_hashes(), benchmark_sha256=core.sha(args.benchmark),
                    benchmark_path=str(Path(args.benchmark).resolve()),
                    output=str(out), prepared=prepared, cells=cells,
                    checkpoint_policy="read-only old overlay; new missing draw checkpoints only",
                    scientific_protocol_unchanged=True,
                    execution_amendment=dict(previous_workers=1, new_workers=workers,
                        draw_assignment="fixed shared draw ID; never worker/model seeded",
                        native_threads_per_worker=1, affinity="one distinct physical core per worker",
                        RLIMIT_AS_per_worker_bytes=4 * core.GIB,
                        retained_memory_reserve_bytes=8 * core.GIB,
                        verification="every draw light plus base/0/1/2499/4999 exact deep; all endpoints intervals independently recomputed"),
                    legacy_active_unmodified=True, scientific_acceptance=False,
                    full_input_admission_required=True,
                    reusable_checkpoints=sum(len(cell["reuse"]) for cell in cells),
                    new_checkpoints=sum(len(cell["missing_draws"]) for cell in cells))
    document["capacity_projection"] = capacity_projection(document)
    document["formal_runtime_evidence_required"] = list(COMPONENTS)
    document["original_schedule"], _ = original_authority()
    document["cumulative_reconciliation_required"] = True
    if getattr(args, "accounting", None):
        document["accounting"] = dict(path=str(Path(args.accounting).resolve()), sha256=core.sha(args.accounting))
    core.write_json(out / "PLAN.json", document)
    print(json.dumps(dict(status="PLANNED_REVIEW_REQUIRED", workers=workers,
                          reusable_checkpoints=document["reusable_checkpoints"],
                          new_checkpoints=document["new_checkpoints"],
                          plan_sha256=core.sha(out / "PLAN.json"))), flush=True)


def validate_plan(document, review, plan_path, entry_utc=None):
    core.require(document["schema"] == "S4-multicore-completion-plan-v1" and document["B"] == 5000,
                 "Full5000 plan required")
    core.require(document["source_sha256"] == core.SOURCE_HASH and
                 document["index_sha256"] == core.INDEX_HASH and
                 document["input_binding_sha256"] == core.INPUT_HASH, "Scientific identity drift")
    core.require(document["runner_files"] == runner_hashes(), "Reviewed runner source drift")
    core.require(review.get("schema") == "S4-multicore-execution-review-v1" and
                 review.get("status") == "PASS" and review.get("plan_sha256") == core.sha(plan_path),
                 "Independent hash-bound execution review required")
    evidence_path = Path(review["evidence_path"])
    core.require(core.sha(evidence_path) == review["evidence_sha256"], "Review evidence drift")
    limits = review["limits"]
    for name in ("max_aggregate_cpu_seconds", "max_wall_seconds", "max_output_bytes",
                 "minimum_free_disk_bytes", "minimum_available_memory_bytes", "deadline_unix_seconds"):
        value = limits.get(name)
        core.require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                     "Finite positive limit required: " + name)
    core.require(limits["minimum_available_memory_bytes"] >= 8 * core.GIB and
                 limits["minimum_free_disk_bytes"] >= 2 * core.GIB, "Resource reserves below contract")
    core.require(limits["deadline_unix_seconds"] > time.time(), "Execution deadline expired")
    projection = review.get("projection", {})
    core.require(set(projection) == set(COMPONENTS), "Full lifecycle measured component projection required")
    for name, component in projection.items():
        core.require(component.get("measured_real_inputs") is True,
                     "Unmeasured formal runtime component: " + name)
        for metric in ("aggregate_cpu_seconds", "wall_seconds"):
            value = component.get(metric)
            core.require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                         "Finite positive projected " + name + ":" + metric)
        core.require(core.sha(component["evidence_path"]) == component["evidence_sha256"],
                     "Projection evidence changed: " + name)
    authority, original_limits = original_authority()
    core.require(document.get("original_schedule") == authority and
                 review.get("original_schedule") == authority,
                 "Exact original schedule hash binding required")
    for name in ("max_aggregate_cpu_seconds", "max_wall_seconds", "max_output_bytes", "deadline_unix_seconds"):
        core.require(limits[name] <= original_limits[name], "Review expands original frozen " + name)
    for name in ("minimum_free_disk_bytes", "minimum_available_memory_bytes"):
        core.require(limits[name] >= original_limits[name], "Review weakens original resource reserve")
    core.require(document.get("cumulative_reconciliation_required") is True and
                 document.get("accounting") == review.get("accounting") and document.get("accounting"),
                 "Hash-bound cumulative reconciliation required")
    _, history, roots = validate_accounting(document["accounting"], time.time() if entry_utc is None else entry_utc)
    for group in ("cooperative_headroom", "terminal_reserve"):
        reserve = review.get(group, {})
        for name in ("cpu_seconds", "wall_seconds", "bytes"):
            core.require(type(reserve.get(name)) in (int, float) and math.isfinite(reserve[name]) and reserve[name] > 0,
                         "Finite positive " + group + ":" + name + " required")
    cpu_projected = 1.5 * sum(item["aggregate_cpu_seconds"] for item in projection.values())
    wall_projected = 1.5 * sum(item["wall_seconds"] for item in projection.values())
    cpu_reserve = sum(review[group]["cpu_seconds"] for group in ("cooperative_headroom", "terminal_reserve"))
    wall_reserve = sum(review[group]["wall_seconds"] for group in ("cooperative_headroom", "terminal_reserve"))
    core.require(limits["max_aggregate_cpu_seconds"] >= history["cpu_seconds"] + cpu_projected + cpu_reserve and
                 limits["max_wall_seconds"] >= history["wall_seconds"] + wall_projected + wall_reserve and
                 limits["deadline_unix_seconds"] >= time.time() + wall_projected + wall_reserve,
                 "Budget/deadline cannot cover complete measured formal lifecycle")
    expected = [(model, tau) for model in core.modules()["config"].MODELS
                for tau in core.modules()["config"].TAUS]
    core.require([(cell["model"], cell["iou"]) for cell in document["cells"]] == expected,
                 "12 scientific cells required, no selection")
    for cell in document["cells"]:
        core.require(cell["key"] == cell_key(cell["model"], cell["iou"]), "Canonical cell path required")
        reuse = {int(draw) for draw in cell["reuse"]}
        missing = cell["missing_draws"]
        core.require(len(missing) == len(set(missing)) and not reuse.intersection(missing) and
                     reuse.union(missing) == set([-1] + list(range(5000))), "Full draw coverage plan drift")
    core.require(document["reusable_checkpoints"] == sum(len(cell["reuse"]) for cell in document["cells"]) and
                 document["new_checkpoints"] == sum(len(cell["missing_draws"]) for cell in document["cells"]),
                 "Current workload totals must agree with complete partition")
    core.require(document["workers"] in (12, 16), "Reviewed 12/16 physical compute workers required")
    required_prepared = {str(file) for model, tau in expected for path in core.prepared_paths(model, tau)
                         for file in (path, path.with_suffix(".json"))}
    core.require(set(document["prepared"]) == required_prepared, "Exact72prepared NPZ/sidecar pairs required")
    capacity = capacity_projection(document)
    core.require(document["capacity_projection"] == capacity and limits["max_output_bytes"] >=
                 sum(retained_bytes(root) for root in roots) + sum(capacity.values()) +
                 review["cooperative_headroom"]["bytes"] + review["terminal_reserve"]["bytes"],
                 "Historical+control+full concurrent capacity under-budgeted")
    resource_proof = review.get("resource_proof", {})
    core.require(resource_proof.get("current_host_cgroup_reviewed") is True and
                 core.sha(resource_proof["evidence_path"]) == resource_proof["evidence_sha256"],
                 "Actual host/cgroup memory+quota review required, unknown is not unlimited")
    return limits


def resolve_checkpoint(cell, draw, out):
    expected = core.checkpoint_metadata(cell["model"], cell["iou"], draw)
    if str(draw) in cell["reuse"] and "window_root" in cell["reuse"][str(draw)]:
        record = cell["reuse"][str(draw)]
        path, root = Path(record["path"]), Path(record["window_root"])
        core.require(root in WINDOW_OVERLAY_ROOTS and
                     path == root / "checkpoints" / cell_key(cell["model"], cell["iou"]) / path.name and
                     path.name == core.checkpoint_name(draw) + ".npz", "Window overlay path escape")
        core.require(core.sha(root / "IDENTITY.json") == record["window_identity_sha256"], "Window identity drift")
        core.require(core.sha(path.with_suffix(".checked.json")) == record["receipt_sha256"], "Window receipt drift")
        core.require(core.sha(path.with_suffix(".json")) == record["sidecar_sha256"], "Window overlay sidecar drift")
        core.require(core.sha(path) == record["sha256"], "Window overlay checkpoint drift")
    elif str(draw) in cell["reuse"]:
        record = cell["reuse"][str(draw)]
        path = Path(record["path"])
        required_path = core.LEGACY / cell_key(cell["model"], cell["iou"]) / path.name
        core.require(path == required_path and path.name == core.checkpoint_name(draw) + ".npz",
                     "Legacy overlay path escape")
        core.require(core.sha(path.with_suffix(".json")) == record["sidecar_sha256"],
                     "Frozen overlay sidecar drift")
        core.require(core.sha(path) == record["sha256"], "Frozen overlay checkpoint drift")
    else:
        path = core.owned_output(out / cell["key"] / (core.checkpoint_name(draw) + ".npz"))
    return core.modules()["io_s4"].load_npz(path, expected)


def process_identity(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    fields = text[text.rfind(")") + 2:].split()
    return dict(pid=pid, start_ticks=fields[19], state=fields[0], ppid=int(fields[1]),
                cpu_seconds=(int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK"))


def host_admission():
    """Fail closed in an ephemeral PID namespace; never inspect process environments."""
    core.require(Path("/proc/1/comm").read_text().strip() == "systemd",
                 "Persistent execution requires host PID namespace")
    status = Path("/proc/self/status").read_text().splitlines()
    nspid = next((line.split()[1:] for line in status if line.startswith("NSpid:")), None)
    core.require(nspid is not None and len(nspid) == 1 and nspid[0].isdigit(),
                 "Nested or unknown PID namespace refused")
    own = {os.getpid()} | {child.pid for child in mp.active_children()}
    names = {"s4_multicore.py", "s4_multicore_full.py", "run_s4.py", "verify_s4.py",
             "run_s4_recovery.py", "launch_recovery2.py", "local_s4_pilot_budget_v2.py"}
    hits = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit() or int(path.name) in own:
            continue
        try:
            args = [(part.decode(errors="replace")) for part in (path / "cmdline").read_bytes().split(b"\0") if part]
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as error:
            raise ValueError("Cannot establish singleton: unreadable process command lines") from error
        if any(Path(arg).name in names for arg in args) or (
                any(Path(arg).name == "resources.py" for arg in args) and "--watch" in args):
            hits.append(int(path.name))
    core.require(not hits, "Competing S4 worker/monitor detected: " + repr(hits))
    return dict(pid=os.getpid(), boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                start_ticks=process_identity(os.getpid())["start_ticks"], host_pid_namespace=True)


@contextmanager
def atomic_execution_claim(out):
    """One shared full-experiment lock and a durable one-shot attempt claim.

    The lock is new to this runner, never a cleared or rewritten legacy lock.
    Its shared inherited open-file description also remains held by pool workers.
    """
    global _CLAIM_FD
    out = core.owned_output(out)
    flags = os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(core.OUTPUT_ROOT / ".s4-full-execution.lock", flags, 0o600)
    acquired = False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another S4 full execution holds the singleton") from error
        acquired = True
        claim = out / "EXECUTION-CLAIM.json"
        core.require(not claim.exists() and not any((out / name).exists() for name in
                     ("ADMISSION.json", "LIFECYCLE.json", "FULL.json", "TERMINAL-COMMIT.json")),
                     "Already attempted migration; preserve evidence, no retry")
        identity = host_admission()
        core.write_json(claim, dict(schema="S4-one-shot-execution-claim-v1", **identity,
                                   utc=core.utc(), automatic_retry=False))
        _CLAIM_FD = fd
        yield identity
    finally:
        _CLAIM_FD = None
        # Do not LOCK_UN: workers share this open description and must retain it
        # until they also close, including after abrupt parent disappearance.
        os.close(fd)


def read_budget_sample(path):
    """Read the supervisor's mutable budget sample through one open file description.

    The parent republishes it about once a second by atomic replace, so a single descriptor always sees one complete
    version. A stat-before/stat-after identity check (checked_json) treats such a replace as a torn read; it stopped
    the second formal attempt (2026-10-04) after 4.5 min of verification.
    """
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd, "rb") as stream:
        return json.loads(stream.read())


class CooperativeGuard:
    """The complete frozen resources interface, including inner geometry ticks."""
    def __init__(self, parent=False):
        self.parent = parent
        self.last = -math.inf

    def check(self, stage=None, allocation=0):
        # Geometry ticks are frequent. STOP and parent identity always have a
        # cheap boundary; larger resource/stat reads run at most every100ms.
        out = Path(_PLAN["output"])
        core.require(_STOP_EVENT is None or not _STOP_EVENT.is_set(), "Cooperative stop requested")
        core.require(not (out / "STOP-REQUEST.json").exists(), "Cooperative stop requested")
        if not self.parent and _PARENT_ID is not None:
            current = process_identity(_PARENT_ID["pid"])
            core.require(current is not None and current["start_ticks"] == _PARENT_ID["start_ticks"] and
                         current["state"] != "Z", "Supervising parent identity disappeared")
        clock = time.monotonic()
        if not allocation and clock - self.last < .10:
            return
        self.last = clock
        if self.parent:
            parent_boundary()
        else:
            full_boundary()
            sample = read_budget_sample(out / "BUDGET-SAMPLE.json")
            # Compare with a clock read after the sample. The parent stamps before it publishes, so a sample
            # published during full_boundary() is never newer than this reading; the earlier throttle clock made
            # such a sample look "future" and stopped attempts 3 and 4 (2026-10-04).
            now = time.monotonic()
            core.require(sample["execution_plan_sha256"] == _PINS["plan_sha256"] and
                         now - sample["monotonic"] <= SAMPLE_MAX_AGE and now >= sample["monotonic"],
                         "Supervisor cumulative budget sample missing/stale")
            core.require(sample["aggregate_cpu_seconds"] < _LIMITS["max_aggregate_cpu_seconds"] and
                         sample["wall_seconds"] < _LIMITS["max_wall_seconds"], "Cumulative worker budget reached")
        core.require(core.shutil.disk_usage(out).free >= _LIMITS["minimum_free_disk_bytes"] + allocation,
                     "Safe-boundary allocation disk reserve refused")

    def before_write(self, path, size):
        path = Path(path).resolve()
        core.require(path.is_relative_to(Path(_PLAN["output"]).resolve()), "Frozen write escapes this run")
        # Full runner writes use shared reservations, not uncoordinated frozen IO.
        raise ValueError("Unreserved frozen IO write refused; use shared write_reservation")

    def after_write(self, path):
        core.require(Path(path).resolve().is_relative_to(Path(_PLAN["output"]).resolve()),
                     "Frozen write escapes this run")
        self.check()


def full_worker_init(cpus, slots, document, limits, inputs_ids, pins, parent_id=None, lock_socket=None, barrier=None, stop_event=None, headroom=None):
    global _PLAN, _LIMITS, _INPUTS, _PINS, _PIN_STATS, _LAST_PIN_HASH, _PARENT_ID, _CLAIM_FD, _READY_BARRIER, _STOP_EVENT
    global _COOPERATIVE_HEADROOM
    with slots.get_lock():
        slot = slots.value
        slots.value += 1
    core.require(slot < len(cpus), "Unexpected worker replacement")
    os.sched_setaffinity(0, {cpus[slot]})
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (min(4 * core.GIB, hard) if hard >= 0 else 4 * core.GIB, hard))
    core.require(all(os.environ.get(key) == "1" for key in core.THREAD_KEYS), "Native thread drift")
    _PLAN, _LIMITS, _PINS = document, limits, pins
    _PARENT_ID = parent_id
    _READY_BARRIER = barrier
    _STOP_EVENT = stop_event
    core.require(headroom is not None and all(type(headroom.get(key)) in (int, float) and
                 math.isfinite(headroom[key]) and headroom[key] > 0 for key in ("cpu_seconds", "wall_seconds", "bytes")),
                 "Reviewed cooperative worker headroom required")
    _COOPERATIVE_HEADROOM = dict(headroom)
    if lock_socket is not None:
        from multiprocessing.reduction import recv_handle
        _CLAIM_FD = recv_handle(lock_socket)
        lock_socket.close()
    _PIN_STATS, _LAST_PIN_HASH = None, None
    pins_unchanged(force_hash=True)
    _INPUTS = type("BoundInput", (), {"ids": inputs_ids})()
    core.modules()
    resources = core.importlib.import_module("resources")
    core.require(resources.CURRENT is None, "No nested frozen resource guard")
    resources.CURRENT = CooperativeGuard()
    core.verify_source()


def worker_ready(_):
    _READY_BARRIER.wait(timeout=30)
    full_boundary()
    return dict(health=core.health())


def full_boundary():
    pins_unchanged()
    out = Path(_PLAN["output"])
    core.require(_STOP_EVENT is None or not _STOP_EVENT.is_set(), "Cooperative stop requested")
    core.require(not (out / "STOP-REQUEST.json").exists(), "Cooperative stop requested")
    core.require(time.time() < _LIMITS["deadline_unix_seconds"], "Absolute deadline reached")
    core.require(core.available_memory() >= _LIMITS["minimum_available_memory_bytes"], "Memory reserve reached")
    core.require(core.shutil.disk_usage(out).free >= _LIMITS["minimum_free_disk_bytes"], "Disk reserve reached")


def compute_missing(task):
    cell, draws = task
    full_boundary()
    data, guards = core.load_data(cell["model"], cell["iou"])
    binding = core.validate_binding()
    _, roles = core.source_roles(binding)
    pipe = core.modules()["statistics_core"].ModelPipeline(data, roles)
    start_cpu, start_wall = time.process_time(), time.monotonic()
    out = Path(_PLAN["output"]) / cell["key"]
    for draw in draws:
        full_boundary()
        core.require(all(core.stat_id(p) == identity for p, identity in guards.items()), "Prepared stat drift")
        dt = time.process_time()
        mult = core.modules()["statistics_core"].multiplicities(roles, draw)
        result = pipe.run(mult)
        result["multiplicity"] = mult
        result["pipeline_cpu_seconds"] = core.modules()["np"].asarray(time.process_time() - dt)
        core.modules()["run_s4"].validate_saved(result, roles, draw)
        core.modules()["verify_s4"].verify_one(result, _INPUTS, data, draw, False)
        path = out / (core.checkpoint_name(draw) + ".npz")
        estimated = sum(array.nbytes for array in result.values()) + 65536
        with write_reservation(path, 2 * estimated, additional_paths=(path.with_suffix(".json"),)):
            core.save_arrays(path, result, dict(core.checkpoint_metadata(cell["model"], cell["iou"], draw),
                                               execution_contract="multicore-completion-20261001-v1"))
    return dict(cell=cell["key"], draws=len(draws), cpu_seconds=time.process_time() - start_cpu,
                wall_seconds=time.monotonic() - start_wall, health=core.health())


def verify_cell(cell):
    """One bounded-memory cell, independent oracle and unchanged CI semantics."""
    full_boundary()
    m = core.modules()
    intervals = core.importlib.import_module("intervals")
    np = m["np"]
    out = core.owned_output(Path(_PLAN["output"]) / cell["key"])
    out.mkdir(parents=True, exist_ok=True)
    data, guards = core.load_data(cell["model"], cell["iou"])
    started_cpu, started_wall = time.process_time(), time.monotonic()
    stacks, scratch = {}, []
    paired = {}
    deep_count = 0
    for draw in [-1] + list(range(5000)):
        full_boundary()
        result = resolve_checkpoint(cell, draw, Path(_PLAN["output"]))
        deep = draw in (-1, 0, 1, 2499, 4999)
        m["verify_s4"].verify_one(result, _INPUTS, data, draw, deep)
        deep_count += int(deep)
        paired[str(draw)] = core.hashlib.sha256(result["multiplicity"].tobytes()).hexdigest()
        if draw >= 0:
            if not stacks:
                for key in ("values", "macro", "contrasts", "macro_contrasts", "all_source"):
                    path = out / ("scratch-" + key + ".npy")
                    stacks[key] = allocate_scratch(path, result[key].dtype, (5000,) + result[key].shape)
                    scratch.append(path)
            for key in stacks:
                stacks[key][draw] = result[key]
    report = {}
    for key, values in stacks.items():
        per = {}
        for ix in np.ndindex(values.shape[1:]):
            full_boundary()
            core.importlib.import_module("resources").tick("verification_intervals")
            level = intervals.coverage_level(cell["iou"], ix[1], m["config"].CONTRASTS[ix[0]]) if key == "macro_contrasts" and ix[-1] == 0 else .95
            vector = values[(slice(None),) + ix]
            got = intervals.interval(vector, level)
            reference = m["verify_s4"].ref_interval(vector, level)
            m["verify_s4"].check_interval(reference, got)
            if key == "macro_contrasts" and ix[-1] == 0:
                got["numerical_resolution"] = intervals.contrast_numerical_status(got)
            per["/".join(map(str, ix))] = got
        report[key] = per
    for path, identity in guards.items():
        core.require(core.stat_id(path) == identity, "Prepared drift during verification")
    core.verify_source()
    interval_path = out / "INTERVALS.json"
    publish_json(interval_path, m["io_s4"].clean(dict(source_sha256=core.SOURCE_HASH,
        input_binding_sha256=core.INPUT_HASH, model=cell["model"], iou=cell["iou"], arrays=report,
        metrics=m["config"].METRICS,
        interpretation="Marginal exploratory except prespecified HCP16 family approximate99.6875percent; no cross-family simultaneous claim")))
    # Delete only this task's listed scratch after durable interval publication.
    for array in stacks.values():
        array.flush()
        array._mmap.close()
    scratch_hashes = {path.name: core.sha(path) for path in scratch}
    for path in scratch:
        core.require(path.parent == out and path.name.startswith("scratch-") and path.suffix == ".npy",
                     "Scratch ownership mismatch")
        with write_reservation(path, 0):
            path.unlink()
    result = dict(cell=cell["key"], checkpoints=5001, deep_draws=deep_count,
                  all_endpoint_domain_checks=True, independent_exact_recalibration=True,
                  all_interval_independent_recompute=True, interval_sha256=core.sha(interval_path),
                  scratch_hashes=scratch_hashes, paired_weight_hashes=paired,
                  cpu_seconds=time.process_time() - started_cpu,
                  wall_seconds=time.monotonic() - started_wall, health=core.health())
    publish_json(out / "CELL-VERIFIED.json", result)
    return result


def cumulative_sample():
    """Charge the saved debit once, reaped CPU once, and live descendants once.

    Only this interpreter's descendant tree is read; no environment/argv dumps.
    Inconsistent process snapshots reject instead of silently dropping CPU.
    """
    core.require(_HISTORY is not None and _START_CPU is not None and _START_WALL is not None,
                 "Cumulative historical budget is not admitted")
    before_child = resource.getrusage(resource.RUSAGE_CHILDREN)
    before_reaped = before_child.ru_utime + before_child.ru_stime
    snapshots = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            identity = process_identity(int(path.name))
            if identity is not None:
                snapshots[identity["pid"]] = identity
    descendants, parents = [], {os.getpid()}
    while True:
        children = [item for pid, item in snapshots.items() if pid not in parents and item["ppid"] in parents]
        if not children:
            break
        descendants.extend(children)
        parents.update(item["pid"] for item in children)
    # rusage includes only waited-for children. A live/zombie not reaped by the
    # parent is additional CPU; if reaping races, retry rather than double-count.
    live_cpu = 0.
    for identity in descendants:
        current = process_identity(identity["pid"])
        if current is not None:
            core.require(current["start_ticks"] == identity["start_ticks"], "Descendant PID identity drift")
            live_cpu += current["cpu_seconds"]
    after_child = resource.getrusage(resource.RUSAGE_CHILDREN)
    after_reaped = after_child.ru_utime + after_child.ru_stime
    core.require(before_reaped == after_reaped, "Child CPU reaping raced budget sample; retry admission")
    own = resource.getrusage(resource.RUSAGE_SELF)
    # Reuse the CHILDREN snapshot whose consistency was just proved. Reading it
    # a third time can reap a counted-live child between the check and the sum.
    after = own.ru_utime + own.ru_stime + after_reaped
    used_cpu = _HISTORY["cpu_seconds"] + after - _START_CPU + live_cpu
    return dict(aggregate_cpu_seconds=used_cpu,
                wall_seconds=_HISTORY["wall_seconds"] + time.monotonic() - _START_WALL,
                monotonic=time.monotonic(), execution_plan_sha256=_PINS["plan_sha256"],
                reaped_and_parent_delta_cpu_seconds=after - _START_CPU,
                live_descendant_cpu_seconds=live_cpu,
                retained_output_bytes=reserved_retained_bytes())


def check_sample(sample, reserve=None):
    reserve = reserve or dict(cpu_seconds=0, wall_seconds=0, bytes=0)
    core.require(sample["aggregate_cpu_seconds"] + reserve["cpu_seconds"] < _LIMITS["max_aggregate_cpu_seconds"] and
                 sample["wall_seconds"] + reserve["wall_seconds"] < _LIMITS["max_wall_seconds"] and
                 sample["retained_output_bytes"] + reserve["bytes"] < _LIMITS["max_output_bytes"] and
                 time.time() + reserve["wall_seconds"] < _LIMITS["deadline_unix_seconds"],
                 "Cumulative full lifecycle budget reached")


def sample_budget(reserve=None):
    global _LAST_SAMPLE, _SAMPLING, _LAST_HOST_CHECK
    core.require(not _SAMPLING, "Recursive budget sampling refused")
    _SAMPLING = True
    try:
        clock = time.monotonic()
        parts = {}
        if _LAST_HOST_CHECK is None or clock - _LAST_HOST_CHECK >= 30:
            host_admission()
            _LAST_HOST_CHECK = clock
            parts["host_admission"] = time.monotonic() - clock
        mark = time.monotonic()
        sample = cumulative_sample()
        parts["cumulative_sample"] = time.monotonic() - mark
        check_sample(sample, reserve)
        mark = time.monotonic()
        replace_control(Path(_PLAN["output"]) / "BUDGET-SAMPLE.json", sample)
        parts["replace_control"] = time.monotonic() - mark
        # Diagnostics only: record any slow sample or long gap since the previous one in the run log.
        gap = None if _LAST_SAMPLE is None else sample["monotonic"] - _LAST_SAMPLE["monotonic"]
        if time.monotonic() - clock > SLOW_SAMPLE_SECONDS or (gap is not None and gap > SLOW_SAMPLE_SECONDS):
            print(json.dumps(dict(stage="slow_budget_sample", utc=core.utc(), gap_seconds=gap,
                                  parts={k: round(v, 3) for k, v in parts.items()})), flush=True)
        _LAST_SAMPLE = sample
        return sample
    finally:
        _SAMPLING = False


def request_stop(out, error):
    if _STOP_EVENT is not None:
        _STOP_EVENT.set()
    stop = Path(out) / "STOP-REQUEST.json"
    if not stop.exists():
        publish_json(stop, dict(reason=repr(error), utc=core.utc()))


def monitor_pool(pool, tasks, function, out, limits, started_cpu, started_wall):
    responses, pending = [], set()
    iterator = iter(tasks)
    exhausted = object()
    # No unbounded full-bootstrap submit queue; at most 2*workers tasks outstanding.
    try:
        for _ in range(2 * pool._max_workers):
            task = next(iterator, exhausted)
            if task is exhausted:
                break
            pending.add(pool.submit(function, task))
            # A spawn-context submit may start a worker and block this thread while the
            # initargs are pickled and read (about 0.3 s each with the full plan). Refresh
            # the shared sample after every submit so workers that are already initializing
            # never see it older than SAMPLE_MAX_AGE (stop of 2026-10-04: 16 spawns, 5.7 s).
            sample_budget(_COOPERATIVE_HEADROOM)
        while pending:
            pins_unchanged()
            done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                responses.append(future.result())
                task = next(iterator, exhausted)
                if task is not exhausted:
                    pending.add(pool.submit(function, task))
            sample_budget(_COOPERATIVE_HEADROOM)
    except BaseException as error:
        request_stop(out, error)
        for future in pending:
            future.cancel()
        raise
    return responses


def parent_boundary():
    full_boundary()
    sample_budget(_COOPERATIVE_HEADROOM)


def assert_worker_health(responses, allowed_cpus):
    observed = {}
    for response in responses:
        value = response["health"]
        core.require(value["threads"] == 1 and len(value["cpu_affinity"]) == 1 and
                     value["cpu_affinity"][0] in allowed_cpus and
                     all(value["blas_threads"][key] == "1" for key in core.THREAD_KEYS),
                     "Actual worker affinity/native-thread proof failed")
        if value["pid"] in observed:
            core.require(observed[value["pid"]] == value["cpu_affinity"], "PID affinity drift")
        observed[value["pid"]] = value["cpu_affinity"]
    core.require(len({tuple(cpus) for cpus in observed.values()}) == len(observed),
                 "Distinct workers share a compute CPU")
    core.require(len(observed) == len(allowed_cpus) and
                 {value[0] for value in observed.values()} == set(allowed_cpus),
                 "Exactly the requested worker count/CPU set must be observed")
    return observed


def run_pool(tasks, function, cpus, inputs_ids):
    """Initialize and prove every worker before submitting scientific tasks."""
    from multiprocessing.reduction import send_handle
    ctx = mp.get_context("spawn")
    slots = ctx.Value("i", 0)
    barrier = ctx.Barrier(len(cpus))
    sender, receiver = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        for _ in cpus:
            send_handle(sender, _CLAIM_FD, os.getpid())
        sample_budget(_COOPERATIVE_HEADROOM)
        with core.single_thread_spawn(), ProcessPoolExecutor(max_workers=len(cpus), mp_context=ctx,
                initializer=full_worker_init,
                initargs=(cpus, slots, _PLAN, _LIMITS, inputs_ids, _PINS, _PARENT_ID, receiver, barrier, _STOP_EVENT,
                          _COOPERATIVE_HEADROOM)) as pool:
            ready = monitor_pool(pool, [None] * len(cpus), worker_ready, Path(_PLAN["output"]),
                                 _LIMITS, _START_CPU, _START_WALL)
            actual = assert_worker_health(ready, cpus)
            result = monitor_pool(pool, tasks, function, Path(_PLAN["output"]),
                                  _LIMITS, _START_CPU, _START_WALL)
            core.require(all(item["health"]["pid"] in actual for item in result),
                         "Unobserved/replacement pool worker refused")
        return result, actual
    finally:
        sender.close()
        receiver.close()


def terminal_acceptance(out):
    """Read-only acceptance of the *whole* terminal contract, never FULL alone."""
    out = Path(out)
    core.require(not any((out / name).exists() for name in ("STOP-REQUEST.json", "FAILURE.json")),
                 "Stopped/failed execution cannot have terminal acceptance")
    commit = core.checked_json(out / "TERMINAL-COMMIT.json")
    core.require(commit.get("status") == "COMPLETE_WITH_CHARGED_TERMINAL_RESERVE" and
                 commit.get("schema") == "S4-terminal-commit-v1", "Final terminal commit absent/incomplete")
    for name, digest in commit["receipt_sha256"].items():
        core.require(name in ("FULL.json", "LIFECYCLE.json", "VERIFIED.json") and
                     core.sha(out / name) == digest, "Terminal receipt hash drift")
    core.require(set(commit["receipt_sha256"]) == {"FULL.json", "LIFECYCLE.json", "VERIFIED.json"},
                 "Complete terminal receipt set required")
    lifecycle = core.checked_json(out / "LIFECYCLE.json")
    result = core.checked_json(out / "FULL.json")
    core.require(lifecycle["status"] == "SEALED_PENDING_TERMINAL_COMMIT" and
                 result["status"] == "VERIFIED_PENDING_TERMINAL_COMMIT" and
                 lifecycle["execution_plan_sha256"] == result["execution_plan_sha256"] == commit["execution_plan_sha256"],
                 "Terminal lifecycle/full/plan mismatch")
    return commit


def closeout(document, workers):
    """Stage non-accepting receipts, account their actual writes, then commit last.

    The finite last-link/fsync/return cost is charged conservatively in advance.
    It is a reviewed contingent reserve, not an unmeasured hard latency guarantee.
    """
    out = Path(document["output"])
    reserved_retained_bytes(force_scan=True)
    parent_boundary()
    pins_unchanged(force_hash=True)
    common = dict(source_sha256=core.SOURCE_HASH, input_binding_sha256=core.INPUT_HASH,
                  execution_plan_sha256=_PINS["plan_sha256"], utc=core.utc(),
                  legacy_receipts_unmodified=True, scientific_synthesis_acceptance=False)
    publish_json(out / "FULL.json", dict(common, status="VERIFIED_PENDING_TERMINAL_COMMIT",
        B=5000, cells=12, workers=workers, reused_checkpoints=document["reusable_checkpoints"],
        new_checkpoints=document["new_checkpoints"]))
    parent_boundary()
    sample = cumulative_sample()
    publish_json(out / "LIFECYCLE.json", dict(common, status="SEALED_PENDING_TERMINAL_COMMIT",
        aggregate_cpu_seconds=sample["aggregate_cpu_seconds"], wall_seconds=sample["wall_seconds"],
        history=_HISTORY, retained_output_bytes=sample["retained_output_bytes"],
        resource_scope="saved debit once + all entry/preflight/children/compute/verify/closeout costs",
        terminal_reserve=_TERMINAL_RESERVE))
    parent_boundary()
    receipts = {name: core.sha(out / name) for name in ("FULL.json", "LIFECYCLE.json", "VERIFIED.json")}
    # The candidate is not accepted on disk until the atomic link below. All
    # pre-publication work, including capacity ledger operations, is measured.
    candidate = dict(common, schema="S4-terminal-commit-v1", status="COMPLETE_WITH_CHARGED_TERMINAL_RESERVE",
        receipt_sha256=receipts, terminal_reserve=_TERMINAL_RESERVE,
        resource_accounting="Measured before final publication + reviewed finite terminal reserve; conditional upper, not exact",
        cumulative_debit_before_commit=cumulative_sample())
    payload = (json.dumps(candidate, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path = out / "TERMINAL-COMMIT.json"
    temp = out / (".terminal-candidate-" + core.uuid.uuid4().hex)
    linked = False
    try:
        # Reservation covers candidate, final inode, failure invalidation and
        # ledger temp. Last cleanup is explicitly inside the charged reserve.
        with write_reservation(path, 2 * len(payload) + int(_TERMINAL_RESERVE["bytes"]), additional_paths=(temp,)):
            with temp.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            core.require(core.checked_json(temp) == candidate, "Terminal candidate readback drift")
            full_boundary()
            before = cumulative_sample()
            check_sample(before, _TERMINAL_RESERVE)
            os.link(temp, path)
            linked = True
            temp.unlink()
            fd = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        after = cumulative_sample()
        check_sample(after)
        core.require(after["aggregate_cpu_seconds"] - before["aggregate_cpu_seconds"] <= _TERMINAL_RESERVE["cpu_seconds"] and
                     after["wall_seconds"] - before["wall_seconds"] <= _TERMINAL_RESERVE["wall_seconds"],
                     "Reviewed terminal reserve exceeded")
        terminal_acceptance(out)
        final = cumulative_sample()
        check_sample(final)
        core.require(final["aggregate_cpu_seconds"] - before["aggregate_cpu_seconds"] <= _TERMINAL_RESERVE["cpu_seconds"] and
                     final["wall_seconds"] - before["wall_seconds"] <= _TERMINAL_RESERVE["wall_seconds"],
                     "Terminal readback/return reserve exceeded")
    except BaseException:
        if linked:
            # A failed new commit is quarantined, not discarded. All previously
            # produced scientific data and receipts remain untouched.
            os.rename(path, out / ("FAILED-TERMINAL-COMMIT-" + core.uuid.uuid4().hex + ".json"))
        raise
    return candidate


def execute(args):
    global _PLAN, _LIMITS, _PINS, _START_CPU, _START_WALL, _PIN_STATS, _LAST_PIN_HASH
    global _HISTORY, _RETAINED_ROOTS, _PARENT_ID, _COOPERATIVE_HEADROOM, _TERMINAL_RESERVE, _STOP_EVENT
    # Charge this interpreter's imports/entry and every preflight, not only the
    # later scientific loop. Reaped child CPU from prior operations is excluded.
    _START_CPU = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime
    _START_WALL = _ENTRY_WALL
    entry_utc = _ENTRY_UTC
    document = core.checked_json(args.plan)
    review = core.checked_json(args.review)
    limits = validate_plan(document, review, args.plan, entry_utc)
    out = core.owned_output(document["output"])
    core.require(Path(args.plan).resolve() == out / "PLAN.json", "Plan output binding mismatch")
    _, _HISTORY, _RETAINED_ROOTS = validate_accounting(document["accounting"], entry_utc)
    _COOPERATIVE_HEADROOM = review["cooperative_headroom"]
    _TERMINAL_RESERVE = review["terminal_reserve"]
    workers = document["workers"]
    cpus = core.affinity_plan(workers)["worker_cpus"]
    core.require(core.available_memory() >= workers * 2 * core.GIB + limits["minimum_available_memory_bytes"],
                 "Full lifecycle memory admission failed")
    core.require(core.shutil.disk_usage(out).free >= limits["max_output_bytes"] + limits["minimum_free_disk_bytes"],
                 "Full output+scratch disk admission failed")
    _PLAN, _LIMITS = document, limits
    _PINS = dict(runner_files=document["runner_files"],
                 plan_path=str(Path(args.plan).resolve()), plan_sha256=core.sha(args.plan),
                 review_path=str(Path(args.review).resolve()), review_sha256=core.sha(args.review),
                 extra_files=[document["original_schedule"], document["accounting"],
                    dict(path=str(core.LEGACY / "attempts/recovery-2/LIFECYCLE.json"), sha256=SAVED_LIFECYCLE_HASH),
                    dict(path=review["evidence_path"], sha256=review["evidence_sha256"]),
                    dict(path=review["resource_proof"]["evidence_path"], sha256=review["resource_proof"]["evidence_sha256"])] +
                    [dict(path=item["evidence_path"], sha256=item["evidence_sha256"]) for item in review["projection"].values()])
    _PIN_STATS, _LAST_PIN_HASH = None, None
    pins_unchanged(force_hash=True)
    _STOP_EVENT = mp.get_context("spawn").Event()
    with atomic_execution_claim(out) as identity:
        _PARENT_ID = identity
        initialize_reservations()
        sample_budget(_COOPERATIVE_HEADROOM)
        return execute_claimed(document, limits, workers, cpus, args)


def execute_claimed(document, limits, workers, cpus, args):
    out = Path(document["output"])
    core.modules()
    resource_module = core.importlib.import_module("resources")
    core.require(resource_module.CURRENT is None, "No nested legacy resource lifecycle allowed")
    resource_module.CURRENT = CooperativeGuard(parent=True)
    try:
        core.verify_source()
        print(json.dumps(dict(stage="full_input_admission", utc=core.utc())), flush=True)
        inputs = core.modules()["inputs"].InputSet(core.INDEX, synthetic=False)
        core.require(inputs.binding_hash == core.INPUT_HASH, "Full InputSet identity drift")
        inputs.guard.check(full=True)
        for path, digest in document["prepared"].items():
            core.require(core.sha(path) == digest, "Prepared plan hash drift")
        parent_boundary()
        publish_json(out / "ADMISSION.json", dict(status="PASS_FULL_REAL_INPUTSET", utc=core.utc(),
            input_binding_sha256=inputs.binding_hash, plan_sha256=core.sha(args.plan),
            review_sha256=core.sha(args.review), files=inputs.files, runner_files=runner_hashes()))
        tasks = []
        for cell in document["cells"]:
            draws = cell["missing_draws"]
            if draws:
                count = min(workers, len(draws))
                tasks.extend((cell, part) for part in core.shards(draws, count))
        host_admission()
        computed, actual_compute_workers = run_pool(tasks, compute_missing, cpus, inputs.ids)
        pins_unchanged(force_hash=True)
        parent_boundary()
        publish_json(out / "COMPUTED.json", dict(status="COMPUTED_AWAITING_VERIFICATION", shards=computed,
                                                   actual_workers=actual_compute_workers,
                                                   utc=core.utc(), scientific_acceptance=False))
        # Four verification workers fit one-cell scratch + data each; compute uses
        # 12/16, while verification deliberately avoids 12 simultaneous1.2GiB scratch.
        verify_workers = min(4, workers)
        host_admission()
        verified, actual_verifiers = run_pool(document["cells"], verify_cell, cpus[:verify_workers], inputs.ids)
        pins_unchanged(force_hash=True)
        first = verified[0]["paired_weight_hashes"]
        core.require(all(cell["paired_weight_hashes"] == first for cell in verified),
                     "Paired source weights differ between cells")
        inputs.guard.check(full=True)
        core.verify_source()
        parent_boundary()
        publish_json(out / "VERIFIED.json", dict(status="PASS_DECLARED_NUMERICAL_ENDPOINT_SCOPE",
            source_sha256=core.SOURCE_HASH, input_binding_sha256=core.INPUT_HASH,
            verification_scope="all draw RNG/domain/support/macro/contrast; five deep draws percell literal exact oracle; all interval independent replay",
            geometry_scope="approved scalar oracle, float-pi joint convention; no new geometry theorem acceptance",
            cell_receipts=verified, actual_verifiers=actual_verifiers,
            utc=core.utc(), final_scientific_synthesis_required=True))
        result = closeout(document, workers)
        return result
    except BaseException as error:
        # Receipts staged during closeout are explicitly non-accepting. A final
        # commit whose publication fails is quarantined, never called COMPLETE.
        request_stop(out, error)
        name = "FAILURE.json" if (out / "LIFECYCLE.json").exists() else "LIFECYCLE.json"
        publish_json(out / name, dict(status="STOPPED", error=repr(error), utc=core.utc(),
            counters=cumulative_sample(), legacy_output_modified=False, scientific_acceptance=False))
        raise
    finally:
        resource_module.CURRENT = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    planning = sub.add_parser("plan")
    planning.add_argument("--benchmark", required=True)
    planning.add_argument("--out", required=True)
    planning.add_argument("--accounting", help="New immutable cumulative reconciliation (required before execution review)")
    running = sub.add_parser("execute")
    running.add_argument("--plan", required=True)
    running.add_argument("--review", required=True)
    args = parser.parse_args()
    plan(args) if args.command == "plan" else execute(args)


if __name__ == "__main__":
    main()

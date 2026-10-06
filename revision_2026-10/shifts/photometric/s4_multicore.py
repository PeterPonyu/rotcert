#!/usr/bin/env python3
"""Isolated multicore S4 candidate, using the unchanged, hash-pinned v3 kernel.

This runner never writes in the legacy S4 tree, clears a legacy lock, or calls
its single-worker Lifecycle. Benchmark receipts are not scientific acceptance.
Only the standard library is imported before spawn has set single-thread BLAS.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import uuid

REPO = Path(os.environ.get("ROTCERT_ROOT", Path(__file__).resolve().parents[3]))
TARGET = REPO / "revision_2026-10/inputs/s4-target"
KERNEL = REPO / "revision_2026-10/execution/s4/v3"
LEGACY = KERNEL / "runs/real-input-full-v1"
INDEX = TARGET / "s4-real-input-index-v1/index.json"
OUTPUT_ROOT = REPO / "var/s4-multicore-20261001"
SOURCE_HASH = "3713b7b4d4e8288ea7832e6e2f6665b3be443b6637c90f0c502950d0242667ef"
INDEX_HASH = "5e73a6bb0da2806289d656f3dc90b226a1f2599bfb64dce1c338f6ffba330396"
INPUT_HASH = "4f0431ee3f3a70fcb8649dc32d662249d806e243a7fc7de23194fbf1bbb86ebc"
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
SCIENCE_KEYS = ("values", "macro", "contrasts", "macro_contrasts", "q",
                "support", "all_source", "multiplicity")
GIB = 1024 ** 3
RNG = "PCG64 SeedSequence([20260928,4,0,draw,role]); one shared source draw across all cells"
_MODULES = None
_DATA = None
_PIPELINE = None
_ROLES = None
_SPEC = None
_SETUP = None
_GUARDS = None
_RUNNER_PIN = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def pin_runner():
    global _RUNNER_PIN
    current = sha(__file__)
    if _RUNNER_PIN is None:
        _RUNNER_PIN = current
    require(current == _RUNNER_PIN, "Runner source changed during execution")
    return _RUNNER_PIN


def owned_output(path):
    path = Path(path).resolve()
    require(path != OUTPUT_ROOT and path.is_relative_to(OUTPUT_ROOT.resolve()),
            "Output must be a new child of var/s4-multicore-20261001")
    require(not path.is_relative_to(KERNEL.resolve()), "Legacy output is read-only")
    return path


def write_json(path, value):
    """No-overwrite atomic publication, never a replacement of user evidence."""
    path = owned_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temp.open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temp.exists():
            temp.unlink()


def save_arrays(path, arrays, metadata):
    np = modules()["np"]
    path = owned_output(path)
    require(not path.exists() and not path.with_suffix(".json").exists(),
            "Checkpoint exists/orphan: preserve rather than overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temp.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        if temp.exists():
            temp.unlink()
    write_json(path.with_suffix(".json"), dict(metadata, file=path.name, sha256=sha(path)))


def modules():
    global _MODULES
    if _MODULES is None:
        for key in THREAD_KEYS:
            os.environ[key] = "1"
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(KERNEL))
        _MODULES = {name: importlib.import_module(name) for name in
                    ("config", "io_s4", "statistics_core", "inputs", "prepare",
                     "run_s4", "verify_s4", "endpoint_reference")}
        _MODULES["np"] = importlib.import_module("numpy")
    return _MODULES


def verify_source():
    require(sha(KERNEL / "FROZEN.json") == SOURCE_HASH, "Pinned S4 freeze drift")
    result = modules()["io_s4"].verify()
    require(result == SOURCE_HASH, "S4 source/environment verification failed")
    require(sha(INDEX) == INDEX_HASH, "Pinned real-input index drift")
    return result


def stat_id(path):
    value = Path(path).stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def checked_json(path, expected_hash=None):
    before = stat_id(path)
    if expected_hash is not None:
        require(sha(path) == expected_hash, "Bound JSON hash drift: " + str(path))
    value = json.loads(Path(path).read_text())
    require(stat_id(path) == before, "JSON changed during read: " + str(path))
    return value


def validate_binding():
    binding = checked_json(LEGACY / "INPUTS.json")
    require(binding["source_sha256"] == SOURCE_HASH, "Legacy input source mismatch")
    require(binding["input_binding_sha256"] == INPUT_HASH and
            object_hash(binding["binding"]) == INPUT_HASH, "Input binding drift")
    require(binding["binding"]["synthetic"] is False, "Real inputs required")
    require(binding["binding"]["index_sha256"] == INDEX_HASH, "Index identity mismatch")
    return binding


def source_roles(binding):
    m = modules()
    index = checked_json(INDEX, INDEX_HASH)
    bundle = (INDEX.parent / index["bundle"]).resolve()
    path = bundle / "sources.jsonl"
    require(sha(path) == binding["binding"]["sources_sha256"], "Source inventory drift")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require(len(rows) == 11738 and len({row["id"] for row in rows}) == 11738,
            "Full unique DIOR inventory required")
    ids = sorted(row["id"] for row in rows)
    roles = m["inputs"].roles_from_ids(ids)
    require(all(row["source_id"] == row["id"] and row["role"] == roles[row["id"]]
                for row in rows), "Source or fixed role drift")
    numeric = m["np"].array([0 if roles[sid] == "calibration" else 1 for sid in ids],
                            dtype=m["np"].int8)
    require(int((numeric == 0).sum()) == 5869 and int((numeric == 1).sum()) == 5869,
            "Fixed role counts drift")
    return ids, numeric


def prepared_paths(model, tau):
    return [LEGACY / "prepared" / f"{model}__{condition}__iou{tau:.1f}.npz"
            for condition in modules()["config"].CONDITIONS]


def load_data(model, tau):
    result = []
    guards = {}
    for condition, path in zip(modules()["config"].CONDITIONS, prepared_paths(model, tau)):
        expected = dict(source_sha256=SOURCE_HASH, input_binding_sha256=INPUT_HASH,
                        model=model, condition=condition, iou=tau, score_floor=.05,
                        schema="S4-matched-v1")
        validate_prepared_provenance(path)
        before = {str(p): stat_id(p) for p in (path, path.with_suffix(".json"))}
        result.append(modules()["io_s4"].load_npz(path, expected))
        require(all(stat_id(p) == identity for p, identity in before.items()),
                "Prepared arrays changed during read")
        guards.update(before)
    modules()["endpoint_reference"].validate_prepared(result, 11738)
    return result, guards


def validate_prepared_provenance(path):
    """Anchor matching arrays to the contract INSIDE the original source freeze."""
    path = Path(path)
    frozen = checked_json(KERNEL / "FROZEN.json", SOURCE_HASH)
    contract_path = KERNEL / "PREPARED-REUSE.json"
    contract_hash = frozen["files"]["PREPARED-REUSE.json"]
    contract = checked_json(contract_path, contract_hash)
    require(contract["schema"] == "S4-reviewed-prepared-reuse-v1" and
            contract["original_input_binding_sha256"] == INPUT_HASH,
            "Prepared trusted contract identity drift")
    record = contract["prepared_files"][path.name]
    side = checked_json(path.with_suffix(".json"))
    require(side["sha256"] == record["sha256"] and sha(path) == record["sha256"],
            "Prepared arrays differ from frozen trusted hash")
    require(side["provenance_reuse"] == dict(contract_sha256=contract_hash,
            original_source_sha256=contract["original_source_sha256"],
            original_sidecar_sha256=record["sidecar_sha256"]),
            "Prepared matching provenance drift")


def affinity_plan(workers, available=None):
    require(type(workers) is int and 1 <= workers <= 16, "Workers must be 1..16")
    cpus = sorted(os.sched_getaffinity(0) if available is None else available)
    # Select one logical CPU per physical core. Threads are not counted as cores.
    physical, selected = set(), []
    for cpu in cpus:
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            identity = ((base / "physical_package_id").read_text().strip(),
                        (base / "core_id").read_text().strip())
        except FileNotFoundError:
            identity = ("unreported", str(cpu))
        if identity not in physical:
            physical.add(identity)
            selected.append(cpu)
    require(len(selected) >= workers, "Fewer physical cores available than requested")
    require(workers == 1 or len(selected) - workers >= 4,
            "At least four physical cores must remain for the OS/other work")
    # This host has no quota file in its exposed cgroup namespace. Do not invent one.
    sys.path.insert(0, str(REPO))
    from rotcert.matching_parallel import _cgroup_cpu_quota
    quota = _cgroup_cpu_quota()
    if quota is not None:
        require(quota >= workers, "Cgroup CPU quota below requested worker count")
    return dict(worker_cpus=selected[:workers], available_physical_cpus=selected,
                reserved_physical_cpus=selected[workers:], cpu_quota=quota,
                quota_scope="readable cgroup v1/v2 hierarchy; null is unavailable, not unlimited")


def cgroup_memory_headroom(cgroup_file=Path("/proc/self/cgroup"),
                          mountinfo_file=Path("/proc/self/mountinfo")):
    """Minimum remaining memory of readable membership/ancestor limits, v1/v2."""
    def read(path):
        try:
            return Path(path).read_text().strip()
        except OSError:
            return None
    memberships = []
    for line in (read(cgroup_file) or "").splitlines():
        fields = line.split(":", 2)
        if len(fields) == 3:
            memberships.append((set(fields[1].split(",")), Path(fields[2])))
    candidates, observed = [], []
    for line in (read(mountinfo_file) or "").splitlines():
        before, separator, after = line.partition(" - ")
        left, right = before.split(), after.split()
        if not separator or len(left) < 5 or len(right) < 3 or right[0] not in ("cgroup", "cgroup2"):
            continue
        kind = right[0]
        if kind == "cgroup" and "memory" not in right[2].split(","):
            continue
        def unescape(value):
            for code, char in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
                value = value.replace(code, char)
            return value
        root, mount = Path(unescape(left[3])), Path(unescape(left[4]))
        for controllers, group in memberships:
            if (kind == "cgroup2" and controllers != {""}) or (kind == "cgroup" and "memory" not in controllers):
                continue
            try:
                relative = group.relative_to(root)
            except ValueError:
                relative = group.relative_to("/")
            current = mount / relative
            while True:
                limit_name, use_name = ("memory.max", "memory.current") if kind == "cgroup2" else ("memory.limit_in_bytes", "memory.usage_in_bytes")
                limit, used = read(current / limit_name), read(current / use_name)
                if limit is not None and used is not None:
                    observed.append(str(current))
                    if limit != "max":
                        try:
                            maximum, usage = int(limit), int(used)
                            if maximum > 0 and usage >= 0:
                                candidates.append(max(0, maximum - usage))
                        except ValueError:
                            raise ValueError("Malformed readable cgroup memory limit") from None
                if current == mount:
                    break
                current = current.parent
    return dict(headroom_bytes=min(candidates) if candidates else None,
                readable_groups=observed,
                status="READABLE_LIMITS" if candidates else "READABLE_UNLIMITED" if observed else "UNAVAILABLE")


def available_memory():
    values = {line.split(":", 1)[0]: int(line.split()[1]) * 1024
              for line in Path("/proc/meminfo").read_text().splitlines()
              if line.startswith(("MemAvailable:", "MemTotal:"))}
    group = cgroup_memory_headroom()
    if group["headroom_bytes"] is not None:
        values["MemAvailable"] = min(values["MemAvailable"], group["headroom_bytes"])
    return values["MemAvailable"]


def health():
    status = Path("/proc/self/status").read_text().splitlines()
    return dict(pid=os.getpid(), cpu_affinity=sorted(os.sched_getaffinity(0)),
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                peak_virtual_address_bytes=int(next(line.split()[1] for line in status
                                                     if line.startswith("VmPeak:"))) * 1024,
                threads=int(next(line.split()[1] for line in status if line.startswith("Threads:"))),
                blas_threads={key: os.environ.get(key) for key in THREAD_KEYS})


@contextmanager
def single_thread_spawn():
    previous = {key: os.environ.get(key) for key in THREAD_KEYS}
    previous_bytecode = os.environ.get("PYTHONDONTWRITEBYTECODE")
    try:
        for key in THREAD_KEYS:
            os.environ[key] = "1"
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if previous_bytecode is None:
            os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
        else:
            os.environ["PYTHONDONTWRITEBYTECODE"] = previous_bytecode


def worker_init(cpus, slots, spec):
    global _DATA, _PIPELINE, _ROLES, _SPEC, _SETUP, _GUARDS
    start_cpu, start_wall = time.process_time(), time.monotonic()
    with slots.get_lock():
        slot = slots.value
        slots.value += 1
    require(slot < len(cpus), "Unexpected worker replacement; refuse affinity reuse")
    os.sched_setaffinity(0, {cpus[slot]})
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = min(4 * GIB, hard) if hard != resource.RLIM_INFINITY else 4 * GIB
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    require(all(os.environ.get(key) == "1" for key in THREAD_KEYS), "BLAS oversubscription")
    require(pin_runner() == spec["runner_sha256"], "Child runner differs from admitted parent")
    verify_source()
    binding = validate_binding()
    ids, _ROLES = source_roles(binding)
    _DATA, _GUARDS = load_data(spec["model"], spec["tau"])
    _PIPELINE = modules()["statistics_core"].ModelPipeline(_DATA, _ROLES)
    _SPEC = dict(spec, ids=ids)
    _SETUP = dict(cpu_seconds=time.process_time() - start_cpu,
                  wall_seconds=time.monotonic() - start_wall, **health())


def check_worker_boundary():
    require(pin_runner() == _SPEC["runner_sha256"], "Worker runner source drift")
    require(time.time() < _SPEC["deadline"], "Candidate wall deadline reached")
    require(not Path(_SPEC["stop_file"]).exists(), "Parent requested cooperative stop")
    require(available_memory() >= _SPEC["memory_reserve_bytes"], "Memory reserve reached")
    require(shutil.disk_usage(OUTPUT_ROOT).free >= _SPEC["disk_reserve_bytes"],
            "Disk reserve reached")
    require(all(stat_id(path) == identity for path, identity in _GUARDS.items()),
            "Prepared input changed during execution")


def array_digest(arrays):
    digest = hashlib.sha256()
    for key in SCIENCE_KEYS:
        arr = modules()["np"].ascontiguousarray(arrays[key])
        require(not arr.dtype.hasobject, "Object arrays are forbidden")
        digest.update(key.encode() + str(arr.dtype).encode() + repr(arr.shape).encode())
        # Normalize NA payload and signed zero before hashing numerical equality.
        if arr.dtype.kind == "f":
            arr = arr.copy()
            arr[modules()["np"].isnan(arr)] = modules()["np"].nan
            arr[arr == 0] = 0.
        digest.update(arr.tobytes())
    return digest.hexdigest()


def exact_compare(actual, expected):
    np = modules()["np"]
    for key in SCIENCE_KEYS:
        require(actual[key].shape == expected[key].shape and actual[key].dtype == expected[key].dtype,
                "Scientific shape/dtype changed: " + key)
        require(np.array_equal(actual[key], expected[key], equal_nan=True),
                "Exact scientific result differs: " + key)


def bounded_independent_check(result, ids, data, draw):
    """Full exact recalibration, one predeclared sparse class, global endpoints.

    The expensive all-class scalar geometry oracle remains mandatory for the
    formal completion verifier; this is explicitly a bounded benchmark check.
    """
    m = modules()
    verifier = m["verify_s4"]
    weights = verifier.expected_weights(type("BoundInput", (), {"ids": ids})(), draw)
    for ci, values in enumerate(data):
        grouped = [verifier.ref_source_groups(values, k) for k in range(20)]
        for method in range(2):
            for k in range(20):
                require(verifier.fit_source_groups(grouped[k], weights[0], method) == result["q"][ci, method, k],
                        "Independent exact threshold differs")
    # Choose by fixed prepared TP counts, not by an outcome or run speed. Prefer
    # nonempty class so the geometry comparison is substantive.
    counts = data[0]["counts"][:, :, 2].sum(axis=0)
    eligible = [k for k in range(20) if counts[k] > 0]
    require(bool(eligible), "No supported real class for bounded oracle")
    chosen = min(eligible, key=lambda k: (int(counts[k]), k))
    for method in range(2):
        threshold = float(result["q"][0, method, chosen])
        reference = m["endpoint_reference"].literal_class(data[0], chosen, threshold, data[0], *weights)
        m["np"].testing.assert_allclose(result["values"][0, method, chosen], reference,
                                       rtol=4e-15, atol=4e-15, equal_nan=True)
        for ai, arm in enumerate(m["config"].ARMS):
            reference_global = m["endpoint_reference"].literal_global(
                data[arm["eval"]], result["q"][arm["cal"], method], weights[1])
            m["np"].testing.assert_allclose(result["all_source"][ai, method], reference_global,
                                           rtol=4e-15, atol=4e-15, equal_nan=True)
    return dict(draw=draw, independent_exact_thresholds=240,
                literal_endpoint_class=chosen, literal_endpoint_arm="clean_anchor",
                literal_class_methods=2, literal_all_source_arms=11, literal_all_source_methods=2,
                all_class_geometry_deep=False,
                scope="bounded performance/equivalence evidence, not formal full endpoint acceptance")


def checkpoint_name(draw):
    return "base" if draw == -1 else f"draw-{draw:04d}"


def checkpoint_metadata(model, tau, draw):
    return modules()["run_s4"].checkpoint_meta(SOURCE_HASH, INPUT_HASH, model, tau, draw)


def compute_shard(draws):
    records = []
    shard_start_cpu, shard_start_wall = time.process_time(), time.monotonic()
    for draw in draws:
        check_worker_boundary()
        start_cpu, start_wall = time.process_time(), time.monotonic()
        mult = modules()["statistics_core"].multiplicities(_ROLES, draw)
        result = _PIPELINE.run(mult)
        result["multiplicity"] = mult
        pipeline_cpu = time.process_time() - start_cpu
        result["pipeline_cpu_seconds"] = modules()["np"].asarray(pipeline_cpu)
        modules()["run_s4"].validate_saved(result, _ROLES, draw)
        checked = modules()["verify_s4"].verify_one(
            result, type("BoundInput", (), {"ids": _SPEC["ids"]})(), _DATA, draw, False)
        path = owned_output(Path(_SPEC["out"]) / checkpoint_name(draw)).with_suffix(".npz")
        saved_reference = LEGACY / _SPEC["model"] / f"iou{_SPEC['tau']:.1f}" / path.name
        against_saved = False
        if saved_reference.exists():
            reference = modules()["io_s4"].load_npz(saved_reference,
                                                   checkpoint_metadata(_SPEC["model"], _SPEC["tau"], draw))
            exact_compare(result, reference)
            against_saved = True
        check_worker_boundary()
        save_start = time.process_time()
        save_arrays(path, result, dict(checkpoint_metadata(_SPEC["model"], _SPEC["tau"], draw),
                                      execution_contract="multicore-candidate-20261001-v1"))
        # Read back on disk, so equality is not only a pre-serialization assertion.
        readback = modules()["io_s4"].load_npz(path, checkpoint_metadata(_SPEC["model"], _SPEC["tau"], draw))
        exact_compare(result, readback)
        records.append(dict(draw=draw, science_sha256=array_digest(result),
                            pipeline_cpu_seconds=pipeline_cpu,
                            fit_check_save_cpu_seconds=time.process_time() - start_cpu,
                            fit_check_save_wall_seconds=time.monotonic() - start_wall,
                            checkpoint_io_cpu_seconds=time.process_time() - save_start,
                            exact_saved_v3_comparison=against_saved,
                            independent_light_verification=checked, bytes=path.stat().st_size))
    check_worker_boundary()
    return dict(records=records, setup=_SETUP, cpu_seconds=time.process_time() - shard_start_cpu,
                wall_seconds=time.monotonic() - shard_start_wall, health=health())


def shards(draws, count):
    """Contiguous draw-ID shards, never worker-specific random streams."""
    require(len(draws) >= count and len(draws) == len(set(draws)), "Unique sufficient draws required")
    return [draws[i * len(draws) // count:(i + 1) * len(draws) // count] for i in range(count)]


def cpu_total():
    own = resource.getrusage(resource.RUSAGE_SELF)
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + child.ru_utime + child.ru_stime


def execute_cell(model, tau, draws, workers, out, wall_limit=1800, cpu_limit=7200):
    source_pin = pin_runner()
    out = owned_output(out)
    require(not out.exists(), "Fresh isolated cell output required")
    plan = affinity_plan(workers)
    require(available_memory() >= workers * 2 * GIB + 8 * GIB,
            "Admission needs 2GiB RSS allowance per worker plus 8GiB available-memory reserve")
    require(shutil.disk_usage(REPO).free >= 4 * GIB, "Insufficient disk reserve")
    require(0 < wall_limit <= 7 * 86400 and 0 < cpu_limit <= 30 * 86400, "Finite bounded limits required")
    out.mkdir(parents=True)
    started_wall, started_cpu = time.monotonic(), cpu_total()
    parent_process_cpu = time.process_time()
    stop = out / "STOP-REQUEST.json"
    spec = dict(model=model, tau=tau, out=str(out), deadline=time.time() + wall_limit,
                stop_file=str(stop), memory_reserve_bytes=8 * GIB, disk_reserve_bytes=2 * GIB,
                runner_sha256=source_pin)
    identity = dict(schema="S4-multicore-candidate-cell-v1", model=model, iou=tau, draws=draws,
                    workers=workers, affinity=plan, source_sha256=SOURCE_HASH,
                    input_binding_sha256=INPUT_HASH, runner_sha256=source_pin,
                    science_semantics_unchanged=True, scientific_acceptance=False,
                    RLIMIT_AS_per_worker_bytes=4 * GIB,
                    memory_scope="4GiB virtual cap; 2GiB RSS admission allowance is not a hard RSS cap",
                    max_wall_seconds=wall_limit, max_aggregate_cpu_seconds=cpu_limit, utc=utc())
    write_json(out / "IDENTITY.json", identity)
    ctx = mp.get_context("spawn")
    slots = ctx.Value("i", 0)
    responses, failure = [], None
    try:
        with single_thread_spawn(), ProcessPoolExecutor(
                max_workers=workers, mp_context=ctx, initializer=worker_init,
                initargs=(plan["worker_cpus"], slots, spec)) as pool:
            try:
                pending = {pool.submit(compute_shard, part) for part in shards(draws, workers)}
                while pending:
                    require(pin_runner() == source_pin, "Parent runner source drift")
                    done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                    for future in done:
                        responses.append(future.result())
                    # Active child CPU is read from /proc; reaped-child costs from rusage
                    # are charged at terminal closeout. No parent-only budget accounting.
                    active_cpu = 0.
                    for child in mp.active_children():
                        try:
                            text = Path(f"/proc/{child.pid}/stat").read_text()
                            fields = text[text.rfind(")") + 2:].split()
                            active_cpu += (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
                        except FileNotFoundError:
                            pass
                    used_cpu = cpu_total() - started_cpu + active_cpu
                    if time.monotonic() - started_wall >= wall_limit or used_cpu >= cpu_limit:
                        raise RuntimeError("Cooperative stop requested at next draw boundary")
            except BaseException as error:
                # Signal before the executor context waits for workers; otherwise
                # an early worker failure could leave other whole shards running.
                if not stop.exists():
                    write_json(stop, dict(reason=repr(error), utc=utc()))
                raise
        require(len(responses) == workers, "Missing shard result")
        records = sorted((record for response in responses for record in response["records"]),
                         key=lambda record: record["draw"])
        require([record["draw"] for record in records] == sorted(draws), "Draw coverage incomplete")
        require(len({response["health"]["pid"] for response in responses}) == workers,
                "Benchmark did not actually use requested distinct workers")
        require(all(len(response["health"]["cpu_affinity"]) == 1 and
                    response["health"]["threads"] == 1 for response in responses),
                "Single-core/thread worker contract violated")
        report = dict(identity, status="COMPUTED_LIGHT_VERIFIED_CANDIDATE_ONLY", records=records,
                      shards=responses, wall_seconds=time.monotonic() - started_wall,
                      aggregate_cpu_seconds=cpu_total() - started_cpu,
                      parent_cpu_seconds=time.process_time() - parent_process_cpu,
                      parent_health=health(), summed_worker_peak_rss_bytes=sum(
                          response["health"]["peak_rss_bytes"] for response in responses),
                      compute_critical_path_seconds=max(response["wall_seconds"] for response in responses),
                      terminal_utc=utc(), independent_deep_verification=False,
                      CI_reduction=False, legacy_output_modified=False)
        require(report["aggregate_cpu_seconds"] <= cpu_limit and report["wall_seconds"] <= wall_limit,
                "Terminal accounting exceeded schedule")
        write_json(out / "CELL.json", report)
        require(pin_runner() == source_pin, "Runner changed before terminal publication")
        require(cpu_total() - started_cpu <= cpu_limit and time.monotonic() - started_wall <= wall_limit,
                "Closeout budget exceeded")
        return report
    except BaseException as error:
        failure = repr(error)
        if not stop.exists():
            write_json(stop, dict(reason=failure, utc=utc()))
        raise
    finally:
        if failure is None and (cpu_total() - started_cpu > cpu_limit or time.monotonic() - started_wall > wall_limit):
            failure = "Closeout budget exceeded"
        write_json(out / "LIFECYCLE.json", dict(
            status="STOPPED" if failure else "COMPLETE_BOUNDED_CANDIDATE_NOT_SCIENTIFIC_ACCEPTANCE",
            aggregate_cpu_seconds=cpu_total() - started_cpu, wall_seconds=time.monotonic() - started_wall,
            resource_scope="parent plus all reaped children, including startup/checkpoint/verification",
            error=failure, utc=utc(), legacy_output_modified=False))
        if failure == "Closeout budget exceeded":
            raise RuntimeError(failure)


def benchmark(args):
    source_pin = pin_runner()
    verify_source()
    binding = validate_binding()
    _, roles = source_roles(binding)
    model, tau = args.model, args.iou
    require(model in modules()["config"].MODELS and tau in modules()["config"].TAUS, "Unplanned model/IoU")
    require(16 <= args.draws <= 256 and args.draws < modules()["config"].B,
            "Benchmark must have 16..256 draws, not formal B5000")
    out = owned_output(args.out)
    require(not out.exists(), "Benchmark output already exists")
    out.mkdir(parents=True)
    started_cpu, started_wall = cpu_total(), time.monotonic()
    data, guard = load_data(model, tau)
    prepared_hashes = {str(path): sha(path) for path in guard}
    reports = []
    chosen = list(range(args.draws))
    # Same workload for each worker count, including independent draws and I/O.
    for workers in args.workers:
        print(json.dumps(dict(stage="benchmark_cell", workers=workers, model=model,
                              draws=len(chosen), utc=utc())), flush=True)
        report = execute_cell(model, tau, chosen, workers, out / f"workers-{workers}",
                              args.wall_limit, args.cpu_limit)
        reports.append(report)
        print(json.dumps(dict(stage="benchmark_cell_complete", workers=workers,
                              wall_seconds=report["wall_seconds"],
                              aggregate_cpu_seconds=report["aggregate_cpu_seconds"],
                              compute_seconds=report["compute_critical_path_seconds"])), flush=True)
    require(len({r["workers"] for r in reports}) == len(reports), "Duplicate worker trials")
    baseline = next(r for r in reports if r["workers"] == 1)
    hashes = {record["draw"]: record["science_sha256"] for record in baseline["records"]}
    for report in reports:
        require(report["runner_sha256"] == source_pin, "Mixed runner versions in cells")
        require({record["draw"]: record["science_sha256"] for record in report["records"]} == hashes,
                "Serial/parallel scientific arrays differ")
        # Compare actual checkpoint arrays as well as hashes.
        for draw in chosen:
            name = checkpoint_name(draw) + ".npz"
            expected = checkpoint_metadata(model, tau, draw)
            ref = modules()["io_s4"].load_npz(out / "workers-1" / name, expected)
            got = modules()["io_s4"].load_npz(out / f"workers-{report['workers']}" / name, expected)
            exact_compare(got, ref)
    # Independent exact calibration + predeclared bounded endpoint oracle. The
    # full all-class geometry oracle is intentionally NOT claimed for benchmark.
    deep = []
    for draw in [0, chosen[-1]]:
        result = modules()["io_s4"].load_npz(out / "workers-1" / (checkpoint_name(draw) + ".npz"),
                                           checkpoint_metadata(model, tau, draw))
        deep.append(bounded_independent_check(result, source_roles(binding)[0], data, draw))
    for path, expected in prepared_hashes.items():
        require(sha(path) == expected, "Prepared files drifted during benchmark")
    verify_source()
    require(pin_runner() == source_pin, "Runner drift before benchmark receipt")
    by_worker = {report["workers"]: report for report in reports}
    require(12 in by_worker and 16 in by_worker, "Both 12 and 16 worker measurements required")
    # Predeclared selection: use 16 for sustained compute if at least 10% faster;
    # otherwise 12. Report cold startup separately, not as a whole-run ETA.
    improvement = 1 - by_worker[16]["compute_critical_path_seconds"] / by_worker[12]["compute_critical_path_seconds"]
    selected = 16 if improvement >= .10 else 12
    receipt = dict(schema="S4-multicore-real-input-benchmark-v1", utc=utc(),
                   status="PASS_EXACT_SERIAL_PARALLEL_BOUNDED_REAL_INPUTS", scientific_acceptance=False,
                   model=model, iou=tau, draws=chosen, source_sha256=SOURCE_HASH,
                   index_sha256=INDEX_HASH, input_binding_sha256=INPUT_HASH,
                   runner_sha256=source_pin, prepared_files=prepared_hashes,
                   entire_benchmark_cpu_seconds=cpu_total() - started_cpu,
                   entire_benchmark_wall_seconds=time.monotonic() - started_wall,
                   memory_cgroup=cgroup_memory_headroom(),
                   selected_workers=selected, selection_rule="16 if sustained compute is >=10% faster than 12, else 12",
                   compute_improvement_16_vs_12=improvement,
                   exact_fields=list(SCIENCE_KEYS), timing_fields_excluded=["pipeline_cpu_seconds"],
                   independent_deep_checks=deep,
                   full_independent_endpoint_verification=False,
                   results=[dict(workers=r["workers"], wall_seconds=r["wall_seconds"],
                                 aggregate_cpu_seconds=r["aggregate_cpu_seconds"],
                                 compute_critical_path_seconds=r["compute_critical_path_seconds"],
                                 speedup_wall_vs_1=baseline["wall_seconds"] / r["wall_seconds"],
                                 speedup_compute_vs_1=baseline["compute_critical_path_seconds"] / r["compute_critical_path_seconds"],
                                 summed_worker_peak_rss_bytes=r["summed_worker_peak_rss_bytes"],
                                 workers_observed=[response["health"] for response in r["shards"]])
                            for r in reports],
                   limitations=["One model/IoU and bounded draw range; not a complete B5000 result or ETA",
                                "Full InputSet/provenance replay is required for production candidate",
                                "No legacy ACTIVE receipt, checkpoint or frozen source was modified",
                                "Full interval/MC-tail reduction and independent acceptance remain required"])
    write_json(out / "BENCHMARK.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "selected_workers", "compute_improvement_16_vs_12")}), flush=True)
    return receipt


def candidate(args):
    """Actual isolated draw execution after full archive admission; not a full receipt."""
    verify_source()
    require(1 <= args.draws <= 256, "Bounded candidate only: full5000 needs an amended reviewed execution contract")
    require(args.workers in (12, 16), "Candidate requires 12 or16 workers")
    receipt = checked_json(args.benchmark)
    require(receipt["status"] == "PASS_EXACT_SERIAL_PARALLEL_BOUNDED_REAL_INPUTS" and
            receipt["source_sha256"] == SOURCE_HASH and receipt["input_binding_sha256"] == INPUT_HASH and
            receipt["runner_sha256"] == sha(__file__), "Valid current-kernel benchmark required")
    require(args.workers == receipt["selected_workers"], "Use measured recommended worker count")
    out = owned_output(args.out)
    require(not out.exists(), "Fresh candidate output required")
    print(json.dumps(dict(stage="full_input_admission", utc=utc())), flush=True)
    inputs = modules()["inputs"].InputSet(INDEX, synthetic=False)
    require(inputs.binding_hash == INPUT_HASH, "Production input binding drift")
    inputs.guard.check(full=True)
    out.mkdir(parents=True)
    write_json(out / "ADMISSION.json", dict(status="PASS_REAL_FULL_INPUTSET_CANDIDATE_ONLY", utc=utc(),
                                            binding=inputs.binding, files=inputs.files,
                                            scientific_acceptance=False, benchmark_sha256=sha(args.benchmark)))
    cells = []
    for model in args.models:
        for tau in args.ious:
            cells.append(execute_cell(model, tau, [-1] + list(range(args.draws)), args.workers,
                                      out / model / f"iou{tau:.1f}", args.wall_limit, args.cpu_limit))
    inputs.guard.check(full=True)
    verify_source()
    write_json(out / "CANDIDATE.json", dict(status="BOUNDED_COMPUTED_AWAITING_FULL_INDEPENDENT_ACCEPTANCE",
                                            scientific_acceptance=False, B=args.draws,
                                            cells=[dict(model=c["model"], iou=c["iou"], wall_seconds=c["wall_seconds"])
                                                   for c in cells], utc=utc()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("benchmark", help="true input 1/12/16-core exact-equivalence and throughput")
    bench.add_argument("--model", default="dior-rtmdet-s0")
    bench.add_argument("--iou", type=float, default=.5)
    bench.add_argument("--draws", type=int, default=48)
    bench.add_argument("--workers", nargs="+", type=int, default=[1, 12, 16])
    run = sub.add_parser("candidate", help="bounded real-input candidate, never legacy/full acceptance")
    run.add_argument("--benchmark", required=True)
    run.add_argument("--workers", type=int, required=True)
    run.add_argument("--draws", type=int, default=48)
    run.add_argument("--models", nargs="+", default=["dior-rtmdet-s0"])
    run.add_argument("--ious", nargs="+", type=float, default=[.5])
    for command in (bench, run):
        command.add_argument("--out", required=True)
        command.add_argument("--wall-limit", type=float, default=1800)
        command.add_argument("--cpu-limit", type=float, default=7200)
    args = parser.parse_args()
    if args.command == "benchmark":
        benchmark(args)
    else:
        candidate(args)


if __name__ == "__main__":
    main()

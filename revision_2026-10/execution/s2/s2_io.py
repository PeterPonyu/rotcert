"""S2 isolated ownership, immutable snapshots, atomic checkpoint journal."""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time
import uuid
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parent
SOURCE_NAMES = ['s2_core.py', 's2_io.py', 'run_s2.py', 'verify_s2.py', 'test_s2.py', 'protocol.json']
INPUT_RELS = ['statistics/STATISTICAL-PROTOCOL.md', 'theory/REPAIR-NOTE.md',
              'REPAIR-DESIGN.md', 'statistics/ANALYSIS-MANIFEST.json']
THREAD_KEYS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def utc():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with tmp.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def environment():
    return dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                machine=platform.machine(), platform=platform.platform())


@contextlib.contextmanager
def lock(root=ROOT):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    f = (root / 'ownership.lock').open('a+')
    try:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('another S2 writer holds ownership lock') from None
        yield
    finally:
        f.close()


def freeze():
    path = ROOT / 'FROZEN.json'
    if path.exists():
        verify_frozen()
        return read_json(path)
    h = ROOT.parent.parent
    dest = ROOT / 'frozen_inputs'
    dest.mkdir(exist_ok=True)
    inputs = []
    for rel in INPUT_RELS:
        p = h / rel
        copy = dest / rel.replace('/', '__')
        if copy.exists() and digest(copy) != digest(p):
            raise RuntimeError('preexisting snapshot differs: ' + str(copy))
        if not copy.exists():
            copy.write_bytes(p.read_bytes())
        inputs.append(dict(original=str(p), local=str(copy.relative_to(ROOT)), sha256=digest(copy)))
    from s2_core import conditions, METHODS, METRICS, rng
    mapping = dict(conditions=conditions(), methods=METHODS, metrics=METRICS,
                   streams={'0':'calibration','1':'fresh_evaluation','2':'evaluation_false_positive_counts'},
                   example_initial_state=rng(0,0,0).bit_generator.state)
    write_json(ROOT / 'CONDITIONS.json', mapping)
    sources = {name:digest(ROOT/name) for name in SOURCE_NAMES + ['CONDITIONS.json']}
    manifest = dict(schema='s2-frozen-source-v1', utc=utc(), sources=sources, inputs=inputs,
                    environment=environment(), requested_model='gpt-6-astra',
                    actual_model_verification='parent-owned; not inferred from requested model')
    write_json(path, manifest)
    return manifest


def verify_frozen():
    frozen = read_json(ROOT/'FROZEN.json')
    for rel, expected in frozen['sources'].items():
        if digest(ROOT/rel) != expected:
            raise RuntimeError('frozen source hash changed: ' + rel)
    for item in frozen['inputs']:
        if digest(ROOT/item['local']) != item['sha256']:
            raise RuntimeError('frozen input snapshot hash changed: ' + item['local'])
        if digest(item['original']) != item['sha256']:
            raise RuntimeError('upstream frozen design changed: ' + item['original'])
    if environment() != frozen['environment']:
        raise RuntimeError('runtime environment differs from freeze')
    return digest(ROOT/'FROZEN.json')


def resources():
    available = None
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            available = int(line.split()[1])*1024
    return dict(rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                available_memory_bytes=available, free_disk_bytes=shutil.disk_usage(ROOT).free,
                pid=os.getpid(), utc=utc())


def set_resource_limit():
    cap = 2 * 1024**3
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    upper = min(cap, hard) if hard != resource.RLIM_INFINITY else cap
    resource.setrlimit(resource.RLIMIT_AS, (upper, hard))
    if any(os.environ.get(key) != '1' for key in THREAD_KEYS):
        raise RuntimeError('BLAS/OMP thread limit not one')


def save_chunk(directory, first, values, aux, frozen_hash):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    n = len(values)
    name = f'reps-{first:04d}-{first+n-1:04d}.npz'
    path = directory / name
    receipt = path.with_suffix('.json')
    if path.exists() or receipt.exists():
        # A crash after NPZ rename but before receipt is recoverable only after
        # exact deterministic regenerated-array equality; never overwrite it.
        if receipt.exists():
            raise RuntimeError('refusing duplicate committed checkpoint ' + name)
        with np.load(path,allow_pickle=False) as old:
            equal = (np.array_equal(old['values'], values, equal_nan=True)
                     and np.array_equal(old['aux'],aux,equal_nan=True)
                     and np.array_equal(old['replication_ids'],np.arange(first,first+n)))
        if not equal:
            raise RuntimeError('orphan checkpoint differs from deterministic replay')
    else:
        temp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
        with temp.open('xb') as f:
            np.savez(f, values=values, aux=aux, replication_ids=np.arange(first,first+n))
            f.flush(); os.fsync(f.fileno())
        os.replace(temp,path)
    write_json(receipt, dict(file=name, sha256=digest(path), first=first, count=n,
                             frozen_hash=frozen_hash, utc=utc(),
                             next_rng_key=[20260928,2,int(directory.name),first+n],
                             restart='per-rep SeedSequence deterministic restart, no advancing shared RNG'))


def load_chunks(directory, frozen_hash, target):
    directory = Path(directory)
    values=[];aux=[];done=0
    for receipt in sorted(directory.glob('reps-*.json')):
        meta=read_json(receipt)
        if meta['first'] != done or meta['frozen_hash'] != frozen_hash:
            raise RuntimeError('noncontiguous or foreign checkpoint')
        path=directory/meta['file']
        if digest(path) != meta['sha256']:
            raise RuntimeError('checkpoint SHA mismatch')
        with np.load(path,allow_pickle=False) as part:
            ids=part['replication_ids']
            if not np.array_equal(ids,np.arange(done,done+meta['count'])):
                raise RuntimeError('replication IDs differ from checkpoint')
            if part['values'].shape!=(meta['count'],6,20) or part['aux'].shape!=(meta['count'],4):
                raise RuntimeError('checkpoint schema mismatch')
            values.append(part['values'].copy());aux.append(part['aux'].copy())
        done+=meta['count']
    if done>target:raise RuntimeError('checkpoint target count changed')
    return values,aux,done

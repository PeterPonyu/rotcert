"""V2 immutable provenance, per-cell ownership, conservative restart and gate."""
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import uuid
import numpy as np
import scipy
import shapely
ROOT=Path(__file__).resolve().parent
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
LIMIT=4*1024**3


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def now():
    from datetime import datetime,timezone
    return datetime.now(timezone.utc).isoformat()


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,np.bool_):return bool(x)
    if isinstance(x,(float,np.floating)):
        return None if np.isnan(x) else ('Infinity' if np.isposinf(x) else '-Infinity' if np.isneginf(x) else float(x))
    return x


def _publish(temp,path,immutable):
    if immutable:
        # link is atomic and refuses overwrite; a stale/orphan file is evidence.
        os.link(temp,path);temp.unlink()
    else:os.replace(temp,path)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def write_json(path,value,immutable=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    try:
        with tmp.open('x') as f:
            json.dump(clean(value),f,indent=2,ensure_ascii=False,allow_nan=False)
            f.write('\n');f.flush();os.fsync(f.fileno())
        _publish(tmp,path,immutable)
    finally:
        if tmp.exists():tmp.unlink()


@contextlib.contextmanager
def lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    f=path.open('a+')
    try:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('S3 ownership lock held: '+str(path)) from None
        yield
    finally:f.close()


def environment():return dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,shapely=shapely.__version__)


def verify_v1():
    base=ROOT.parent;m=json.loads((base/'FROZEN.json').read_text())
    for f,h in m['files'].items():
        if sha(base/f)!=h:raise RuntimeError('Frozen v1 source differs: '+f)
    return sha(base/'FROZEN.json')


def freeze():
    if (ROOT/'FROZEN.json').exists():return verify()
    files=list(ROOT.glob('*.py'))+[ROOT/n for n in ['EVENT-CORRECTION.md','RUNTIME-AMENDMENT.md','IMPLEMENTATION-AMENDMENT.md','protocol.json','cells.json','SOURCE-SNAPSHOTS.json','CONTRACT.json','VERIFIER-CORRECTION.md']]
    files += [p for p in (ROOT/'snapshot').rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json')]
    inputs=[]
    for c in json.loads((ROOT/'cells.json').read_text()):
        if sha(c['cache_path'])!=c['cache_sha256']:raise RuntimeError('Accepted archive mismatch')
        with np.load(c['cache_path'],allow_pickle=False) as z:
            if str(z['source_sha256'])!=c['sha256']:raise RuntimeError('Matching source mismatch')
        inputs.append(dict(path=c['cache_path'],sha256=c['cache_sha256']))
    origin={str(ROOT.parent/n):sha(ROOT.parent/n) for n in ['FROZEN.json','pilot/PILOT.json','pilot/VERIFIED.json','PARENT-READINESS.json']}
    origin.update({str(p):sha(p) for p in [ROOT.parent/'v2/FROZEN.json',ROOT.parent/'v2/pilot/PILOT.json',ROOT.parent/'v2/pilot.log']})
    m=dict(schema='S3-frozen-execution-v3',utc=now(),environment=environment(),files={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)},inputs=inputs,
           v1_frozen_sha256=verify_v1(),origin=origin,status='FROZEN_BEFORE_V2_PILOT_AND_FULL',requested_model='gpt-6-astra',actual_model='parent verifies runtime')
    write_json(ROOT/'FROZEN.json',m,immutable=True)
    return sha(ROOT/'FROZEN.json')


def verify(check_inputs=True,cell=None):
    m=json.loads((ROOT/'FROZEN.json').read_text())
    if m['environment']!=environment():raise RuntimeError('Environment changed')
    for f,h in m['files'].items():
        if sha(ROOT/f)!=h:raise RuntimeError('Source changed: '+f)
    for f,h in m['origin'].items():
        if sha(f)!=h:raise RuntimeError('v1 provenance changed: '+f)
    if check_inputs:
        for inp in m['inputs']:
            if cell is None or inp['path']==cell['cache_path']:
                if sha(inp['path'])!=inp['sha256']:raise RuntimeError('Input changed: '+inp['path'])
    return sha(ROOT/'FROZEN.json')


def cap(cpu=None,limit=LIMIT):
    if not 0<limit<=LIMIT:raise ValueError('Worker AS limit exceeds frozen cap')
    _,hard=resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS,(limit if hard<0 else min(limit,hard),hard))
    for k in THREADS:
        if os.environ.get(k)!='1':raise RuntimeError('Thread setting not1: '+k)
    available=sorted(os.sched_getaffinity(0))
    cpu=available[0] if cpu is None else int(cpu)
    if cpu not in available:raise ValueError('CPU outside inherited affinity')
    os.sched_setaffinity(0,{cpu})
    return cpu


def save_result(path,result,frozen_hash,cell_index,draw):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() or path.with_suffix('.json').exists():raise RuntimeError('Existing checkpoint: no overwrite')
    tmp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    try:
        with tmp.open('xb') as f:
            np.savez_compressed(f,**result);f.flush();os.fsync(f.fileno())
        _publish(tmp,path,True)
    finally:
        if tmp.exists():tmp.unlink()
    write_json(path.with_suffix('.json'),dict(file=path.name,sha256=sha(path),frozen_hash=frozen_hash,cell_index=cell_index,
        draw_id=draw,utc=now(),rng_key=[20260928,3,cell_index,draw],compression='lossless NPZ DEFLATE; original dtypes/shapes preserved',
        restart='independent fixed draw/role seeds; any orphan refuses automatic overwrite'),immutable=True)


def load_result(path,frozen_hash,cell_index=None,draw=None):
    path=Path(path)
    if not path.exists() or not path.with_suffix('.json').exists():raise RuntimeError('Incomplete/orphan checkpoint: '+str(path))
    m=json.loads(path.with_suffix('.json').read_text())
    if m['file']!=path.name or m['sha256']!=sha(path) or m['frozen_hash']!=frozen_hash:raise RuntimeError('Checkpoint integrity')
    if cell_index is not None and m['cell_index']!=cell_index:raise RuntimeError('Checkpoint cell identity')
    if draw is not None and m['draw_id']!=draw:raise RuntimeError('Checkpoint draw identity')
    if m['rng_key']!=[20260928,3,m['cell_index'],m['draw_id']]:raise RuntimeError('Checkpoint RNG identity')
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def gate(frozen,workers,path=None):
    path=ROOT/'PARENT-SCHEDULE.json' if path is None else Path(path)
    if not path.exists():raise RuntimeError('Full blocked: missing parent launch gate')
    p=json.loads(path.read_text())
    required=(p.get('approve_full') is True and p.get('frozen_sha256')==frozen and p.get('B')==5000
              and p.get('cells')=='all16' and p.get('independent_review_passed') is True)
    if not required:raise RuntimeError('Full blocked: gate does not approve this exact protocol/review')
    count=p.get('max_workers')
    if type(count)!=int or not 1<=workers<=count<=6:raise RuntimeError('Parent/frozen worker limit')
    limit=p.get('worker_address_space_bytes',LIMIT)
    if type(limit)!=int or not 0<limit<=LIMIT:raise RuntimeError('Parent/frozen memory limit')
    if type(p.get('max_cpu_seconds')) not in (int,float) or not math.isfinite(p['max_cpu_seconds']) or p['max_cpu_seconds']<=0:raise RuntimeError('Finite positive parent CPU budget required')
    if type(p.get('max_wall_seconds')) not in (int,float) or not math.isfinite(p['max_wall_seconds']) or p['max_wall_seconds']<=0:raise RuntimeError('Finite positive parent wall budget required')
    reserve=p.get('minimum_free_disk_bytes',15*1024**3)
    if not isinstance(reserve,int) or reserve<15*1024**3:raise RuntimeError('At least15GiB stop-boundary reserve required')
    launch=p.get('launch_minimum_free_disk_bytes',100*1024**3)
    if not isinstance(launch,int) or launch<100*1024**3:raise RuntimeError('At least100GiB free disk required at launch')
    p['launch_minimum_free_disk_bytes']=launch
    p['worker_address_space_bytes']=limit;p['minimum_free_disk_bytes']=reserve
    p['gate_sha256']=sha(path)
    return p


def health():
    return dict(utc=now(),pid=os.getpid(),rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                cpu_affinity=sorted(os.sched_getaffinity(0)),address_space_soft_limit_bytes=resource.getrlimit(resource.RLIMIT_AS)[0])

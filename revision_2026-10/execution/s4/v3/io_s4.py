"""Owner-only immutable checkpoints, source/input hashes and resource guard."""
from config import ROOT
from contextlib import contextmanager
import fcntl,hashlib,json,math,os,resource,sys,uuid
from pathlib import Path
import numpy as np
import scipy
import shapely
from resources import tick,write_allowance,write_finished


def now():
 from datetime import datetime,timezone
 return datetime.now(timezone.utc).isoformat()


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):
   tick();h.update(b)
 return h.hexdigest()


def require(ok,message):
 if not ok:raise ValueError(message)


def owned(p):
 p=Path(p).resolve();require(p.is_relative_to(ROOT),'Output outside S4 ownership');return p


def clean(v):
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,np.ndarray):return clean(v.tolist())
 if isinstance(v,(bool,np.bool_)):return bool(v)
 if isinstance(v,np.integer):return int(v)
 if isinstance(v,(float,np.floating)):
  return None if np.isnan(v) else 'Infinity' if np.isposinf(v) else '-Infinity' if np.isneginf(v) else float(v)
 return v


def publish(temp,path):
 os.link(temp,path);temp.unlink()
 fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)


def write_json(p,value,replace=False):
 p=owned(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp-'+uuid.uuid4().hex)
 try:
  payload=json.dumps(clean(value),indent=2,allow_nan=False)+'\n'
  write_allowance(p,len(payload.encode()))
  with tmp.open('x') as f:f.write(payload);f.flush();os.fsync(f.fileno())
  if replace:os.replace(tmp,p)
  else:publish(tmp,p)
  write_finished(p)
 finally:
  if tmp.exists():tmp.unlink()


def json_rows(p):
 with Path(p).open() as f:
  for l in f:
   tick()
   if l.strip():yield json.loads(l)


def write_rows(p,rows):
 p=owned(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:
  for r in rows:f.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
  f.flush();os.fsync(f.fileno())


@contextmanager
def lock(p):
 p=owned(p);p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a+')
 try:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError('S4 ownership busy '+str(p)) from None
  yield
 finally:f.close()


def save_npz(p,arrays,metadata):
 p=owned(p);p.parent.mkdir(parents=True,exist_ok=True)
 require(not p.exists() and not p.with_suffix('.json').exists(),'Checkpoint exists/orphan; no overwrite')
 tmp=p.with_name(p.name+'.tmp-'+uuid.uuid4().hex)
 try:
  write_allowance(p,sum(a.nbytes for a in arrays.values())+16384+len(arrays)*512)
  with tmp.open('xb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
  publish(tmp,p)
  write_finished(p)
 finally:
  if tmp.exists():tmp.unlink()
 write_json(p.with_suffix('.json'),dict(metadata,file=p.name,sha256=sha(p)))


def load_npz(p,expected):
 p=Path(p);require(p.exists() and p.with_suffix('.json').exists(),'Missing/orphan checkpoint')
 m=json.loads(p.with_suffix('.json').read_text())
 require(m['file']==p.name and m['sha256']==sha(p),'Checkpoint bytes mismatch')
 for k,v in expected.items():require(m.get(k)==v,'Checkpoint metadata mismatch '+k)
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def environment():return dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,shapely=shapely.__version__)


def freeze():
 p=ROOT/'FROZEN.json'
 if p.exists():return verify()
 files=[*ROOT.glob('*.py'),*ROOT.glob('*.md'),ROOT/'SOURCE-SNAPSHOTS.json',ROOT/'protocol.json',ROOT/'V1-LINEAGE.json',ROOT/'AMENDMENT.json',ROOT/'PREPARED-REUSE.json',ROOT/'NUMERICAL-PERFORMANCE-AMENDMENT.json']
 files.extend(x for base in ['vendor','snapshot','tests'] for x in (ROOT/base).rglob('*') if x.is_file() and x.suffix in ('.py','.md','.json'))
 write_json(p,dict(schema='S4-source-freeze-v2',utc=now(),files={str(x.relative_to(ROOT)):sha(x) for x in sorted(files)},environment=environment(),model='gpt-6-astra parent metadata verified',status='ISOLATED_IMPLEMENTATION_REPAIR_AFTER_FAILED_V2_PILOT_BEFORE_V3_PILOT',protocol_unchanged=True,original_freeze_sha256='863b43423d500aa2dacfd7bd7cedff7534a728e31c5c85aec36cc6d6714f917c'))
 return sha(p)


def verify():
 m=json.loads((ROOT/'FROZEN.json').read_text());require(m['environment']==environment(),'Environment drift')
 for p,h in m['files'].items():require(sha(ROOT/p)==h,'Source drift '+p)
 return sha(ROOT/'FROZEN.json')


class InputGuard:
 def __init__(self,files):
  self.files=files;self.stats={}
  for name,h in files.items():
   a=self.stat(name);require(sha(name)==h,'Input hash mismatch '+name)
   require(self.stat(name)==a,'Input changed during hash');self.stats[name]=a
 def stat(self,p):
  s=Path(p).stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 def check(self,full=False):
  for p,h in self.files.items():
   tick()
   require(self.stat(p)==self.stats[p],'Input stat changed '+p)
   if full:require(sha(p)==h and self.stat(p)==self.stats[p],'Input hash changed '+p)


def cap():
 limit=4*1024**3;_,h=resource.getrlimit(resource.RLIMIT_AS)
 resource.setrlimit(resource.RLIMIT_AS,(limit if h<0 else min(limit,h),h))
 cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpus[0]})
 for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:require(os.environ.get(k)=='1','BLAS setting')


def health():
 status=Path('/proc/self/status').read_text().splitlines()
 peak_vms=int(next(line.split()[1] for line in status if line.startswith('VmPeak:')))*1024
 return dict(utc=now(),pid=os.getpid(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_virtual_address_bytes=peak_vms,cpu_affinity=sorted(os.sched_getaffinity(0)))

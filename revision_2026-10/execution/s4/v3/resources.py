"""Cumulative lifecycle accounting; cooperative stop, never kill a scientific task.

The independent watchdog process samples the exact worker's /proc identity even
while a long Python/native call runs. It requests stop; main checks at bounded
I/O/source/draw/interval boundaries. No claim of instantaneous hard cancellation.
"""
from pathlib import Path
import json,os,time,math,shutil,resource,subprocess,sys,uuid

CURRENT=None
GIB=1024**3

def tick(stage=None,allocation=0):
 if CURRENT is not None:CURRENT.check(stage,allocation)

def write_allowance(path,size):
 if CURRENT is not None:CURRENT.before_write(path,size)

def write_finished(path):
 if CURRENT is not None:CURRENT.after_write(path)

def _dump(path,doc):
 temp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
 with temp.open('x') as f:json.dump(doc,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(temp,path)

def _identity(pid):
 text=Path(f'/proc/{pid}/stat').read_text();a=text[text.rfind(')')+2:].split()
 return a[19],(int(a[11])+int(a[12]))/os.sysconf('SC_CLK_TCK')

def available_memory():
 for line in Path('/proc/meminfo').read_text().splitlines():
  if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
 raise RuntimeError('Cannot read available memory')

class BudgetStop(RuntimeError):pass

class Lifecycle:
 def __init__(self,out,limits,identity,monitor=True,clock=time.monotonic,cpu_clock=time.process_time,disk=None,memory=None):
  self.out=Path(out).resolve();self.out.mkdir(parents=True,exist_ok=True)
  self.limits=dict(limits);self.identity=identity;self.clock=clock;self.cpu_clock=cpu_clock
  self.disk=disk or (lambda:shutil.disk_usage(self.out).free);self.memory=memory or available_memory
  for k in ['max_cpu_seconds','max_wall_seconds','max_output_bytes','minimum_free_disk_bytes','minimum_available_memory_bytes']:
   v=self.limits.get(k)
   if not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:raise ValueError('Finite positive lifecycle limit '+k)
  if 'deadline_unix_seconds' in self.limits:
   if not math.isfinite(self.limits['deadline_unix_seconds']) or self.limits['deadline_unix_seconds']<=0:raise ValueError('Invalid absolute deadline')
  self.path=self.out/'LIFECYCLE.json';self.stage='input_acceptance';self.started=self.clock();self.cpu_started=0.
  self.previous_cpu=0.;self.previous_wall=0.;self.previous_child_cpu=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime;self.monitor_enabled=monitor;self.monitor=None
  self.attempt=0;self.sizes={str(p):p.stat().st_size for p in self.out.rglob('*') if p.is_file()}
  if self.path.exists():
   old=json.loads(self.path.read_text())
   if old['identity']!=identity or old['limits']!=self.limits:raise ValueError('Lifecycle budget/identity changed; explicit new version required')
   if old['status']=='ACTIVE':raise BudgetStop('Unclean prior attempt: preserve files; parent audit required, cannot reset cumulative budget')
   if old['status']=='COMPLETE':raise BudgetStop('Lifecycle already complete')
   self.previous_cpu=old['cpu_seconds'];self.previous_wall=old['wall_seconds'];self.attempt=old['attempt']+1
  self.initial_bytes=sum(self.sizes.values());self.current_bytes=self.initial_bytes
  self.stop_file=self.out/f'budget-stop-{self.attempt}.json';self.done_file=self.out/f'watchdog-done-{self.attempt}.json'
  self.last_persist=-math.inf;self.last_check=-math.inf;self.events=[];self.status='ACTIVE';self.check('input_acceptance')
  if monitor:
   start_id,_=_identity(os.getpid())
   spec=dict(pid=os.getpid(),start_id=start_id,limits=self.limits,previous_cpu=self.previous_cpu,previous_wall=self.previous_wall,
             start_monotonic=self.started,attempt=self.attempt,stop=str(self.stop_file),done=str(self.done_file),out=str(self.out))
   spec_path=self.out/f'watchdog-spec-{self.attempt}.json';_dump(spec_path,spec)
   self.monitor=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--watch',str(spec_path)],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 def counters(self):return self.previous_cpu+self.cpu_clock(),self.previous_wall+self.clock()-self.started
 def _state(self,status):
  cpu,wall=self.counters()
  return dict(schema='S4-lifecycle-budget-v2',status=status,identity=self.identity,limits=self.limits,attempt=self.attempt,stage=self.stage,
              cpu_seconds=cpu,wall_seconds=wall,current_output_bytes=self.current_bytes,initial_output_bytes=self.initial_bytes,
              pid=os.getpid(),utc_seconds=time.time(),stop_semantics='cooperative next safe boundary; watchdog never sends signals',events=self.events)
 def persist(self,status=None):
  _dump(self.path,self._state(status or self.status));self.last_persist=self.clock()
 def check(self,stage=None,allocation=0):
  changed=stage is not None and stage!=self.stage
  if not changed and not allocation and self.clock()-self.last_check<.10:return
  self.last_check=self.clock()
  if changed:
   cpu,wall=self.counters();self.events.append(dict(stage=self.stage,cpu_seconds=cpu,wall_seconds=wall));self.stage=stage
  cpu,wall=self.counters();reasons=[]
  if self.stop_file.exists():reasons.append('independent watchdog requested stop')
  if time.time()>=self.limits.get('deadline_unix_seconds',math.inf):reasons.append('absolute schedule deadline reached')
  if cpu>=self.limits['max_cpu_seconds']:reasons.append('cumulative CPU budget exhausted')
  if wall>=self.limits['max_wall_seconds']:reasons.append('cumulative wall budget exhausted')
  if self.disk()<self.limits['minimum_free_disk_bytes']+int(allocation):reasons.append('disk reserve/allocation refused')
  if self.current_bytes+int(allocation)>self.limits['max_output_bytes']:reasons.append('cumulative retained-output capacity exhausted')
  if self.memory()<self.limits['minimum_available_memory_bytes']:reasons.append('available memory below reserve')
  if reasons:self.persist('STOPPED');raise BudgetStop('; '.join(reasons))
  if self.clock()-self.last_persist>=1:self.persist()
 def before_write(self,path,size):
  path=Path(path).resolve()
  if not path.is_relative_to(self.out):raise BudgetStop('Lifecycle write outside run output')
  # Atomic replacement temporarily needs the entire new file on disk.
  self.check(allocation=max(0,int(size)))
 def after_write(self,path):
  p=Path(path).resolve();key=str(p);size=p.stat().st_size if p.exists() else 0
  self.current_bytes+=size-self.sizes.get(key,0);self.sizes[key]=size
  self.check()
 def preflight(self,plan):
  for key in ['cpu_seconds','wall_seconds','peak_additional_bytes']:
   if key not in plan or not math.isfinite(plan[key]) or plan[key]<0:raise ValueError('Invalid lifecycle projection '+key)
  cpu,wall=self.counters()
  if time.time()+plan['wall_seconds']>self.limits.get('deadline_unix_seconds',math.inf):raise BudgetStop('Projected complete lifecycle exceeds absolute deadline')
  if cpu+plan['cpu_seconds']>self.limits['max_cpu_seconds'] or wall+plan['wall_seconds']>self.limits['max_wall_seconds']:raise BudgetStop('Complete lifecycle including verification exceeds remaining schedule')
  self.check('full_lifecycle_preflight',plan['peak_additional_bytes'])
 def finish(self,status):
  global CURRENT
  if self.monitor is not None:
   _dump(self.done_file,dict(status=status));self.monitor.wait(timeout=5)
   # Watchdog overhead itself is charged; it is the only child this suite starts.
   self.previous_cpu+=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime-self.previous_child_cpu
  self.status=status;self.persist(status);CURRENT=None
 def __enter__(self):
  global CURRENT
  if CURRENT is not None:raise RuntimeError('Nested lifecycle')
  CURRENT=self;return self
 def __exit__(self,typ,value,tb):self.finish('COMPLETE' if typ is None else 'STOPPED')

def watchdog(spec_path):
 spec=json.loads(Path(spec_path).read_text());out=Path(spec['out']);last=-math.inf
 log=out/f'HEALTH-{spec["attempt"]}.jsonl';clock_start=time.process_time()
 while not Path(spec['done']).exists():
  try:
   identity,cpu=_identity(spec['pid'])
   if identity!=spec['start_id']:raise RuntimeError('Worker PID recycled')
   total_cpu=spec['previous_cpu']+cpu+time.process_time()-clock_start
   wall=spec['previous_wall']+time.monotonic()-spec['start_monotonic'];free=shutil.disk_usage(out).free;mem=available_memory();l=spec['limits']
   reasons=[]
   if time.time()>=l.get('deadline_unix_seconds',math.inf):reasons.append('absolute_deadline')
   if total_cpu>=l['max_cpu_seconds']:reasons.append('cpu')
   if wall>=l['max_wall_seconds']:reasons.append('wall')
   if free<l['minimum_free_disk_bytes']:reasons.append('disk')
   if mem<l['minimum_available_memory_bytes']:reasons.append('memory')
   row=dict(pid=spec['pid'],start_id=identity,utc_seconds=time.time(),cpu_seconds=total_cpu,wall_seconds=wall,free_disk_bytes=free,available_memory_bytes=mem,reasons=reasons)
   if reasons and not Path(spec['stop']).exists():_dump(Path(spec['stop']),row)
   if time.monotonic()-last>=30 or reasons:
    with log.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
    last=time.monotonic()
   if reasons:return
  except (FileNotFoundError,ProcessLookupError):return
  except Exception as e:
   if not Path(spec['stop']).exists():_dump(Path(spec['stop']),dict(error=repr(e)))
   return
  time.sleep(.5)

if __name__=='__main__':
 if len(sys.argv)!=3 or sys.argv[1]!='--watch':raise SystemExit('Only --watch <owned-spec>')
 watchdog(sys.argv[2])

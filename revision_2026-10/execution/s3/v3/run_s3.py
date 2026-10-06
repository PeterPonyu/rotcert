#!/usr/bin/env python
"""Cell-parallel orchestration. Pilot one CPU; full requires exact parent gate."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
import json
from pathlib import Path
import resource
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
import numpy as np
from s3_io import ROOT,LIMIT,write_json,lock,freeze,verify,cap,save_result,load_result,sha,now,gate,health
from s3_integrity import InputGuard,bind_roles,canonical_roles,validate_science,checkpoint_prefix
from s3_scores import all_static_scores,ARMS
from s3_geometry import METRICS,STRATA,GEOMETRY_REASON
from s3_pipeline import Cell
from s3_parity import parity


def load_cache(cell):
    with np.load(cell['cache_path'],allow_pickle=False) as z:return {k:z[k] for k in z.files}


def stop_requested(out,run_id,start,permit,gate_hash):
    launch=json.loads((out/'control'/('RUN-'+run_id+'.json')).read_text())
    if os.getppid()!=launch['scheduler_pid']:return 'scheduler parent exited'
    if (out/'control'/('STOP-'+run_id+'.json')).exists():return 'scheduler stop request'
    if permit:
        if sha(ROOT/'PARENT-SCHEDULE.json')!=gate_hash:return 'parent gate changed'
        if time.monotonic()-start>=permit['max_wall_seconds']:return 'parent wall budget reached'
    return None


def worker(mode,index,cpu,run_id,gate_hash=None):
    if mode not in ('pilot','full') or not re.fullmatch(r'[0-9a-f]{32}',run_id or ''):raise RuntimeError('invalid worker invocation')
    launch=json.loads((ROOT/mode/'control'/('RUN-'+run_id+'.json')).read_text())
    ticket=json.loads((ROOT/mode/'control'/('TICKET-'+run_id+'-'+str(index)+'.json')).read_text())
    if launch['scheduler_pid']!=os.getppid() or ticket!={'run_id':run_id,'cell_index':index,'cpu':cpu,'scheduler_pid':os.getppid()}:
        raise RuntimeError('worker must be a direct, ticketed child of active scheduler')
    cells=json.loads((ROOT/'cells.json').read_text());cell=cells[index]
    if cell['cell_index']!=index:raise RuntimeError('cell IDs must match frozen order')
    frozen=verify(cell=cell)
    if launch['frozen_sha256']!=frozen or launch['mode']!=mode:raise RuntimeError('worker launch identity')
    permit=gate(frozen,launch['workers']) if mode=='full' else None
    if permit and permit['gate_sha256']!=gate_hash:raise RuntimeError('gate changed before worker startup')
    cap(cpu,permit['worker_address_space_bytes'] if permit else LIMIT)
    out=ROOT/mode;cid=cell['cell_id'];target=out/cid
    B=2 if mode=='pilot' else 5000;start=time.monotonic();cput=time.process_time()
    with lock(ROOT/'locks'/('cell-'+cid+'.lock')):
        guard=InputGuard(cell['cache_path'],cell['cache_sha256'])
        cache=load_cache(cell);guard.check()
        prepared=ROOT/'prepared'/cid;prepared.mkdir(parents=True,exist_ok=True)
        if any(prepared.glob('*.tmp-*')):raise RuntimeError('orphan prepared temporary')
        sp=prepared/'static_scores.npz'
        if sp.exists() or sp.with_suffix('.json').exists():static=load_result(sp,frozen,index,-2)
        else:
            static=all_static_scores(cache);save_result(sp,static,frozen,index,-2)
        roles=bind_roles(prepared,cache,cell,frozen);functional=Cell(cache,static)
        for a,b in zip(roles,functional.roles):np.testing.assert_array_equal(a,b)
        p=prepared/'PARITY.json'
        if p.exists():
            saved=json.loads(p.read_text())
            if saved.get('frozen_sha256')!=frozen:raise RuntimeError('parity freeze mismatch')
        else:write_json(p,dict(frozen_sha256=frozen,**parity(cache,static,cell)),immutable=True)
        guard.check(force_hash=True);verify(check_inputs=False)
        prefix=checkpoint_prefix(target,B);target.mkdir(exist_ok=True)
        runstate=target/('state-'+run_id+'.json')
        records=[]
        for draw in [-1]+list(range(B)):
            path=target/('base.npz' if draw<0 else f'draw-{draw:04d}.npz')
            draw_started=time.process_time()
            # This stat check is performed on each boundary; periodic full hash below.
            guard.check()
            stopped=stop_requested(out,run_id,start,permit,gate_hash)
            if stopped:
                write_json(runstate,dict(status='PAUSED_SAFE_BOUNDARY',reason=stopped,run_id=run_id,
                    cell=cid,cpu_seconds=time.process_time()-cput,frozen_sha256=frozen,**health()))
                return 20
            if permit and shutil.disk_usage(ROOT).free<permit['minimum_free_disk_bytes']+10*1024**2:
                raise RuntimeError('disk reserve reached before next checkpoint')
            if draw in prefix:r=load_result(path,frozen,index,draw)
            else:
                r=functional.run(index,draw)
                validate_science(r,cache,roles,index,draw)
                guard.check();save_result(path,r,frozen,index,draw)
            validate_science(r,cache,roles,index,draw)
            records.append(dict(draw=draw,cpu_seconds=float(r['total_cpu_seconds']),
                compressed_bytes=path.stat().st_size,raw_array_bytes=sum(v.nbytes for v in r.values()),
                fit_check_save_cpu_seconds=time.process_time()-draw_started,reused_checkpoint=(draw in prefix),
                tuned_alpha_available=int(np.isfinite(r['selected_alpha'][:,1]).sum())))
            write_json(runstate,dict(status='RUNNING',run_id=run_id,cell=cid,cell_index=index,draw_completed=draw,
                complete_draws=max(0,draw+1),target_draws=B,cpu_seconds=time.process_time()-cput,
                wall_seconds=time.monotonic()-start,frozen_sha256=frozen,**health()))
            if draw>=0 and draw%25==0:verify(check_inputs=False)
            if draw>=0 and draw%100==0:guard.check(force_hash=True)
        guard.check(force_hash=True);verify(check_inputs=False)
        receipt=dict(status='COMPLETE',mode=mode,cell=cid,cell_index=index,draws_per_cell=B,frozen_sha256=frozen,
                     total_cpu_seconds=time.process_time()-cput,total_wall_seconds=time.monotonic()-start,
                     records=records,**health())
        complete=target/'COMPLETE.json'
        if complete.exists():
            old=json.loads(complete.read_text())
            if old['frozen_sha256']!=frozen or old['draws_per_cell']!=B:raise RuntimeError('complete manifest identity')
        else:write_json(complete,receipt,immutable=True)
        write_json(runstate,dict(status='COMPLETE',run_id=run_id,cell=cid,cpu_seconds=time.process_time()-cput,
                                frozen_sha256=frozen,**health()))
        print(json.dumps(dict(status='COMPLETE',mode=mode,cell=cid,wall_seconds=receipt['total_wall_seconds'],cpu_seconds=receipt['total_cpu_seconds'])),flush=True)
    return 0


def make_stop(out,run_id,reason):
    p=out/'control'/('STOP-'+run_id+'.json')
    if not p.exists():write_json(p,dict(reason=reason,utc=now(),action='finish current draw, checkpoint then exit; no signals or kills'),immutable=True)


def available_memory():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    raise RuntimeError('memory availability unavailable')


def schedule(mode,workers):
    with lock(ROOT/'locks/scheduler.lock'):
        frozen=verify();permit=gate(frozen,workers) if mode=='full' else None
        if mode=='pilot' and workers!=1:raise RuntimeError('pilot single CPU only')
        cpus=sorted(os.sched_getaffinity(0))
        if workers>len(cpus):raise RuntimeError('fewer available CPUs than workers')
        if mode=='pilot':os.sched_setaffinity(0,{cpus[0]})
        if permit:
            if available_memory()<workers*permit['worker_address_space_bytes']+2*1024**3:
                raise RuntimeError('insufficient available RAM for bounded worker allocation')
            readiness=ROOT/'PARENT-READINESS.json'
            if not readiness.exists():raise RuntimeError('measured v2 readiness required')
            estimate=json.loads(readiness.read_text())
            if estimate['frozen_sha256']!=frozen:raise RuntimeError('readiness hash differs')
            if estimate['projected_full_cpu_seconds_safety1p5']>permit['max_cpu_seconds']:
                raise RuntimeError('measured full CPU projection exceeds parent budget')
            if estimate['projected_parallel_wall_seconds_safety1p5'][str(workers)]>permit['max_wall_seconds']:
                raise RuntimeError('measured wall projection exceeds parent budget')
            if shutil.disk_usage(ROOT).free<max(permit['launch_minimum_free_disk_bytes'],estimate['projected_compressed_draw_bytes_safety1p5']+permit['minimum_free_disk_bytes']):
                raise RuntimeError('insufficient disk for conservative full projection plus reserve')
        out=ROOT/mode;out.mkdir(exist_ok=True);run_id=uuid.uuid4().hex
        write_json(out/'control'/('RUN-'+run_id+'.json'),dict(run_id=run_id,mode=mode,frozen_sha256=frozen,
                   workers=workers,scheduler_pid=os.getpid(),gate_hash=permit['gate_sha256'] if permit else None,utc=now()),immutable=True)
        cells=json.loads((ROOT/'cells.json').read_text());B=2 if mode=='pilot' else 5000
        for c in cells:checkpoint_prefix(out/c['cell_id'],B)
        # Unchanging frozen cell order; no result-based dispatch ordering.
        waiting=list(range(len(cells)));active={};handles=[];errors=[]
        start=time.monotonic();finished=[];stopping=None;slots=list(cpus[:workers])
        states={};prior_cpu=0.;last_health=0.
        try:
            while waiting or active:
                if permit:
                    if sha(ROOT/'PARENT-SCHEDULE.json')!=permit['gate_sha256']:stopping='parent gate changed'
                    if time.monotonic()-start>=permit['max_wall_seconds']:stopping='parent wall budget reached'
                    if shutil.disk_usage(ROOT).free<permit['minimum_free_disk_bytes']+workers*10*1024**2:stopping='disk reserve reached'
                if stopping:
                    make_stop(out,run_id,stopping);waiting.clear()
                while waiting and slots and not stopping:
                    i=waiting.pop(0);cpu=slots.pop(0);cid=cells[i]['cell_id']
                    log=out/'logs'/(cid+'-'+run_id+'.log');log.parent.mkdir(exist_ok=True)
                    f=log.open('x');handles.append(f)
                    cmd=[sys.executable,'-B',str(ROOT/'run_s3.py'),'worker','--mode',mode,'--cell',str(i),
                         '--cpu',str(cpu),'--run-id',run_id]
                    if permit:cmd+=['--gate-hash',permit['gate_sha256']]
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
                    write_json(out/'control'/('TICKET-'+run_id+'-'+str(i)+'.json'),dict(run_id=run_id,cell_index=i,cpu=cpu,scheduler_pid=os.getpid()),immutable=True)
                    proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,env=env)
                    active[i]=(proc,cpu)
                for i,(proc,cpu) in list(active.items()):
                    state=out/cells[i]['cell_id']/('state-'+run_id+'.json')
                    if state.exists():states[i]=json.loads(state.read_text())
                    code=proc.poll()
                    if code is not None:
                        del active[i];slots.append(cpu)
                        if code==0:finished.append(i)
                        else:
                            errors.append(dict(cell=cells[i]['cell_id'],exit_code=code));stopping='worker failure or safe pause'
                        print(json.dumps(dict(cell=cells[i]['cell_id'],exit_code=code,complete_cells=len(finished))),flush=True)
                consumed=sum(float(s.get('cpu_seconds',0)) for s in states.values())
                # Include live process CPU to bound a slow current draw, using /proc
                # only for PIDs created by this invocation. Sampling never signals.
                live=0.
                for i,(proc,_) in active.items():
                    try:
                        fields=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()
                        live+=max(float(states.get(i,{}).get('cpu_seconds',0)),(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK'))
                    except FileNotFoundError:live+=float(states.get(i,{}).get('cpu_seconds',0))
                consumed=sum(float(s.get('cpu_seconds',0)) for i,s in states.items() if i not in active)+live
                if permit and consumed>=permit['max_cpu_seconds']:stopping='parent CPU budget reached'
                if time.monotonic()-last_health>=30 or not active:
                    write_json(out/'control'/('state-'+run_id+'.json'),dict(status='DRAINING' if stopping else 'RUNNING',
                       reason=stopping,run_id=run_id,active_cells=list(active),complete_cells=finished,errors=errors,
                       cpu_seconds=consumed,wall_seconds=time.monotonic()-start,frozen_sha256=frozen,utc=now(),
                       available_memory_bytes=available_memory(),free_disk_bytes=shutil.disk_usage(ROOT).free,
                       children=[dict(cell=i,pid=p.pid,cpu=cpu) for i,(p,cpu) in active.items()]))
                    last_health=time.monotonic()
                if active:time.sleep(1)
        except BaseException as exc:
            make_stop(out,run_id,'scheduler exception '+repr(exc))
            # Own workers observe STOP only after their current atomic draw. Never
            # signal/kill an unrelated process. Drain without spawning new work.
            for proc,_ in active.values():proc.wait()
            raise
        finally:
            for f in handles:f.close()
        if stopping or errors:
            write_json(out/'control'/('STOPPED-'+run_id+'.json'),dict(status='STOPPED_WITH_CHECKPOINTS',reason=stopping,errors=errors,utc=now()),immutable=True)
            raise RuntimeError('Run incomplete; preserved all evidence: '+str(stopping))
        verify();receipts=[json.loads((out/c['cell_id']/'COMPLETE.json').read_text()) for c in cells]
        report=dict(status='COMPLETE',mode=mode,frozen_sha256=frozen,run_id=run_id,workers=workers,
            cells=receipts,total_worker_cpu_seconds=sum(r['total_cpu_seconds'] for r in receipts),
            current_invocation_wall_seconds=time.monotonic()-start,metrics=METRICS,arms=ARMS,strata=STRATA,geometry=GEOMETRY_REASON,
            full_not_launched=(mode=='pilot'),utc=now())
        name='PILOT.json' if mode=='pilot' else 'RUN.json'
        if not (out/name).exists():write_json(out/name,report,immutable=True)
        if not (out/'COMPLETE.json').exists():write_json(out/'COMPLETE.json',dict(status='COMPLETE',mode=mode,frozen_sha256=frozen,
                    cells=16,draws_per_cell=B,utc=now()),immutable=True)
        # All exact children have exited; verification runs while the scheduler
        # lock remains held, and acquires each cell lock independently.
        from verify_s3 import main as verify_completed
        verify_completed(mode)
        print(json.dumps({k:report[k] for k in ['status','mode','total_worker_cpu_seconds','current_invocation_wall_seconds','workers']}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','pilot','full','worker'])
    p.add_argument('--workers',type=int,default=1);p.add_argument('--mode',choices=['pilot','full'])
    p.add_argument('--cell',type=int);p.add_argument('--cpu',type=int);p.add_argument('--run-id');p.add_argument('--gate-hash')
    a=p.parse_args()
    if a.command=='freeze':
        with lock(ROOT/'locks/scheduler.lock'):print(freeze())
        return 0
    if a.command=='worker':return worker(a.mode,a.cell,a.cpu,a.run_id,a.gate_hash)
    if not 1<=a.workers<=6:raise RuntimeError('maximum6 workers')
    schedule(a.command,a.workers);return 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as exc:
        write_json(ROOT/'errors'/f'ERROR-{os.getpid()}-{time.time_ns()}.json',dict(error=repr(exc),traceback=traceback.format_exc(),utc=now()),immutable=True)
        raise

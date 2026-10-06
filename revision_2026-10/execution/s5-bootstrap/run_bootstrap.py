#!/usr/bin/env python3
"""Bounded pilot / independently gated full run, one writer and one CPU worker."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse,contextlib,datetime,fcntl,hashlib,json,platform,resource,signal,sys,threading,time,traceback,uuid
import scipy
import numpy as np
from bootstrap_core import CellData,ROLES,ROUTES,METRICS,SUPPORT,sha

HERE=Path(__file__).resolve().parent


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def atomic_json(path,data):
    path=Path(path);tmp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    encoded=(json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    with tmp.open('xb') as f:f.write(encoded);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    directory=os.open(path.parent,os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)


@contextlib.contextmanager
def writer_lock(path):
    with Path(path).open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('another owned S5 bootstrap writer holds the lock; not started')
        yield


def check_freeze(expected_signature=None):
    """Validate one stable manifest and anchor all rechecks to the run's identity."""
    manifest=HERE/'SOURCE-FREEZE.json'
    raw=manifest.read_bytes();signature=hashlib.sha256(raw).hexdigest()
    if expected_signature is not None and signature!=expected_signature:
        raise RuntimeError('source manifest replaced during the run; original signature required')
    freeze=json.loads(raw)
    environment={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__}
    if freeze['environment']!=environment:raise RuntimeError('runtime versions changed; refuse RNG/numeric replay')
    for row in freeze['files']:
        if sha(HERE/row['path'])!=row['sha256']:raise RuntimeError('source/input hash changed: '+row['path'])
    if manifest.read_bytes()!=raw:raise RuntimeError('source manifest changed while checking files')
    return signature


def run_directory(mode,signature):
    # A repaired implementation gets a new directory; earlier evidence survives.
    return HERE/'runs'/(mode+'-'+signature)


def resource_state(path=HERE):
    mem={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        k,v=line.split(':',1);mem[k]=int(v.strip().split()[0])*1024
    disk=os.statvfs(path);available=disk.f_bavail*disk.f_frsize
    return {'utc':now(),'available_memory_bytes':mem['MemAvailable'],'free_disk_bytes':available,
            'rss_highwater_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'resource_ok':mem['MemAvailable']>=2*1024**3 and available>=10*1024**3}


class Health:
    def __init__(self,run):
        self.run=run;self.progress={'state':'preparing'};self.stop=threading.Event();self.done=threading.Event()
        self.thread=threading.Thread(target=self.watch,name='owned-bootstrap-health',daemon=True)
    def watch(self):
        while not self.done.is_set():
            try:
                state=resource_state(self.run)|dict(self.progress)
                atomic_json(self.run/'health.json',state)
                with (self.run/'health.jsonl').open('a') as f:f.write(json.dumps(state,allow_nan=False)+'\n')
                if not state['resource_ok']:self.stop.set()
            except Exception:
                self.stop.set()
            self.done.wait(30)
    def start(self):self.thread.start()
    def close(self):self.done.set();self.thread.join(timeout=5)


def validate_draw(cell,data_path,record_path,previous,rng_state):
    record=json.loads(record_path.read_text())
    if record['previous_record_sha256']!=previous:raise RuntimeError('draw hash chain broken')
    if sha(data_path)!=record['data_sha256']:raise RuntimeError('draw NPZ changed')
    if record['role_rng_before']!=rng_state:raise RuntimeError('RNG sequence broken')
    rng=cell.role_rng();rng.bit_generator.state=rng_state;weights=cell.draw_weights(rng)
    with np.load(data_path,allow_pickle=False) as z:
        for role in ROLES:
            if not np.array_equal(z['weights_'+role],weights[role]):raise RuntimeError('saved source copies do not replay')
        for k in range(cell.k):
            _,_,owner,index,repstate=cell.representative(k,weights['G1_cal'],record['draw_id'])
            if not np.array_equal(z[f'representative_source_{k}'],owner) or not np.array_equal(z[f'representative_index_{k}'],index):
                raise RuntimeError('representative copy RNG does not replay')
            if record['representative_rng'][k]!=repstate:raise RuntimeError('representative RNG state mismatch')
    if rng.bit_generator.state!=record['role_rng_after']:raise RuntimeError('RNG final state mismatch')
    return sha(record_path),rng.bit_generator.state


def restore_cell(run,cell,signature,target,resume):
    folder=run/cell.name
    folder.mkdir(exist_ok=True)
    checkpoint=folder/'checkpoint.json'
    rng=cell.role_rng();previous=None;next_draw=0
    prefix_states=[{'hash':None,'rng':json.loads(json.dumps(rng.bit_generator.state))}]
    draws=sorted(p for p in folder.glob('draw-*') if p.is_dir())
    for i,directory in enumerate(draws):
        if directory.name!=f'draw-{i:05d}':raise RuntimeError('non-contiguous committed draw directories')
        if not resume:raise RuntimeError('committed draws already exist; explicit --resume required')
        meta=json.loads((directory/'record.json').read_text())
        if meta['signature']!=signature or meta['draw_id']!=i or meta['cell']!=cell.name:raise RuntimeError('committed metadata mismatch')
        previous,state=validate_draw(cell,directory/'data.npz',directory/'record.json',previous,rng.bit_generator.state)
        rng.bit_generator.state=state;next_draw=i+1
        prefix_states.append({'hash':previous,'rng':json.loads(json.dumps(state))})
    if next_draw>target:raise RuntimeError('draw count exceeds frozen run length')
    if checkpoint.exists():
        old=json.loads(checkpoint.read_text())
        old_index=old.get('next_draw')
        if type(old_index) is not int or not 0<=old_index<=next_draw or old.get('signature')!=signature or old.get('target')!=target:
            raise RuntimeError('checkpoint incompatible with immutable committed draws')
        if next_draw-old_index>1:raise RuntimeError('more than one uncheckpointed atomic draw; investigate')
        prefix=prefix_states[old_index]
        if old.get('last_record_sha256')!=prefix['hash'] or old.get('role_rng_state')!=prefix['rng']:
            raise RuntimeError('checkpoint does not match its reconstructed committed prefix')
    elif next_draw>1:raise RuntimeError('missing checkpoint for nontrivial run')
    # A crash after atomic draw commit but before checkpoint replacement is
    # recoverable by replay of that one complete draw. Partial directories remain.
    partials=[str(x.name) for x in folder.glob('.partial-*')]
    state={'signature':signature,'next_draw':next_draw,'target':target,'last_record_sha256':previous,
           'role_rng_state':rng.bit_generator.state,'preserved_partial_attempts':partials,'utc':now()}
    atomic_json(checkpoint,state)
    return folder,rng,previous,next_draw


def commit_draw(folder,cell,b,arrays,details,signature,previous,before,after,seconds):
    directory=folder/f'draw-{b:05d}'
    if directory.exists():raise RuntimeError('refuse overwrite committed draw')
    tmp=folder/('.partial-'+f'{b:05d}-'+uuid.uuid4().hex);tmp.mkdir()
    file=tmp/'data.npz'
    with file.open('xb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    record={'schema':'S5-bootstrap-draw-v1','signature':signature,'cell':cell.name,'draw_id':b,
            'previous_record_sha256':previous,'data_sha256':sha(file),'role_rng_before':before,
            'role_rng_after':after,'compute_seconds':seconds,'utc':now(),**details}
    atomic_json(tmp/'record.json',record)
    os.rename(tmp,directory)
    dfd=os.open(folder,os.O_DIRECTORY)
    try:os.fsync(dfd)
    finally:os.close(dfd)
    return sha(directory/'record.json')


def launch_gate(args,signature):
    if args.mode=='pilot':return
    if not args.approval:raise RuntimeError('full5000 blocked: independent audit receipt required')
    receipt=json.loads(Path(args.approval).read_text())
    pilot=run_directory('pilot',signature)/'PILOT-COMPLETE.json'
    if (receipt.get('status')!='PASS_S5_BOOTSTRAP_FULL_LAUNCH' or receipt.get('source_freeze_sha256')!=signature
        or receipt.get('protocol_sha256')!=sha(HERE/'PROTOCOL.json') or not pilot.exists()
        or receipt.get('pilot_receipt_sha256')!=sha(pilot)):
        raise RuntimeError('full5000 audit receipt missing or inconsistent')


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['pilot','full'],required=True)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--approval')
    args=parser.parse_args(argv)
    with writer_lock(HERE/'writer.lock'):
        signature=check_freeze();launch_gate(args,signature)
        protocol=json.loads((HERE/'source_snapshot/s5/PROTOCOL.json').read_text())
        run=run_directory(args.mode,signature);run.mkdir(parents=True,exist_ok=True)
        complete=run/('PILOT-COMPLETE.json' if args.mode=='pilot' else 'FULL-COMPUTE-COMPLETE.json')
        if complete.exists() and not args.resume:raise RuntimeError('run already complete; use verifier instead of rerunning')
        target=2 if args.mode=='pilot' else 5000
        config={'mode':args.mode,'draws_per_cell':target,'source_freeze_sha256':signature,
                'protocol_sha256':sha(HERE/'PROTOCOL.json'),'cell_names':[x['name'] for x in protocol['cells']],
                'roles':ROLES,'routes':ROUTES,'metrics':METRICS,'support_columns':SUPPORT}
        config=json.loads(json.dumps(config))
        if (run/'run-config.json').exists():
            if json.loads((run/'run-config.json').read_text())!=config:raise RuntimeError('existing run configuration differs')
        else:atomic_json(run/'run-config.json',config)
        health=Health(run);health.start()
        for signum in (signal.SIGTERM,signal.SIGINT):signal.signal(signum,lambda *_:health.stop.set())
        started=time.monotonic();rows=[]
        try:
            for cell_info in protocol['cells']:
                if health.stop.is_set():break
                t=time.monotonic();cell=CellData(cell_info);load_seconds=time.monotonic()-t
                folder,rng,previous,next_draw=restore_cell(run,cell,signature,target,args.resume)
                timings=[];bytes_written=[]
                for b in range(next_draw,target):
                    if health.stop.is_set():break
                    if b%100==0:
                        check_freeze(signature)
                        if not resource_state(run)['resource_ok']:health.stop.set();break
                    health.progress={'state':'computing','cell':cell.name,'draw_id':b,'target':target,'last_progress_utc':now()}
                    before=rng.bit_generator.state;t=time.monotonic();weights=cell.draw_weights(rng)
                    arrays,details=cell.compute(weights,b)
                    arrays.update({'weights_'+role:weights[role] for role in ROLES})
                    compute=time.monotonic()-t
                    previous=commit_draw(folder,cell,b,arrays,details,signature,previous,before,rng.bit_generator.state,compute)
                    atomic_json(folder/'checkpoint.json',{'signature':signature,'next_draw':b+1,'target':target,
                        'last_record_sha256':previous,'role_rng_state':rng.bit_generator.state,'utc':now()})
                    timings.append(time.monotonic()-t);bytes_written.append(sum(p.stat().st_size for p in (folder/f'draw-{b:05d}').iterdir()))
                    health.progress={'state':'draw_committed','cell':cell.name,'draw_id':b,'target':target,'last_progress_utc':now()}
                count=len(list(folder.glob('draw-*')))
                rows.append({'cell':cell.name,'prepare_seconds':load_seconds,'committed_draws':count,
                             'new_draw_wall_seconds':timings,'new_draw_bytes':bytes_written})
                print(json.dumps(rows[-1]),flush=True)
            check_freeze(signature)
            finished=len(rows)==len(protocol['cells']) and all(r['committed_draws']==target for r in rows)
            output={'status':('PILOT_COMPLETE_NOT_SCIENTIFIC_INTERVALS' if args.mode=='pilot' else 'FULL_COMPUTE_COMPLETE_UNVERIFIED') if finished else 'CHECKPOINTED_RESOURCE_OR_SIGNAL_STOP',
                    'utc':now(),'source_freeze_sha256':signature,'protocol_sha256':sha(HERE/'PROTOCOL.json'),
                    'cells':rows,'wall_seconds':time.monotonic()-started,'resources':resource_state(run),
                    'scientific_intervals_published':False,'parent_launch_still_required':args.mode=='pilot'}
            atomic_json(complete if finished else run/'STOPPED.json',output)
            print(json.dumps({'status':output['status'],'wall_seconds':output['wall_seconds']}),flush=True)
            return 0 if finished else 2
        except Exception:
            atomic_json(run/'ERROR.json',{'utc':now(),'traceback':traceback.format_exc(),'progress':health.progress,
                                        'action':'preserve all committed/partial data; no process killed; inspect before explicit resume'})
            raise
        finally:health.close()

if __name__=='__main__':raise SystemExit(main())

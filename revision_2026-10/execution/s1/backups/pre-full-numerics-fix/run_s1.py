"""Resumable S1 paired bootstrap, immutable manifests and per-cell writer locks.

pilot: exactly 10 outer draws per cell, never used as a scientific interval.
run: primary B5000/R20 and predeclared fixed-role B5000 sensitivity.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import fcntl
import hashlib
import json
import shutil
import sys
import time
import traceback
import numpy as np
from core import prepare, one_inner, roles_from_codes, original_roles, METHODS, METRICS
from vendor.gwd import obb_gwd

HERE=Path(__file__).resolve().parent
MANIFEST=HERE/'EXECUTION-MANIFEST.json'
CELLS=json.loads(MANIFEST.read_text())['cells']
B,R=5000,20
CODE_FILES=[HERE/'run_s1.py',HERE/'core.py',HERE/'vendor/gwd.py',HERE/'vendor/splits.py',MANIFEST]
STARTUP_HASHES={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in CODE_FILES}


def utc():return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()
def dump(path,value):
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def check_sources():
    for p,s in STARTUP_HASHES.items():
        if sha(p)!=s:raise RuntimeError('Loaded source or protocol changed: '+p)
def resources():
    mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    free=shutil.disk_usage(HERE).free
    return {'available_memory_bytes':mem['MemAvailable'],'free_disk_bytes':free,'utc':utc()}
def check_resources():
    result=resources()
    if result['available_memory_bytes']<4*1024**3 or result['free_disk_bytes']<15*1024**3:
        raise RuntimeError('Resource guard: '+json.dumps(result))
    return result

def load_cell(cell):
    p=Path(cell['cache']);before=p.stat()
    if sha(p)!=cell['cache_sha256']:raise RuntimeError('Cache mismatch')
    with np.load(p,allow_pickle=False) as z:c={k:z[k] for k in z.files}
    scores=obb_gwd(c['tp_pred'],c['tp_gt'])
    n=len(c['scene_names']);nc=len(c['class_names'])
    data=prepare(scores,c['tp_scene'],c['tp_cls'],c['tp_pred'],c['tp_gt'],n,nc)
    fingerprint=(before.st_size,before.st_mtime_ns,before.st_ino)
    return c,data,np.unique(c['tp_scene']),fingerprint


def input_unchanged(cell,fingerprint,full=False):
    p=Path(cell['cache']);s=p.stat()
    if (s.st_size,s.st_mtime_ns,s.st_ino)!=fingerprint:raise RuntimeError('Input stat changed')
    if full and sha(p)!=cell['cache_sha256']:raise RuntimeError('Input hash changed')


def atomic_npz(path,**arrays):
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def one_draw(b,rng,data,universe,ns,kind,fixed_roles):
    if kind=='primary':
        counts=np.bincount(rng.choice(universe,len(universe),replace=True),minlength=ns)
        roles=np.stack([roles_from_codes(counts,10000*b+r) for r in range(R)])
    else:
        counts=np.zeros(ns,dtype=np.int64)
        for role in (1,2,3):
            ids=np.flatnonzero(fixed_roles==role)
            counts+=np.bincount(rng.choice(ids,len(ids),replace=True),minlength=ns)
        roles=fixed_roles[None,:].copy()
    qs=[];coverages=[];supports=[]
    for role in roles:
        q,cov,sup=one_inner(data,counts,role)
        qs.append(q);coverages.append(cov[:,:,0]);supports.append(sup)
    return counts,roles,np.stack(qs),np.stack(coverages),np.stack(supports)


def run_cell(cell,mode,kind,out):
    check_sources();cellout=out/kind/cell['name'];cellout.mkdir(parents=True,exist_ok=True)
    with (cellout/'writer.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Existing writer for '+str(cellout))
        lock.seek(0);lock.truncate();lock.write(json.dumps({'pid':os.getpid(),'utc':utc()}));lock.flush()
        check_resources();c,data,universe,fingerprint=load_cell(cell)
        ns=len(c['scene_names']);nc=len(c['class_names']);limit=10 if mode=='pilot' else B
        aid=1 if kind=='primary' else 101
        signature={'code_sha256':STARTUP_HASHES,'cache_sha256':cell['cache_sha256'],'cell_id':cell['cell_id'],
                   'kind':kind,'B':B,'R_inner':R if kind=='primary' else 1,'limit_for_mode':limit,'mode':mode,
                   'seed_sequence':[20260928,aid,cell['cell_id']],'numpy_version':np.__version__,
                   'class_names':c['class_names'].tolist(),'n_sources':ns,'n_tp_positive_sources':len(universe)}
        rng=np.random.default_rng(np.random.SeedSequence(signature['seed_sequence']))
        cp=cellout/'checkpoint.json';start=0;elapsed=0.;chunks=[]
        if cp.exists():
            saved=json.loads(cp.read_text())
            if saved['signature']!=signature:raise RuntimeError('Resume signature mismatch')
            start=saved['next_draw'];elapsed=saved['elapsed_seconds'];chunks=saved['chunks'];rng.bit_generator.state=saved['rng_state']
            for item in chunks:
                if sha(cellout/item['file'])!=item['sha256']:raise RuntimeError('Checkpoint chunk hash mismatch')
        else:
            # Preserve uncommitted tail files from an interrupted initial commit.
            for p in cellout.glob('chunk-*.npz'):
                p.rename(p.with_name(p.name+'.uncommitted-'+str(time.time_ns())))
            dump(cellout/'signature.json',signature)
        fixed_roles=original_roles(c['scene_names'],universe,0)
        began=time.monotonic();block=[];block_start=start
        try:
            # Original source point estimates; separate from resampled CI.
            point=cellout/'point-estimate.npz'
            if not point.exists():
                counts=np.zeros(ns,dtype=np.int64);counts[universe]=1
                npoint=R if kind=='primary' else 1
                pointroles=np.stack([original_roles(c['scene_names'],universe,r) for r in range(npoint)])
                pointrows=[one_inner(data,counts,role,geometry=True) for role in pointroles]
                atomic_npz(point,counts=counts,roles=pointroles,q=np.stack([a[0] for a in pointrows]),
                           metrics=np.stack([a[1] for a in pointrows]),support=np.stack([a[2] for a in pointrows]),
                           metric_names=np.asarray(METRICS),class_names=c['class_names'])
            point_sha=sha(point)
            for b in range(start,limit):
                block.append(one_draw(b,rng,data,universe,ns,kind,fixed_roles))
                if (b+1)%25==0 or b+1==limit:
                    check_sources();input_unchanged(cell,fingerprint);health=check_resources()
                    dest=cellout/f'chunk-{block_start:05d}-{b+1:05d}.npz'
                    # A crash between npz rename and checkpoint commit can leave a valid orphan.
                    if dest.exists():dest.rename(dest.with_name(dest.name+'.uncommitted-'+str(time.time_ns())))
                    arrays=[np.stack([v[j] for v in block]) for j in range(5)]
                    atomic_npz(dest,draw_ids=np.arange(block_start,b+1),counts=arrays[0].astype(np.uint16),
                               roles=arrays[1].astype(np.uint8),q=arrays[2],coverage=arrays[3],support=arrays[4].astype(np.int32))
                    chunks.append({'file':dest.name,'sha256':sha(dest),'start':block_start,'end':b+1,'bytes':dest.stat().st_size})
                    total=elapsed+time.monotonic()-began
                    state={'signature':signature,'next_draw':b+1,'rng_state':rng.bit_generator.state,'chunks':chunks,
                           'elapsed_seconds':total,'point_sha256':point_sha,'pid':os.getpid(),'utc':utc(),'resources':health}
                    dump(cp,state)
                    print(json.dumps({'cell':cell['name'],'kind':kind,'draw':b+1,'limit':limit,'seconds':round(total,2)}),flush=True)
                    block=[];block_start=b+1
            input_unchanged(cell,fingerprint,full=True);check_sources()
            result={'status':'PILOT_COMPLETE_NOT_SCIENTIFIC_RESULT' if mode=='pilot' else 'DRAWS_COMPLETE_AWAITING_VERIFICATION',
                    'utc':utc(),'cell':cell['name'],'kind':kind,'draws':limit,'elapsed_seconds':elapsed+time.monotonic()-began,
                    'signature':signature,'checkpoint_sha256':sha(cp),'point_sha256':point_sha,
                    'projected_full_seconds_from_pilot':(elapsed+time.monotonic()-began)*B/limit if mode=='pilot' else None}
            dump(cellout/'completion.json',result)
            return result
        except BaseException as exc:
            dump(cellout/'failure.json',{'utc':utc(),'pid':os.getpid(),'error':repr(exc),'traceback':traceback.format_exc(),
                 'committed_next_draw':json.loads(cp.read_text())['next_draw'] if cp.exists() else 0,
                 'signature':signature,'resources':resources()})
            raise


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['pilot','run']);p.add_argument('--kind',choices=['primary','fixed-role'],default='primary')
    p.add_argument('--workers',type=int,default=2);p.add_argument('--cells',nargs='+',choices=[c['name'] for c in CELLS]);p.add_argument('--out',type=Path)
    args=p.parse_args()
    if not 1<=args.workers<=3:raise ValueError('one to three local workers only')
    out=args.out or HERE/('pilot' if args.mode=='pilot' else 'results')
    out.mkdir(exist_ok=True,parents=True)
    selected=[c for c in CELLS if not args.cells or c['name'] in args.cells]
    outputs=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks=[pool.submit(run_cell,c,args.mode,args.kind,out) for c in selected]
        for f in as_completed(tasks):outputs.append(f.result())
    dump(out/f'{args.kind}-batch-complete.json',{'utc':utc(),'mode':args.mode,'results':outputs})

if __name__=='__main__':main()

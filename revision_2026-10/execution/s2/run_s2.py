#!/usr/bin/env python
"""Single-worker S2 pilot/full. Never auto-starts a second process."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from s2_core import conditions, replicate, summarize, METHODS, METRICS, POIS_TAIL
from s2_io import (ROOT, write_json, read_json, digest, utc, lock, freeze, verify_frozen,
                   save_chunk, load_chunks, resources, set_resource_limit)


def budget_gate():
    report=read_json(ROOT/'pilot'/'PILOT.json')
    if report['frozen_hash']!=verify_frozen():raise RuntimeError('pilot uses different source')
    max_cpu=1500.;max_memory=2*1024**3
    parent=ROOT/'parent-budget.json'
    if parent.exists():
        extra=read_json(parent)
        if extra.get('hold',False):raise RuntimeError('parent scheduler hold')
        max_cpu=min(max_cpu, float(extra.get('max_cpu_seconds',max_cpu)))
        max_memory=min(max_memory,int(extra.get('max_memory_bytes',max_memory)))
    if report['projected_full_cpu_seconds_with_safety_factor']>=max_cpu:
        raise RuntimeError('pilot CPU cost requires parent scheduler')
    if report['projected_peak_memory_bytes']>max_memory:
        raise RuntimeError('pilot memory cost requires parent scheduler')
    return dict(cpu_limit=max_cpu,memory_limit=max_memory,checked=utc(),parent_budget_present=parent.exists())


def run(mode):
    target=10 if mode=='pilot' else 2000
    with lock():
        freeze();frozen_hash=verify_frozen()
        gate=budget_gate() if mode=='full' else None
        directory=ROOT/mode
        if (directory/'COMPLETE.json').exists():
            meta=read_json(directory/'COMPLETE.json')
            if meta['frozen_hash']!=frozen_hash:raise RuntimeError('completed phase foreign hash')
            print(json.dumps(dict(status='ALREADY_COMPLETE',mode=mode,receipt=str(directory/'COMPLETE.json'))));return
        set_resource_limit();directory.mkdir(exist_ok=True)
        started_wall=time.monotonic();started_cpu=time.process_time()
        write_json(ROOT/'active-run.json',dict(mode=mode,frozen_hash=frozen_hash,gate=gate,started=utc(),**resources()))
        all_summary=[];timings=[]
        for condition in conditions():
            cid=condition['condition_id'];cd=directory/f'{cid:02d}'
            cd.mkdir(exist_ok=True)
            previous,previous_aux,done=load_chunks(cd,frozen_hash,target)
            cpu=time.process_time();wall=time.monotonic();start_done=done
            if done<target:
                buffer=[];buffer_aux=[];first=done
                for rep in range(done,target):
                    v,a=replicate(condition,rep)
                    buffer.append(v);buffer_aux.append(a)
                    if len(buffer)==25 or rep==target-1:
                        values=np.stack(buffer);aux=np.stack(buffer_aux)
                        save_chunk(cd,first,values,aux,frozen_hash)
                        previous.append(values);previous_aux.append(aux);done=rep+1
                        health=resources()
                        if health['free_disk_bytes']<1024**3:
                            raise RuntimeError('less than1GiB free disk; checkpoint saved, stopping safely')
                        write_json(directory/'state.json',dict(status='RUNNING',mode=mode,
                            condition_id=cid,completed_in_condition=done,completed_total=cid*target+done,
                            target_total=36*target,frozen_hash=frozen_hash,**health))
                        buffer=[];buffer_aux=[];first=done
                verify_frozen()
            values=np.concatenate(previous);aux=np.concatenate(previous_aux)
            summary=summarize(condition,values,aux)
            write_json(cd/'summary.json',summary)
            tm=dict(condition_id=cid,replications_computed_this_invocation=done-start_done,
                    cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall)
            prior_time=cd/'timing.json'
            if prior_time.exists():
                old=read_json(prior_time)
                tm['cpu_seconds']+=old['cpu_seconds'];tm['wall_seconds']+=old['wall_seconds']
                tm['replications_computed_this_invocation']+=old['replications_computed_this_invocation']
            write_json(prior_time,tm)
            timings.append(tm);all_summary.append(summary)
            print(json.dumps(dict(mode=mode,condition_id=cid,completed=done,cpu_seconds=tm['cpu_seconds'],wall_seconds=tm['wall_seconds'])),flush=True)
        verify_frozen()
        totalcpu=sum(x['cpu_seconds'] for x in timings)
        totalwall=sum(x['wall_seconds'] for x in timings)
        health=resources()
        common=dict(mode=mode,status='COMPLETE',frozen_hash=frozen_hash,conditions=36,
                    replications_per_condition=target,total_replications=36*target,
                    evaluation_scenes_per_replication=2000,methods=METHODS,metrics=METRICS,
                    timings=timings,total_condition_cpu_seconds=totalcpu,total_condition_wall_seconds=totalwall,
                    current_invocation_cpu_seconds=time.process_time()-started_cpu,
                    current_invocation_wall_seconds=time.monotonic()-started_wall,
                    poisson65plus_tail_bound=POIS_TAIL,**health)
        write_json(directory/'summary.json',dict(**common,results=all_summary))
        if mode=='pilot':
            # Full retains only one condition in memory; raw arrays <2MiB,
            # allow256MiB extra plus2x observed high-water mark for allocator drift.
            peak=2*health['rss_peak_bytes']+256*1024**2
            report=dict(**common,full_to_pilot_replication_ratio=200,
                        timing_safety_factor=1.5,
                        projected_full_cpu_seconds=totalcpu*200,
                        projected_full_wall_seconds=totalwall*200,
                        projected_full_cpu_seconds_with_safety_factor=totalcpu*200*1.5,
                        projected_full_wall_seconds_with_safety_factor=totalwall*200*1.5,
                        projected_peak_memory_bytes=peak,
                        auto_full_eligible=(totalcpu*200*1.5<1500 and peak<=2*1024**3),
                        scientific_use='timing/validation only; no scientific conclusions from10 reps')
            write_json(directory/'PILOT.json',report)
        write_json(directory/'COMPLETE.json',dict(**common,summary_sha256=digest(directory/'summary.json')))
        write_json(directory/'state.json',dict(status='COMPLETE',completed_total=36*target,target_total=36*target,**health))
        write_json(ROOT/'active-run.json',dict(status='COMPLETE',mode=mode,utc=utc(),pid=os.getpid()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['freeze','pilot','full'])
    args=parser.parse_args()
    if args.mode=='freeze':
        with lock():freeze()
        print(verify_frozen());return
    try:run(args.mode)
    except Exception as exc:
        # Error evidence lives in owner directory. A failed lock acquisition
        # must not overwrite an active runner's state.
        path=ROOT/f'ERROR-{os.getpid()}-{time.time_ns()}.json'
        write_json(path,dict(mode=args.mode,error=repr(exc),traceback=traceback.format_exc(),utc=utc()))
        raise

if __name__=='__main__':main()

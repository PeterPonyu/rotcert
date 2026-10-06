#!/usr/bin/env python
"""Independent checkpoint-to-summary arithmetic verification plus data replay.

Does not import the production summarizer. Reference mean/variance uses fsum.
Full verification replays first and last replication in each condition, and
compares pilot rep0..9 to full exactly. Scientific Fraction QA in test_s2.py.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import math
import numpy as np
from s2_io import ROOT, read_json, write_json, verify_frozen, load_chunks, digest, utc, lock
from s2_core import replicate


def independent_summary(values):
    vals=sorted(float(x) for x in values if math.isfinite(float(x)))
    n=len(vals)
    if not n:return {'mean':None,'sd':None,'mcse':None,'q025':None,'median':None,'q975':None,'minimum':None,'maximum':None,'n':0,'missing':len(values)}
    mean=math.fsum(vals)/n
    sd=math.sqrt(math.fsum((v-mean)**2 for v in vals)/(n-1)) if n>1 else None
    def quantile(p):
        at=(n-1)*p;lo=math.floor(at);hi=math.ceil(at)
        return vals[lo]+(at-lo)*(vals[hi]-vals[lo])
    return dict(mean=mean,sd=sd,mcse=sd/math.sqrt(n) if sd is not None else None,
                q025=quantile(.025),median=quantile(.5),q975=quantile(.975),
                minimum=vals[0],maximum=vals[-1],n=n,missing=len(values)-n)


def verify(mode):
    frozen_hash=verify_frozen();mapping=read_json(ROOT/'CONDITIONS.json')
    phase=ROOT/mode;target=10 if mode=='pilot' else 2000
    complete=read_json(phase/'COMPLETE.json')
    if complete['summary_sha256']!=digest(phase/'summary.json'):
        raise RuntimeError('aggregate summary hash mismatch')
    if complete['frozen_hash']!=frozen_hash:raise RuntimeError('foreign completed run')
    aggregate=read_json(phase/'summary.json');checks=0;max_error=0.;replay=0;manifests=[]
    for condition in mapping['conditions']:
        cid=condition['condition_id'];directory=phase/f'{cid:02d}'
        parts,aux_parts,n=load_chunks(directory,frozen_hash,target)
        if n!=target:raise AssertionError('incomplete condition')
        values=np.concatenate(parts);aux=np.concatenate(aux_parts)
        if np.any(np.isinf(values)):raise AssertionError('infinite metrics must use explicit fractions not accidental infinity')
        expected=read_json(directory/'summary.json')
        if expected!=aggregate['results'][cid]:raise AssertionError('condition aggregate disagreement')
        for a,method in enumerate(mapping['methods']):
            for j,metric in enumerate(mapping['metrics']):
                ref=independent_summary(values[:,a,j]);record=expected['methods'][method][metric]
                for key,value in ref.items():
                    actual=record[key]
                    if value is None:
                        if actual is not None:raise AssertionError('missingness differs')
                    else:
                        err=abs(actual-value);max_error=max(max_error,err)
                        if err>3e-12*max(1,abs(value)):raise AssertionError((cid,method,metric,key,actual,value))
                    checks+=1
        for rep in [0,target-1]:
            v,a=replicate(condition,rep)
            if not (np.array_equal(v,values[rep],equal_nan=True) and np.array_equal(a,aux[rep],equal_nan=True)):
                raise AssertionError('deterministic raw replay failed')
            replay+=1
        if condition['regime']=='constant8':
            ix=mapping['metrics'].index('scene_risk');iy=mapping['metrics'].index('test_object_ratio')
            if not np.array_equal(values[:,:,ix],values[:,:,iy]):raise AssertionError('constant-m exact identity')
        if mode=='full':
            p,pa,pn=load_chunks(ROOT/'pilot'/f'{cid:02d}',frozen_hash,10)
            if pn!=10 or not np.array_equal(values[:10],np.concatenate(p),equal_nan=True):
                raise AssertionError('pilot/full raw values mismatch')
            if not np.array_equal(aux[:10],np.concatenate(pa),equal_nan=True):
                raise AssertionError('pilot/full auxiliary mismatch')
        manifests.append(dict(condition_id=cid,summary_sha256=digest(directory/'summary.json'),replications=n))
    result=dict(status='PASS',mode=mode,utc=utc(),frozen_hash=frozen_hash,
                conditions=36,replications_per_condition=target,independent_scalar_checks=checks,
                maximum_summary_rounding_difference=max_error,deterministic_replay_replications=replay,
                pilot_full_match=True if mode=='full' else None,
                limitations='Finite-data verification and Monte Carlo evidence, not proof of validity or PAC; simulated model only',
                condition_receipts=manifests)
    write_json(phase/'VERIFIED.json',result)
    print({key:result[key] for key in ['status','mode','independent_scalar_checks','maximum_summary_rounding_difference','deterministic_replay_replications']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['pilot','full']);args=p.parse_args()
    with lock():verify(args.mode)

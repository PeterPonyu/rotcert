"""Independent saved-draw audit and fixed-family intervals; does not import runner/core.

Never reports a complete-dictionary CI if a required draw is missing. Partial
runs fail before interval construction. Verifies every count/role RNG replay,
all artifact/code hashes, and exact rational recalibration on fixed audit draws.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
from fractions import Fraction
from datetime import datetime,timezone
import argparse
import hashlib
import json
import math
import time
import numpy as np
from scipy.stats import binom
from vendor.gwd import obb_gwd
HERE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def dump(path,obj):
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(obj,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def utc():return datetime.now(timezone.utc).isoformat()
def expected_roles(counts,seed):
    ids=np.flatnonzero(counts);r=np.zeros(len(counts),np.uint8)
    if len(ids)<3:return r
    shuffled=ids[np.random.default_rng(seed).permutation(len(ids))]
    nc=max(1,round(.4*len(ids)));nd=max(1,round(.2*len(ids)))
    if nc+nd>=len(ids):nc,nd=len(ids)-2,1
    r[shuffled[:nc]]=1;r[shuffled[nc:nc+nd]]=2;r[shuffled[nc+nd:]]=3
    return r

def quantile_mc(a,p):
    n=len(a);x=np.sort(a)
    lower=int(binom.ppf(.025,n,p));upper=int(binom.ppf(.975,n,p))+1
    return {'p':p,'order_statistic_ranks_1based':[lower,upper],
            'bounds':[float(x[lower-1]) if lower>=1 else None,float(x[upper-1]) if upper<=n else None],
            'interpretation':'95% binomial/order-statistic Monte Carlo bracket for this bootstrap quantile; not a scientific confidence interval'}

def interval(a,level,blocks=True):
    a=np.asarray(a,dtype=float);good=np.isfinite(a);missing=int((~good).sum())
    result={'B':len(a),'valid_draws':int(good.sum()),'missing_draws':missing,'level':level,
            'status':'UNAVAILABLE_MISSING_REQUIRED_DRAWS' if missing else 'AVAILABLE_APPROXIMATE_PERCENTILE'}
    if missing:
        result['interval']=None;return result
    p=(1-level)/2
    result.update(mean=float(a.mean()),interval=np.quantile(a,[p,1-p],method='linear').tolist(),
                  quantile_mc=[quantile_mc(a,p),quantile_mc(a,1-p)])
    if blocks:
        result['five_block_intervals']=[np.quantile(a[i*1000:(i+1)*1000],[p,1-p],method='linear').tolist() for i in range(5)]
    return result


def rational_hcp(s,sc,m,cw):
    k=int(cw[m>0].sum())
    if k<9:return math.inf
    active=cw[sc]>0;s=s[active];sc=sc[active]
    if not len(s):return math.inf
    candidates=np.unique(s);target=Fraction(9*(k+1),10)
    def mass(q):
        cv=np.bincount(sc[s<=q],minlength=len(m))*cw
        # Aggregate exact integer numerators by each distinct scene denominator.
        ids=np.flatnonzero(cv)
        totals={}
        for i in ids:totals[int(m[i])]=totals.get(int(m[i]),0)+int(cv[i])
        return sum((Fraction(v,d) for d,v in totals.items()),Fraction())
    lo,hi=0,len(candidates)-1
    if mass(candidates[hi])<target:return math.inf
    while lo<hi:
        mid=(lo+hi)//2
        if mass(candidates[mid])>=target:hi=mid
        else:lo=mid+1
    return float(candidates[lo])


def recompute_inner(cache,scores,counts,roles):
    ns=len(counts);nk=len(cache['class_names']);q=np.full((2,nk),np.inf);cov=np.full((2,nk),np.nan)
    support=np.zeros((3,nk,3),dtype=np.int64)
    for k in range(nk):
        use=cache['tp_cls']==k;s=scores[use];sc=cache['tp_scene'][use];m=np.bincount(sc,minlength=ns)
        cw=counts*(roles==1);ew=counts*(roles==3)
        for role in (1,2,3):
            w=counts*(roles==role);ok=(m>0)&(w>0)
            support[role-1,k]=[ok.sum(),w[ok].sum(),(w*m).sum()]
        expanded=np.repeat(s,cw[sc]);rank=math.ceil(Fraction(9,10)*(len(expanded)+1))
        if rank<=len(expanded):q[0,k]=float(np.sort(expanded)[rank-1])
        q[1,k]=rational_hcp(s,sc,m,cw)
        ev=(m>0)&(ew>0)
        if ev.any():
            for method in range(2):
                covered=np.bincount(sc,weights=s<=q[method,k],minlength=ns)
                cov[method,k]=np.average(covered[ev]/m[ev],weights=ew[ev])
    return q,cov,support


def verify_cell(root,cell,kind,pilot=False):
    folder=root/kind/cell['name'];cp=json.loads((folder/'checkpoint.json').read_text());sig=cp['signature']
    expectedB=10 if pilot else 5000
    assert cp['next_draw']==expectedB,'Partial draws cannot form scientific intervals'
    assert sig['B']==5000 and sig['mode']==('pilot' if pilot else 'run')
    assert sig['R_inner']==(20 if kind=='primary' else 1)
    for p,h in sig['code_sha256'].items():assert sha(p)==h,('source changed',p)
    assert sha(cell['cache'])==cell['cache_sha256']==sig['cache_sha256']
    assert sha(folder/'point-estimate.npz')==cp['point_sha256']
    with np.load(cell['cache'],allow_pickle=False) as z:cache={k:z[k] for k in z.files}
    universe=np.unique(cache['tp_scene']);ns=len(cache['scene_names']);nk=len(cache['class_names'])
    scores=obb_gwd(cache['tp_pred'],cache['tp_gt'])
    rng=np.random.default_rng(np.random.SeedSequence([20260928,1 if kind=='primary' else 101,cell['cell_id']]))
    original_ids=np.asarray(sorted(universe,key=lambda i:str(cache['scene_names'][i])))
    fixed=np.zeros(ns,np.uint8);fixed[original_ids]=expected_roles(np.ones(len(original_ids),int),0)
    cover=[];qrows=[];supports=[];unique=[];audit=[];position=0
    auditdraws={0,expectedB-1};auditr={0,19} if kind=='primary' else {0}
    for chunk in cp['chunks']:
        path=folder/chunk['file'];assert sha(path)==chunk['sha256'];assert chunk['start']==position
        with np.load(path,allow_pickle=False) as z:
            assert z['draw_ids'].tolist()==list(range(chunk['start'],chunk['end']))
            for j,b in enumerate(z['draw_ids']):
                b=int(b);counts=z['counts'][j].astype(np.int64)
                if kind=='primary':
                    ec=np.bincount(rng.choice(universe,len(universe),replace=True),minlength=ns)
                else:
                    ec=np.zeros(ns,dtype=np.int64)
                    for role in (1,2,3):
                        ids=np.flatnonzero(fixed==role)
                        ec+=np.bincount(rng.choice(ids,len(ids),replace=True),minlength=ns)
                assert np.array_equal(counts,ec),'outer RNG/count mismatch'
                assert int(counts.sum())==len(universe)
                for r,roles in enumerate(z['roles'][j]):
                    er=expected_roles(counts,10000*b+r) if kind=='primary' else fixed
                    assert np.array_equal(roles,er),'role/RNG mismatch'
                    if b in auditdraws and r in auditr:
                        rq,rc,rs=recompute_inner(cache,scores,counts,roles)
                        assert np.array_equal(rq,z['q'][j,r]),('exact rational q mismatch',cell['name'],b,r)
                        assert np.allclose(rc,z['coverage'][j,r],rtol=0,atol=3e-14,equal_nan=True)
                        assert np.array_equal(rs,z['support'][j,r])
                        audit.append({'draw':b,'inner':r,'q':'EXACT_RATIONAL_PASS','coverage':'INDEPENDENT_SCENE_MEAN_PASS'})
                unique.append(int(np.count_nonzero(counts)))
            cover.append(z['coverage'].copy());qrows.append(z['q'].copy());supports.append(z['support'].copy())
        position=chunk['end']
    assert position==expectedB
    assert rng.bit_generator.state==cp['rng_state'],'final RNG mismatch'
    c=np.concatenate(cover);q=np.concatenate(qrows);s=np.concatenate(supports)
    assert c.shape==(expectedB,sig['R_inner'],2,nk)
    cm=np.mean(c,axis=1);delta=cm[:,1,:]-cm[:,0,:];macro=np.mean(delta,axis=1)
    with np.load(folder/'point-estimate.npz',allow_pickle=False) as z:
        pm=np.mean(z['metrics'][:,:,:,0],axis=0)
        pd=pm[1]-pm[0]
        point={'per_class_delta':[float(v) if np.isfinite(v) else None for v in pd],
               'class_macro_delta':float(pd.mean()) if np.isfinite(pd).all() else None,
               'per_class_coverage':[[float(v) if np.isfinite(v) else None for v in row] for row in pm]}
    row={'cell':cell['name'],'kind':kind,'status':'PILOT_INTEGRITY_VERIFIED_NO_CI' if pilot else 'VERIFIED',
         'B':expectedB,'R_inner':sig['R_inner'],'point':point,'independent_recalibration':audit,
         'all_draw_roles_counts_rng_verified':True,'unique_support':{'original_TP_sources':len(universe),'draw_min':min(unique),'draw_mean':float(np.mean(unique)),'draw_max':max(unique)},
         'source_independence':'assumed, not established by IDs or bootstrap','elapsed_seconds':cp['elapsed_seconds'],
         'interval_kind':'Approximate source-bootstrap; per-class95% marginal; six-cell macro99.1667% family adjustment; not finite-sample simultaneous theorem',
         'per_class':[],'checkpoint_sha256':sha(folder/'checkpoint.json')}
    if not pilot:row['macro_delta_interval']=interval(macro,1-.05/6)
    for k,name in enumerate(cache['class_names']):
        cr={'class':str(name),'missing_inner_evaluations':int(np.isnan(c[:,:,:,k]).any(axis=2).sum()),
            'cal_unique_min':int(s[:,:,0,k,0].min()),'eval_unique_min':int(s[:,:,2,k,0].min()),
            'cal_weighted_mean':float(s[:,:,0,k,1].mean()),'eval_weighted_mean':float(s[:,:,2,k,1].mean()),
            'infinite_threshold_fraction':[float(np.isinf(q[:,:,m,k]).mean()) for m in range(2)],
            'q_finite_median':[float(np.median(q[:,:,m,k][np.isfinite(q[:,:,m,k])])) if np.isfinite(q[:,:,m,k]).any() else None for m in range(2)]}
        if not pilot:
            cr['delta_interval95_marginal']=interval(delta[:,k],.95)
            cr['coverage_intervals95_marginal']=[interval(cm[:,m,k],.95) for m in range(2)]
        row['per_class'].append(cr)
    dump(folder/'verified.json',row);return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=HERE/'results');p.add_argument('--kind',choices=['primary','fixed-role'],default='primary');p.add_argument('--pilot',action='store_true');args=p.parse_args()
    start=time.monotonic();m=json.loads((HERE/'EXECUTION-MANIFEST.json').read_text())
    rows=[]
    for cell in m['cells']:
        rows.append(verify_cell(args.root,cell,args.kind,args.pilot));print('VERIFIED',cell['name'],args.kind,flush=True)
    output={'status':'PILOT_VERIFIED_NO_SCIENCE_CI' if args.pilot else 'S1_COMPLETE_INDEPENDENTLY_VERIFIED',
            'utc':utc(),'kind':args.kind,'cells':rows,'verifier_sha256':sha(__file__),'scipy_bounded_tail_method':'binom order statistics',
            'elapsed_seconds':time.monotonic()-start,'protocol_sha256':sha(HERE/'EXECUTION-MANIFEST.json'),
            'known_limitations':['Source identities do not demonstrate iid/geographic independence','Outer resampling does not increase independent source support','Duplicate-safe split reduces unique support; fixed-role sensitivity must accompany primary','Tail Monte Carlo uncertainty is distinct from population sampling uncertainty','Scientific bootstrap intervals are approximate and class intervals marginal']}
    dump(args.root/f'{args.kind}-VERIFIED.json',output)

if __name__=='__main__':main()

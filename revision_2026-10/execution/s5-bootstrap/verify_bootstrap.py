#!/usr/bin/env python3
"""Read-only draw verification and complete-B marginal descriptive intervals.

Pilot validation expands copied sources and uses frozen reference functions for
calibration; it never issues percentile science results from two draws. The full
path independently replays hashes/RNG and recomputes every interval from all5000
saved outputs, with fixed five-block and binomial quantile-MC diagnostics.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from fractions import Fraction
from pathlib import Path
import argparse,json,math,sys,time
import numpy as np
from scipy.stats import binom
from bootstrap_core import CellData,ROLES,ROUTES,METRICS,SUPPORT,Q,sha
from rotcert_contracts.exact import hcp_threshold
from rotcert_contracts.bounds import empirical_bernstein,tolerance_rank
from run_bootstrap import check_freeze,atomic_json,writer_lock,run_directory

HERE=Path(__file__).resolve().parent


def extended_linear(values,p):
    """NumPy method=linear for finite values, extended-real +inf by continuity.

    NA never removed. At a nonzero interpolation weight with right endpoint+inf,
    output+inf; equal +inf endpoints remain+inf rather than inf-inf=NaN.
    """
    a=np.asarray(values,float)
    if a.ndim!=1 or not len(a) or np.any(np.isnan(a)):return None
    if np.any(np.isneginf(a)):raise ValueError('negative infinity not an S5 endpoint')
    a=np.sort(a);index=(len(a)-1)*float(p);lo=int(math.floor(index));hi=int(math.ceil(index))
    if lo==hi or a[lo]==a[hi]:return float(a[lo])
    if math.isinf(a[hi]):return math.inf
    return float(a[lo]+(index-lo)*(a[hi]-a[lo]))


def tail_mc(values,p):
    a=np.sort(values);n=len(a)
    lower_count=int(binom.ppf(.025,n,p));upper_count=int(binom.ppf(.975,n,p))
    # For a population quantile Q_p, the count <=Q_p is binomial under continuity.
    # [X_(l),X_(u)] with l=qbinom(.025),u=qbinom(.975)+1. Endpoints outside the
    # sample are explicitly unavailable instead of silently clipping certainty.
    lower_rank=lower_count;upper_rank=upper_count+1
    return {'nominal_MC_coverage':.95,'tail_probability':p,'lower_order_1based':lower_rank,
            'upper_order_1based':upper_rank,
            'lower':None if lower_rank<1 else float(a[lower_rank-1]),
            'upper':None if upper_rank>n else float(a[upper_rank-1]),
            'assumption':'iid numerical bootstrap draws; order-statistic MC diagnostic, not statistical population CI'}


def interval(values,required=5000):
    a=np.asarray(values,float);missing=int(np.isnan(a).sum());infinite=int(np.isposinf(a).sum())
    if len(a)!=required:return {'status':'UNAVAILABLE_INCOMPLETE_B','observed_B':len(a),'required_B':required}
    base={'B':len(a),'missing_draws':missing,'infinite_draws':infinite,'interval_kind':'95% marginal descriptive approximate percentile bootstrap'}
    if missing:return {**base,'status':'UNAVAILABLE_REQUIRED_DRAW_OR_DEPLOYMENT_NA','interval':None}
    ci=[extended_linear(a,.025),extended_linear(a,.975)]
    return {**base,'status':'AVAILABLE_EXTENDED_REAL','interval':ci,
            'five_fixed_blocks':[[extended_linear(block,.025),extended_linear(block,.975)] for block in np.split(a,5)],
            'quantile_MC':[tail_mc(a,.025),tail_mc(a,.975)]}


def sanitize(obj):
    if isinstance(obj,np.generic):obj=obj.item()
    if isinstance(obj,float) and not math.isfinite(obj):return 'infinity' if obj>0 else 'NA' if math.isnan(obj) else '-infinity'
    if isinstance(obj,dict):return {k:sanitize(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [sanitize(v) for v in obj]
    return obj


def direct_sequence(loss_arrays,eta,limit):
    chosen=None
    for j in range(len(Q)-1,-1,-1):
        bound=empirical_bernstein(loss_arrays[j],eta)
        if not bound.passes(limit):break
        chosen=float(Q[j])
    return chosen


def compare_number(actual,reference,context,tolerance=1e-12):
    """Fail closed on availability or infinity disagreement in either direction."""
    actual=float(actual);reference=float(reference)
    if math.isnan(reference):
        if not math.isnan(actual):raise RuntimeError(context+': expected NA')
        return 0.
    if math.isinf(reference):
        if actual!=reference:raise RuntimeError(context+': infinity mismatch')
        return 0.
    if not math.isfinite(actual):raise RuntimeError(context+': finite reference has nonfinite output')
    error=abs(actual-reference)
    if error>tolerance:raise RuntimeError(context+': numerical mismatch')
    return error


def verify_representatives(cell,weights,record,saved,b):
    """Replay prescribed independent copy randomness for every committed draw.

    Derive seeds from the protocol and loop index, never from recorded seeds.
    This does no calibration and is intentionally separate from CellData's
    optimized representative function and the bounded scientific oracle.
    """
    states=record['representative_rng']
    if len(states)!=cell.k:raise RuntimeError('representative RNG class count mismatch')
    for k in range(cell.k):
        one=cell.roles['G1_cal'][k]
        owners=np.repeat(np.arange(len(one.m),dtype=np.int32),weights['G1_cal'])
        owners=owners[one.m[owners]>0]
        seed=[20260928,5,cell.cell_id,1,b,k]
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
        initial=rng.bit_generator.state
        indices=rng.integers(0,one.m[owners],dtype=np.int64)
        expected={'seed':seed,'initial':initial,'final':rng.bit_generator.state}
        if states[k]!=expected:raise RuntimeError('noncanonical representative seed or RNG state')
        owner=saved[f'representative_source_{k}'];index=saved[f'representative_index_{k}']
        if owner.dtype.kind not in 'iu' or index.dtype.kind not in 'iu':raise RuntimeError('representative indices must be integers')
        if not np.array_equal(owner,owners) or not np.array_equal(index,indices):
            raise RuntimeError('representative copy owner/index does not replay')


def verify_metadata(record,cell,b,signature,previous):
    if (record.get('schema')!='S5-bootstrap-draw-v1' or record.get('signature')!=signature
        or record.get('cell')!=cell.name or type(record.get('draw_id')) is not int
        or record['draw_id']!=b or record.get('previous_record_sha256')!=previous):
        raise RuntimeError('draw metadata or chain mismatch')


def verify_checkpoint(checkpoint,target,signature,previous,state):
    if (type(checkpoint.get('next_draw')) is not int or checkpoint['next_draw']!=target
        or type(checkpoint.get('target')) is not int or checkpoint['target']!=target
        or checkpoint.get('signature')!=signature or checkpoint.get('last_record_sha256')!=previous
        or checkpoint.get('role_rng_state')!=state):
        raise RuntimeError('checkpoint final signature/target/prefix mismatch')


def verify_saved_invariants(cell,weights,saved):
    """Validate denominators, availability, and cached summaries on every draw."""
    m=saved['metrics'];s=saved['status'];u=saved['support'];gp=saved['g2_pass']
    if m.shape!=(cell.k,len(ROUTES),len(METRICS)) or s.shape!=(cell.k,len(ROUTES)):
        raise RuntimeError('metric/status shape mismatch')
    if u.shape!=(cell.k,len(ROLES),len(SUPPORT)) or gp.shape!=(cell.k,2):raise RuntimeError('support/G2 shape mismatch')
    if s.dtype.kind not in 'iu' or np.any(~np.isin(s,[0,1,2])):raise RuntimeError('invalid deployment status')
    if u.dtype.kind not in 'iu' or np.any(u<0):raise RuntimeError('invalid support counts')
    if gp.dtype.kind not in 'iu' or np.any(~np.isin(gp,[0,1])):raise RuntimeError('invalid G2 flags')
    if np.any(np.isneginf(m)):raise RuntimeError('negative infinity not permitted')
    for k in range(cell.k):
        for j,role in enumerate(ROLES):
            if not np.array_equal(u[k,j],cell.roles[role][k].support(weights[role])):
                raise RuntimeError('source-copy or instance denominator mismatch')
        ev=cell.roles['eval'][k];pos=ev.H>0;den=int(weights['eval'][pos].sum())
        reference=float(weights['eval'][pos]@(ev.fn[pos]/ev.H[pos])/den) if den else math.nan
        compare_number(saved['raw_FNR'][k],reference,'raw recall denominator')
        for r in range(len(ROUTES)):
            if s[k,r]==0:
                if not np.all(np.isnan(m[k,r])):raise RuntimeError('no certificate has non-NA deployment metrics')
            elif s[k,r]==1:
                if not math.isfinite(m[k,r,0]) or m[k,r,0]<0:raise RuntimeError('finite deployment q invalid')
            elif m[k,r,0]!=math.inf:raise RuntimeError('all-space deployment must retain infinite q')
            if s[k,r]>0:
                # Support determines NA independently of the serialized metric.
                tp_supported=bool(u[k,2,SUPPORT.index('TP_positive_copies')])
                gt_supported=bool(u[k,2,SUPPORT.index('GT_positive_copies')])
                for j in range(1,len(METRICS)):
                    has_support=gt_supported if METRICS[j] in ('GT_scene_FNR','GT_scene_E2E_risk') else tp_supported
                    value=m[k,r,j]
                    if not has_support:
                        if not math.isnan(value):raise RuntimeError('missing evaluation support must remain NA')
                    elif s[k,r]==2 and METRICS[j]=='normalized_radius_scene_mean':
                        if value!=math.inf:raise RuntimeError('all-space geometry cost hidden')
                    elif not math.isfinite(value):raise RuntimeError('available deployment endpoint cannot become NA/inf')
                if tp_supported:
                    compare_number(m[k,r,METRICS.index('TP_scene_coverage')],1-m[k,r,METRICS.index('TP_scene_risk')],'risk/coverage complement')
                    compare_number(m[k,r,METRICS.index('infinite_fraction')],float(s[k,r]==2),'infinite fraction')
    expected={'macro_metrics':np.mean(m,axis=0),'macro_FNR':np.asarray(np.mean(saved['raw_FNR'])),
              'class_fraction_reported':np.mean(s>0,axis=0),'class_fraction_finite':np.mean(s==1,axis=0),
              'class_fraction_all_space':np.mean(s==2,axis=0),'class_fraction_G2':np.mean(gp,axis=0)}
    for key,value in expected.items():
        if not np.array_equal(saved[key],value,equal_nan=True):raise RuntimeError('cached summary mismatch: '+key)


def reference_draw(cell,weights,record,saved):
    """Literal expanded-copy calibration, without optimized histograms/rank cache logic."""
    comparisons=0;max_error=0.
    for k in range(cell.k):
        one=cell.roles['G1_cal'][k];two=cell.roles['G2_cal'][k]
        one_groups=one.score_groups();two_groups=two.score_groups()
        expanded=[g for g,c in zip(one_groups,weights['G1_cal']) for _ in range(int(c))]
        hcp=hcp_threshold(expanded,Fraction(1,10))
        owner=saved[f'representative_source_{k}'];index=saved[f'representative_index_{k}']
        expected_owner=np.repeat(np.arange(len(one.m)),weights['G1_cal']);expected_owner=expected_owner[one.m[expected_owner]>0]
        if not np.array_equal(owner,expected_owner):raise RuntimeError('representative owner ordering mismatch')
        scores=np.array([one_groups[int(i)][int(j)] for i,j in zip(owner,index)])
        rank=tolerance_rank(len(scores),Fraction(1,10),cell.eta)
        rep=None if rank.rank is None else float(np.sort(scores)[rank.rank-1])
        g1loss=[[Fraction(int(np.count_nonzero(g>q)),len(g)) for g in expanded if len(g)] for q in Q]
        eb=direct_sequence(g1loss,cell.eta,Fraction(1,10))
        copies=np.repeat(np.arange(len(two.m)),weights['G2_cal']);gt=copies[two.H[copies]>0]
        recall_loss=[Fraction(int(two.fn[i]),int(two.H[i])) for i in gt]
        bound=empirical_bernstein(recall_loss,cell.eta)
        if bound.upper!=saved['g2_upper'][k]:raise RuntimeError('optimized G2 bound not bitwise reference-equivalent')
        g2=[bound.passes(Fraction(1,5)),bound.passes(Fraction(1,10))]
        directloss=[[Fraction(int(two.H[i])-int(np.count_nonzero(two_groups[i]<=q)),int(two.H[i])) for i in gt] for q in Q]
        direct20=direct_sequence(directloss,cell.direct_eta,Fraction(3,10));direct10=direct_sequence(directloss,cell.direct_eta,Fraction(1,5))
        qs=[hcp,rep,eb,direct20,direct10,rep if g2[0] else None,rep if g2[1] else None,
            eb if g2[0] else None,eb if g2[1] else None]
        ev=cell.roles['eval'][k];evgroups=ev.score_groups();evcopies=np.repeat(np.arange(len(ev.m)),weights['eval'])
        e_tp=evcopies[ev.m[evcopies]>0];e_gt=evcopies[ev.H[evcopies]>0]
        for r,q in enumerate(qs):
            status=0 if q is None else 2 if math.isinf(q) else 1
            if saved['status'][k,r]!=status:raise RuntimeError('certificate status mismatch')
            if q is None:
                if not np.all(np.isnan(saved['metrics'][k,r])):raise RuntimeError('refused deployment has non-NA metrics')
                continue
            if saved['metrics'][k,r,0]!=q:raise RuntimeError('refit q mismatch')
            risk=float(np.mean([np.mean(evgroups[i]>q) for i in e_tp])) if len(e_tp) else math.nan
            fnr=float(np.mean([ev.fn[i]/ev.H[i] for i in e_gt])) if len(e_gt) else math.nan
            e2e=float(np.mean([(ev.fn[i]+np.count_nonzero(evgroups[i]>q))/ev.H[i] for i in e_gt])) if len(e_gt) else math.nan
            for metric,reference in [('TP_scene_risk',risk),('GT_scene_FNR',fnr),('GT_scene_E2E_risk',e2e)]:
                actual=saved['metrics'][k,r,METRICS.index(metric)];comparisons+=1
                err=compare_number(actual,reference,'evaluation denominator/weight '+metric)
                max_error=max(max_error,err)
    return comparisons,max_error


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['pilot','full'],required=True);args=parser.parse_args(argv)
    with writer_lock(HERE/'writer.lock'):
        signature=check_freeze();run=run_directory(args.mode,signature);target=2 if args.mode=='pilot' else 5000
        protocol=json.loads((HERE/'source_snapshot/s5/PROTOCOL.json').read_text());rows=[];intervals={}
        started=time.monotonic()
        for info in protocol['cells']:
            cell_start=time.monotonic();heavy_seconds=0.;light_seconds=[]
            cell=CellData(info);folder=run/cell.name;draws=sorted(p for p in folder.glob('draw-*') if p.is_dir())
            if len(draws)!=target:raise RuntimeError('cannot verify incomplete predetermined B')
            rng=cell.role_rng();previous=None;metrics=[];status=[];support=[];recall=[];g2=[];comparison=0;max_error=0.
            for b,directory in enumerate(draws):
                light_start=time.monotonic()
                if directory.name!=f'draw-{b:05d}':raise RuntimeError('draw gap')
                if b%100==0:check_freeze(signature)
                record_path=directory/'record.json';record_raw=record_path.read_bytes()
                record=json.loads(record_raw);file=directory/'data.npz'
                verify_metadata(record,cell,b,signature,previous)
                if sha(file)!=record['data_sha256']:raise RuntimeError('draw integrity mismatch')
                if record['role_rng_before']!=rng.bit_generator.state:raise RuntimeError('RNG before mismatch')
                weights={}
                for role in ROLES:
                    n=len(cell.role_names[role]);chosen=rng.integers(0,n,size=n,dtype=np.int64)
                    weights[role]=np.bincount(chosen,minlength=n)
                if record['role_rng_after']!=rng.bit_generator.state:raise RuntimeError('RNG after mismatch')
                with np.load(file,allow_pickle=False) as z:
                    saved={k:z[k] for k in z.files}
                for role in ROLES:
                    if not np.array_equal(saved['weights_'+role],weights[role]):raise RuntimeError('role copies not replayable')
                verify_representatives(cell,weights,record,saved,b)
                verify_saved_invariants(cell,weights,saved)
                reference_seconds=0.
                if args.mode=='pilot' or b in (0,999,1999,2999,3999,4999):
                    reference_start=time.monotonic()
                    n,error=reference_draw(cell,weights,record,saved);comparison+=n;max_error=max(max_error,error)
                    reference_seconds=time.monotonic()-reference_start;heavy_seconds+=reference_seconds
                metrics.append(saved['metrics']);status.append(saved['status']);support.append(saved['support']);recall.append(saved['raw_FNR']);g2.append(saved['g2_pass'])
                if record_path.read_bytes()!=record_raw or sha(file)!=record['data_sha256']:
                    raise RuntimeError('draw changed during verification')
                previous=sha(record_path)
                light_seconds.append(time.monotonic()-light_start-reference_seconds)
            checkpoint=json.loads((folder/'checkpoint.json').read_text())
            verify_checkpoint(checkpoint,target,signature,previous,rng.bit_generator.state)
            rows.append({'cell':cell.name,'draws_verified':target,'canonical_representative_draws_verified':target,'support_and_availability_draws_verified':target,'expanded_reference_metric_comparisons':comparison,'max_metric_error':max_error,
                         'cell_verification_seconds':time.monotonic()-cell_start,'expanded_reference_seconds':heavy_seconds,
                         'per_draw_integrity_seconds':light_seconds})
            if args.mode=='full':
                m=np.asarray(metrics);s=np.asarray(status);u=np.asarray(support);raw=np.asarray(recall);gp=np.asarray(g2)
                out={'class_intervals':{},'macro_intervals':{},'availability_intervals':{},'support_intervals':{},'raw_FNR':{}}
                for k,name in enumerate(cell.classes):
                    out['class_intervals'][str(name)]={route:{metric:interval(m[:,k,r,j]) for j,metric in enumerate(METRICS)} for r,route in enumerate(ROUTES)}
                    out['raw_FNR'][str(name)]=interval(raw[:,k])
                    out['support_intervals'][str(name)]={role:{key:interval(u[:,k,j,t]) for t,key in enumerate(SUPPORT)} for j,role in enumerate(ROLES)}
                for r,route in enumerate(ROUTES):
                    out['macro_intervals'][route]={metric:interval(np.mean(m[:,:,r,j],axis=1)) for j,metric in enumerate(METRICS)}
                    out['availability_intervals'][route]={name:interval(np.mean(mask[:,:,r],axis=1)) for name,mask in [('reported',s>0),('finite',s==1),('all_space',s==2)]}
                out['G2_class_fraction']={beta:interval(np.mean(gp[:,:,i],axis=1)) for i,beta in enumerate(['.2','.1'])}
                out['raw_FNR_macro']=interval(np.mean(raw,axis=1))
                intervals[cell.name]=out
            print(json.dumps(rows[-1]),flush=True)
        check_freeze(signature)
        result={'status':'PASS_PILOT_EXACT_REFERENCE_AND_INTEGRITY' if args.mode=='pilot' else 'PASS_FULL_INTERVAL_RECOMPUTATION',
                'source_freeze_sha256':signature,'protocol_sha256':sha(HERE/'PROTOCOL.json'),'wall_seconds':time.monotonic()-started,
                'cells':rows,'independence_of_audit':'separate expanded reference implementation, same worker; external independent reviewer still required',
                'bootstrap_interpretation':'marginal descriptive approximate intervals; not PAC theorem, new independent scenes, or familywide confidence'}
        if args.mode=='full':atomic_json(run/'DESCRIPTIVE-INTERVALS.json',sanitize(intervals))
        atomic_json(run/'VERIFIED.json',result)
        print(json.dumps({'status':result['status'],'seconds':result['wall_seconds']}),flush=True)

if __name__=='__main__':main()

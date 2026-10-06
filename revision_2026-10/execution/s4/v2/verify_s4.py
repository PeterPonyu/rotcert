#!/usr/bin/env python
"""Independent S4 verifier: role/RNG replay, reference recalibration and CI tails.

Does not import statistics_core threshold/calibration/aggregation routines or
runner. Uses explicit source groups + accepted exact contracts as reference,
and directly re-aggregates selected endpoint sufficient statistics.
"""
from config import *
from io_s4 import *
from inputs import InputSet,roles_from_ids
from prepare import load_model
from rotcert_contracts.exact import pooled_threshold,hcp_threshold
from rotcert_contracts.geometry import angle_projection,canonical_box,rp1_distance
from fractions import Fraction
from scipy.stats import binom
import argparse,math,time,gc
from endpoint_reference import domains,reconstruct,validate_prepared
from capacity import shape_plan,STACK_KEYS
from resources import tick,write_allowance,write_finished
from functools import lru_cache


def expected_weights(inputs,draw):
 roles=roles_from_ids(inputs.ids);w=np.zeros((2,len(inputs.ids)),dtype=np.int64)
 for ri,name in enumerate(['calibration','evaluation']):
  ids=np.array([i for i,s in enumerate(inputs.ids) if roles[s]==name])
  if draw==-1:w[ri,ids]=1
  else:
   gen=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20260928,4,0,draw,ri])))
   chosen=gen.choice(ids,len(ids),replace=True)
   np.add.at(w[ri],chosen,1)
 return w


def expected_meta(source,ih,model,tau,draw):
 return dict(schema='S4-draw-v1',source_sha256=source,input_binding_sha256=ih,model=model,iou=tau,draw_id=draw,
             RNG='PCG64 SeedSequence([20260928,4,0,draw,role]); one shared source draw across all cells')


def ref_fit(data,k,cm,method):
 ix=data['tp_class']==k;s=data['tp_source'][ix];v=data['tp_score'][ix]
 # Reference reconstructs repeated source groups literally, independently of
 # runner's cardinality-group prefix arrays; bounded sample checks only.
 groups=[]
 for si in np.unique(s):groups.extend([v[s==si]]*int(cm[si]))
 return (pooled_threshold if method==0 else hcp_threshold)(groups,'0.1')


def ref_coverage(data,k,q,em):
 gt,det,tp=data['counts'][:,k].T
 ix=data['tp_class']==k;s=data['tp_source'][ix];score=data['tp_score'][ix]
 hit=np.bincount(s[score<=q],minlength=len(em));support=em[tp>0].sum()
 scene=float(np.dot(em[tp>0],hit[tp>0]/tp[tp>0])/support) if support else math.nan
 numerator=int(em@hit);total=int(em@tp)
 obj=numerator/total if total else math.nan
 recall=int(em@tp)/int(em@gt) if int(em@gt) else math.nan
 e2e=numerator/int(em@gt) if int(em@gt) else math.nan
 return scene,obj,recall,e2e


def verify_one(r,inputs,data,draw,deep):
 w=expected_weights(inputs,draw);np.testing.assert_array_equal(r['multiplicity'],w)
 expected=np.zeros((6,2,20,len(SUPPORT_NAMES)),dtype=np.int64)
 for ci,d in enumerate(data):
  for ri,m in enumerate(w):
   for k in range(20):
    gt,det,tp=d['counts'][:,k].T
    expected[ci,ri,k]=sum(m),sum(m>0),m[gt>0].sum(),m[tp>0].sum(),m@gt,m@tp,m@det
 np.testing.assert_array_equal(expected,r['support'])
 domains(r,data,w)
 np.testing.assert_array_equal(r['values'].mean(axis=2),r['macro'])
 with np.errstate(invalid='ignore'):
  for i,c in enumerate(CONTRASTS):
   np.testing.assert_array_equal(r['values'][c['left']]-r['values'][c['right']],r['contrasts'][i])
   np.testing.assert_array_equal(r['macro'][c['left']]-r['macro'][c['right']],r['macro_contrasts'][i])
 # Independently reconstruct global all-source counts, including zeroGT/zeroTP.
 for ai,arm in enumerate(ARMS):
  d=data[arm['eval']];gt,det,tp=d['counts'].sum(axis=1).T;m=w[1]
  expected_all=[int(m.sum()),int(m[gt>0].sum()),int(m[tp>0].sum()),int(m[tp==0].sum()),int(m[(gt>0)&(tp==0)].sum()),int(m@gt),int(m@tp),int(m@(det-tp)),int(m@(gt-tp)),int(m@det)]
  for method in range(2):np.testing.assert_array_equal(r['all_source'][ai,method,:10],expected_all)
 # Mass and raw count conservation tested for all arms/methods/classes.
 for label in ['tp_count','gt_count','detection_count']:
  require(np.all(r['values'][...,METRICS.index(label)]>=0),'Negative count')
 np.testing.assert_array_equal(r['values'][...,METRICS.index('tp_count')]+r['values'][...,METRICS.index('fn_count')],r['values'][...,METRICS.index('gt_count')])
 np.testing.assert_array_equal(r['values'][...,METRICS.index('tp_count')]+r['values'][...,METRICS.index('fp_count')],r['values'][...,METRICS.index('detection_count')])
 if deep:
  for ci,d in enumerate(data):
   for method in range(2):
    for k in range(20):require(ref_fit(d,k,w[0],method)==r['q'][ci,method,k],'Independent exact recalibration failed')
  compared=reconstruct(r,data,w)
 else:compared=0
 return dict(exact_rng=True,canonical_support=True,strict_macro=True,all_endpoint_domain_contracts=True,independent_recalibration=deep,
             literal_class_and_global_endpoint_scalars=compared,geometry_scope='approved immutable scalar oracle plus independently reconstructed source-copy aggregation/full boundary; float orientation convention replay only, no new geometry theorem certification')



def ref_quantile(x,p):
 """Independent order-statistic interpolation, including infinities."""
 z=sorted(float(v) for v in x);pos=(len(z)-1)*p;i=int(math.floor(pos));j=int(math.ceil(pos))
 if i==j or z[i]==z[j]:return z[i]
 if z[i]==-math.inf and z[j]==math.inf:return math.nan
 if math.isinf(z[i]):return z[i]
 if math.isinf(z[j]):return z[j]
 return (1-(pos-i))*z[i]+(pos-i)*z[j]



@lru_cache(maxsize=16)
def independent_mc_ranks(N,p):
 rank=np.arange(N+1);cdf=binom.cdf(rank,N,p)
 return int(np.searchsorted(cdf,.025)),int(np.searchsorted(cdf,.975))+1

def ref_interval(x,level):
 x=np.asarray(x,float);N=len(x);miss=int(np.isnan(x).sum())
 base=dict(B=N,missing_draws=miss,positive_infinite_draws=int(np.isposinf(x).sum()),negative_infinite_draws=int(np.isneginf(x).sum()),level=level)
 if miss:return dict(base,status='UNAVAILABLE_REQUIRED_DRAW_NA',percentile=None,tail_MC=None,five_blocks1000=None)
 tails=[(1-level)/2,(1+level)/2];ends=[ref_quantile(x,p) for p in tails]
 if any(math.isnan(v) for v in ends):return dict(base,status='UNAVAILABLE_INDETERMINATE_INFINITY_INTERPOLATION',percentile=None,tail_MC=None,five_blocks1000=None)
 z=np.sort(x);mc=[]
 for p in tails:
  # Independent inversion of binomial cdf rather than library ppf call.
  lo,hi=independent_mc_ranks(N,p)
  mc.append(dict(tail_probability=p,point=ref_quantile(x,p),binomial_order95_bracket=[-math.inf if lo==0 else float(z[lo-1]),math.inf if hi>N else float(z[hi-1])],ranks=[lo,hi],expected_tail_draws=N*min(p,1-p)))
 return dict(base,status='APPROXIMATE_PERCENTILE',percentile=ends,tail_MC=mc,five_blocks1000=[[ref_quantile(x[i*1000:(i+1)*1000],p) for p in tails] for i in range(5)] if N==5000 else None)


def check_interval(ref,got):
 for k in ['status','B','missing_draws','positive_infinite_draws','negative_infinite_draws']:require(ref[k]==got[k],'Independent CI mismatch '+k)
 if ref['percentile'] is not None:
  np.testing.assert_allclose(ref['percentile'],got['percentile'],rtol=5e-15,atol=5e-15,equal_nan=True)
  if ref['five_blocks1000'] is not None:np.testing.assert_allclose(ref['five_blocks1000'],got['five_blocks1000'],rtol=5e-15,atol=5e-15,equal_nan=True)
  for a,b in zip(ref['tail_MC'],got['tail_MC']):
   require(a['ranks']==b['ranks'],'Quantile MC rank mismatch');np.testing.assert_array_equal(a['binomial_order95_bracket'],b['binomial_order95_bracket'])


def scratch_allocate(directory,count,example):
 # Preflight all five cell tensors before first allocation. Never retain12cells.
 total=sum(count*example[k].nbytes+1024 for k in STACK_KEYS)
 tick('verification_scratch',allocation=total)
 stacks={};paths=[]
 for key in STACK_KEYS:
  p=directory/('verify-'+key+'.npy');require(not p.exists(),'Prior scratch remains: preserve/investigate')
  write_allowance(p,count*example[key].nbytes+1024)
  stacks[key]=np.lib.format.open_memmap(p,mode='w+',dtype=example[key].dtype,shape=(count,)+example[key].shape)
  paths.append(p);write_finished(p)
 return stacks,paths


def scratch_release(stacks,paths,receipt):
 # Only scratch listed by this invocation, only after an immutable verified CI
 # receipt commits. Inputs/draw checkpoints are never removal candidates.
 require(receipt.exists(),'No committed interval receipt; scratch must be preserved')
 digests={str(p.name):sha(p) for p in paths}
 for arr in stacks.values():arr.flush();arr._mmap.close()
 stacks.clear();gc.collect()
 for p in paths:
  require(p.name.startswith('verify-') and p.suffix=='.npy','Not owned scratch')
  p.unlink();write_finished(p)
 return digests


def interval_benchmark():
 from intervals import interval
 # Algorithm cost diagnostic, not a scientific interval from2draws.
 x=np.sin(np.arange(5000)/13)+np.arange(5000)/5000;cost=[]
 for level in [.95,.996875]:
  for i in range(3):
   tick('interval_capacity_pilot');start=time.process_time();a=interval(x+i,level);b=ref_interval(x+i,level);check_interval(b,a);cost.append(time.process_time()-start)
 return dict(scalar_vectors=6,B=5000,scientific_results=False,max_cpu_per_endpoint=max(cost),mean_cpu_per_endpoint=float(np.mean(cost)))


def verify_run(inputs,out,source,count,deep=True):
 out=owned(out);tick('verification');verify();require(count in (2,5000),'Verifier count')
 records=[];paired_hash={};interval_records=[];light_cost=[];deep_cost=[];start_cpu=time.process_time();start_wall=time.monotonic()
 for model in MODELS:
  for tau in TAUS:
   tick('verification');directory=out/model/f'iou{tau:.1f}';data=load_model(inputs,out/'prepared',source,model,tau);validate_prepared(data,len(inputs.ids))
   require(len(list(directory.glob('*.npz')))==count+1,'Unexpected draw files')
   require(not list(directory.glob('*.tmp-*')),'Orphan temporary')
   stacks={};mempaths=[];cell_deep=0
   for draw in [-1]+list(range(count)):
    tick('verification');begin=time.process_time()
    p=directory/('base.npz' if draw==-1 else f'draw-{draw:04d}.npz')
    r=load_npz(p,expected_meta(source,inputs.binding_hash,model,tau,draw))
    chosen=deep and (count==2 or draw in [-1,0,1,2499,4999])
    # Pilot times light and full paths separately to budget the entire full audit.
    if count==2:
     verify_one(r,inputs,data,draw,False);light_cost.append(time.process_time()-begin)
    check=verify_one(r,inputs,data,draw,chosen)
    (deep_cost if chosen else light_cost).append(time.process_time()-begin)
    if chosen:cell_deep+=1
    mh=__import__('hashlib').sha256(r['multiplicity'].tobytes()).hexdigest()
    if draw in paired_hash:require(paired_hash[draw]==mh,'Source bootstrap differs across model/tau cells')
    else:paired_hash[draw]=mh
    if count==5000 and draw>=0:
     if not stacks:stacks,mempaths=scratch_allocate(directory,count,r)
     for key in STACK_KEYS:stacks[key][draw]=r[key]
    if draw>=0 and draw%100==0:inputs.guard.check();verify()
   if count==5000:
    from intervals import interval,coverage_level,contrast_numerical_status
    result={}
    for key,values in stacks.items():
     per={}
     for ix in np.ndindex(values.shape[1:]):
      tick('verification_intervals')
      level=coverage_level(tau,ix[1],CONTRASTS[ix[0]]) if key=='macro_contrasts' and ix[-1]==0 else .95
      a=values[(slice(None),)+ix];got=interval(a,level);ref=ref_interval(a,level);check_interval(ref,got)
      if key=='macro_contrasts' and ix[-1]==0:got['numerical_resolution']=contrast_numerical_status(got)
      per['/'.join(map(str,ix))]=got
     result[key]=per
    for arr in stacks.values():arr.flush()
    del values,a
    receipt=directory/'INTERVALS.json'
    write_json(receipt,dict(source_sha256=source,input_binding_sha256=inputs.binding_hash,model=model,iou=tau,arrays=result,metrics=METRICS,interpretation='Marginal exploratory, except prespecified HCP16family approximate99.6875percent; no cross-family simultaneous claim'))
    scratch_hashes=scratch_release(stacks,mempaths,receipt)
    interval_records.append(dict(model=model,iou=tau,independent_all_interval_recompute='PASS',released_scratch_hashes=scratch_hashes,interval_sha256=sha(receipt)))
    del result,per,r;gc.collect()
   records.append(dict(model=model,iou=tau,checkpoints=count+1,deep_draws=cell_deep,all_endpoint_domains_every_draw=True,independent_literal_all_endpoints_deep=True))
 inputs.guard.check(full=True);verify();tick('verification')
 cost=dict(max_light_cpu_per_draw=max(light_cost,default=0.),max_deep_cpu_per_draw=max(deep_cost,default=0.),
           cpu_seconds=time.process_time()-start_cpu,wall_seconds=time.monotonic()-start_wall)
 cost['wall_to_cpu_ratio']=max(1.,cost['wall_seconds']/max(cost['cpu_seconds'],1e-9))
 bench=interval_benchmark() if count==2 else None
 record=dict(status='PASS_DECLARED_NUMERICAL_ENDPOINT_SCOPE',source_sha256=source,input_binding_sha256=inputs.binding_hash,synthetic=inputs.synthetic,draws_per_cell=count,
             exact_role_RNG_replay='ALL',paired_weight_hashes=paired_hash,receipts=records,interval_receipts=interval_records,
             pilot_no_intervals=(count==2),scientific_claim='No true new-shift conclusions from synthetic fixtures',utc=now(),cost=cost,interval_capacity_benchmark=bench,
             verification_scope='all-draw domain/support/hash/RNG/strict macro/contrasts; selected deep draws full literal class/global endpoint reconstruction; angle geometry uses approved scalar oracle, independently aggregated; float-pi joint membership replay, not directed transcendental proof',
             external_geometry_theorem_review='not replaced by this verifier')
 write_json(out/'VERIFIED.json',record)
 return dict(status=record['status'],verified_sha256=sha(out/'VERIFIED.json'),cost=cost,interval_capacity_benchmark=bench)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--out',required=True);p.add_argument('--synthetic',action='store_true');a=p.parse_args()
 from run_s4 import run
 out=owned(a.out);mode=json.loads((out/'COMPLETE.json').read_text())['mode']
 print(run(a.index,mode,out,a.synthetic))

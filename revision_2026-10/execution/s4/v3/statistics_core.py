"""Shared-weight fixed-role full-pipeline S4 functional, exact class thresholds."""
from config import *
from io_s4 import require
from fractions import Fraction
from functools import lru_cache
import math
import numpy as np
from rotcert_contracts.geometry import angle_projection,canonical_box,rp1_distance


def multiplicities(roles,draw):
 out=np.zeros((2,len(roles)),dtype=np.int64)
 for role in range(2):
  ids=np.flatnonzero(roles==role)
  if draw==-1:out[role,ids]=1
  else:
   require(type(draw) is int and 0<=draw<B,'Draw id domain')
   # One shared analysis cell0, no model/condition/tau/method key.
   g=np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEED,4,0,draw,role])))
   out[role]=np.bincount(g.choice(ids,size=len(ids),replace=True),minlength=len(roles))
 return out


class ExactWeightedThreshold:
 """CDF mass by original scene size; all decisions use integer/Fraction math.

 Score/order arrays are prepared once. Per-draw source multiplicities multiply
 object copies but do not change the within-original-scene count denominator.
 """
 def __init__(self,values,sources,n):
  self.v=np.asarray(values,float);self.s=np.asarray(sources,np.int64)
  require(np.isfinite(self.v).all() and (self.v>=0).all(),'Threshold scores')
  self.count=np.bincount(self.s,minlength=n);self.candidates=np.unique(self.v)
  self.sorted=[]
  for size in np.unique(self.count[self.count>0]):
   ix=np.flatnonzero(self.count[self.s]==size);ix=ix[np.argsort(self.v[ix],kind='stable')]
   self.sorted.append((int(size),self.v[ix],self.s[ix]))
  self.order=np.argsort(self.v,kind='stable')
 def fit(self,mult,method):
  if method==0:
   counts=mult[self.s[self.order]];n=int(counts.sum());rank=(9*(n+1)+9)//10
   if rank>n:return math.inf
   return float(self.v[self.order[np.searchsorted(np.cumsum(counts),rank,side='left')]])
  K=int(mult[self.count>0].sum());target=Fraction(9*(K+1),10)
  if not K or K<target:return math.inf
  tables=[(size,val,np.r_[0,np.cumsum(mult[src],dtype=np.int64)]) for size,val,src in self.sorted]
  def reached(q):
   return sum((Fraction(int(pref[np.searchsorted(val,q,side='right')]),size) for size,val,pref in tables),Fraction())>=target
  lo=0;hi=len(self.candidates)-1
  while lo<hi:
   mid=(lo+hi)//2
   if reached(self.candidates[mid]):hi=mid
   else:lo=mid+1
  return float(self.candidates[lo])


# The original scalar geometry remains the independent reference. A separate
# array kernel accepts only certified binary64 endpoint bounds or falls back.
from collections import OrderedDict
from resources import tick


def _projection_arrays(w, h, q):
 from fast_geometry import project_arrays
 return project_arrays(w, h, q)


def _centers_inside(pred, truth, q):
 """Exact Fraction decisions, with a directed longdouble interval fast path."""
 if math.isinf(q):return np.ones(len(pred),dtype=bool)
 L=np.longdouble
 neg=L(-np.inf);pos=L(np.inf)
 # Float64 inputs are exact in longdouble on this platform; otherwise fallback.
 from fast_geometry import _platform_supported
 if not _platform_supported():
  return np.asarray([(Fraction(float(p[0]))-Fraction(float(g[0])))**2+(Fraction(float(p[1]))-Fraction(float(g[1])))**2<=Fraction(q)**2 for p,g in zip(pred,truth)],bool)
 delta=pred[:,:2].astype(L)-truth[:,:2].astype(L)
 lo=np.nextafter(delta,neg);hi=np.nextafter(delta,pos)
 low_abs=np.where((lo<=0)&(hi>=0),L(0),np.minimum(np.abs(lo),np.abs(hi)))
 high_abs=np.maximum(np.abs(lo),np.abs(hi))
 low_square=np.nextafter(low_abs*low_abs,neg);high_square=np.nextafter(high_abs*high_abs,pos)
 lower=np.nextafter(low_square.sum(axis=1),neg);upper=np.nextafter(high_square.sum(axis=1),pos)
 q2=L(q)*L(q);ql=np.nextafter(q2,neg);qh=np.nextafter(q2,pos)
 inside=upper<=ql;outside=lower>qh
 uncertain=~(inside|outside)
 for i in np.flatnonzero(uncertain):
  dx=Fraction(float(pred[i,0]))-Fraction(float(truth[i,0]));dy=Fraction(float(pred[i,1]))-Fraction(float(truth[i,1]))
  inside[i]=dx*dx+dy*dy<=Fraction(q)**2
 return inside


def _mean_by_source(values, starts, counts):
 values=np.asarray(values,dtype=np.longdouble)
 if not len(values):return np.empty(0,dtype=np.longdouble)
 return np.add.reduceat(values,starts)/counts


def _weighted_source_mean(values, source_ids, multiplicity):
 weights=multiplicity[source_ids];active=weights>0
 if not active.any():return math.nan
 values=np.asarray(values,dtype=np.longdouble)[active];weights=weights[active]
 if np.isnan(values).any():return math.nan
 if np.isposinf(values).any():return math.inf
 if np.isneginf(values).any():return -math.inf
 return float(np.sum(values*weights,dtype=np.longdouble)/int(weights.sum()))


def weighted_mean(x,w):
 """Stable weighted reduction; exact constant indicators stay exactly 0/1."""
 active=np.asarray(w)>0
 if not active.any():return math.nan
 x=np.asarray(x)[active];w=np.asarray(w)[active]
 if np.isnan(x).any():return math.nan
 if np.isposinf(x).any() and np.isneginf(x).any():return math.nan
 if np.isinf(x).any():return math.inf if np.isposinf(x).any() else -math.inf
 return float(np.sum(x.astype(np.longdouble)*w,dtype=np.longdouble)/np.sum(w,dtype=np.longdouble))


def object_weights(sources,counts,mult):return mult[sources]/counts[sources]
def ratio(num,den):return float(num/den) if den else math.nan


def scene_rate(numer,denom,mult):
 active=(denom>0)&(mult>0)
 if not active.any():return math.nan
 return float(np.sum(numer[active].astype(np.longdouble)/denom[active]*mult[active],dtype=np.longdouble)/int(mult[active].sum()))


class ClassData:
 """Immutable q-independent indices; only evaluation-role objects retained."""
 def __init__(self,data,k,roles):
  self.counts=data['counts'][:,k]
  tp_rows=np.flatnonzero((data['tp_class']==k)&(roles[data['tp_source']]==1))
  det_rows=np.flatnonzero((data['det_class']==k)&(roles[data['det_source']]==1))
  self.tp_source=data['tp_source'][tp_rows];self.score=data['tp_score'][tp_rows]
  self.pred=data['tp_pred'][tp_rows];self.truth=data['tp_gt'][tp_rows]
  self.det_source=data['det_source'][det_rows];self.det_boxes=data['det_box'][det_rows]
  self.tp_ids,self.tp_starts,self.tp_counts=np.unique(self.tp_source,return_index=True,return_counts=True)
  self.det_ids,self.det_starts,self.det_counts=np.unique(self.det_source,return_index=True,return_counts=True)
  require(np.all(np.diff(self.tp_source)>=0) and np.all(np.diff(self.det_source)>=0),'Prepared source ordering')
  self.distance=np.empty(len(self.pred));self.truth_nonsquare=np.empty(len(self.pred),dtype=bool)
  for i,(p,g) in enumerate(zip(self.pred,self.truth)):
   pw,ph,pt=canonical_box(*map(float,p[2:]));gw,gh,gt=canonical_box(*map(float,g[2:]))
   self.distance[i]=rp1_distance(pt,gt);self.truth_nonsquare[i]=gw!=gh
  self.tp_scale=np.sqrt(self.pred[:,2]*self.pred[:,3]);self.det_scale=np.sqrt(self.det_boxes[:,2]*self.det_boxes[:,3])

 def sufficient(self,q):
  """Per-original-source deterministic summaries independent of bootstrap draw.

  q is still refitted on every draw. Reuse is only a memo of the same fixed
  data+q function; source multiplicities and calibration support are never cached.
  """
  tick('fitting')
  low,high,full=_projection_arrays(self.pred[:,2],self.pred[:,3],q)
  inside=_centers_inside(self.pred,self.truth,q)
  jlo=inside&(full|(self.truth_nonsquare&(self.distance<=low)))
  jhi=inside&(full|(self.truth_nonsquare&(self.distance<=high)))
  covered=self.score<=q
  hits=np.add.reduceat(covered.astype(np.int64),self.tp_starts) if len(covered) else np.empty(0,dtype=np.int64)
  norm=np.full(len(self.pred),q)/self.tp_scale
  tp_means=np.asarray([_mean_by_source(a,self.tp_starts,self.tp_counts) for a in [norm,~full,full,low,high,jlo,jhi]])
  from fast_geometry import full_arrays
  det_full=full_arrays(self.det_boxes[:,2],self.det_boxes[:,3],q)
  dn=np.full(len(self.det_boxes),q)/self.det_scale
  det_means=np.asarray([_mean_by_source(a,self.det_starts,self.det_counts) for a in [dn,~det_full]])
  return hits,tp_means,det_means

 def endpoint(self,q,cal_support,em,summary):
  gt,det,tp=self.counts.T;n=len(gt)
  hits,tp_means,det_means=summary
  tpsum=int(em@tp);gtsum=int(em@gt);detsum=int(em@det)
  hitsum=int(em[self.tp_ids]@hits)
  full_hits=np.zeros(n,dtype=np.int64);full_hits[self.tp_ids]=hits
  positive=tp>0
  v=[scene_rate(full_hits,tp,em),ratio(hitsum,tpsum),weighted_mean(full_hits[positive]==tp[positive],em[positive]),q,float(math.isfinite(q))]
  v.extend(_weighted_source_mean(a,self.tp_ids,em) for a in tp_means)
  alln=int(em.sum())
  v.extend([scene_rate(tp,gt,em),ratio(tpsum,gtsum),scene_rate(full_hits,gt,em),ratio(hitsum,gtsum),
    ratio(int(em@(tp>0)),alln),ratio(int(em@(tp==0)),alln),ratio(int(em@((gt>0)&(tp==0))),alln),
    tpsum,detsum-tpsum,gtsum-tpsum,gtsum,detsum,int(em@(tp>0)),int(em@(gt>0)),int(em@(tp==0)),
    cal_support[0],cal_support[1],float(cal_support[0]==0),float(math.isfinite(q)) if detsum else math.nan,
    _weighted_source_mean(det_means[0],self.det_ids,em),_weighted_source_mean(det_means[1],self.det_ids,em)])
  require(len(v)==len(METRICS),'Endpoint schema differs')
  return np.asarray(v,dtype=float)


# Public scalar wrappers retained for focused regression fixtures.
def geometry(boxes,truth,q):
 low,high,full=_projection_arrays(boxes[:,2],boxes[:,3],q)
 normalized=np.full(len(boxes),q)/np.sqrt(boxes[:,2]*boxes[:,3])
 if truth is None:return normalized,low,high,full,None,None
 inside=_centers_inside(boxes,truth,q);alo=np.empty(len(boxes),bool);ahi=np.empty(len(boxes),bool)
 for i,(p,g) in enumerate(zip(boxes,truth)):
  pw,ph,pt=canonical_box(*map(float,p[2:]));gw,gh,gt=canonical_box(*map(float,g[2:]));d=rp1_distance(pt,gt)
  alo[i]=full[i] or (gw!=gh and d<=low[i]);ahi[i]=full[i] or (gw!=gh and d<=high[i])
 return normalized,low,high,full,inside&alo,inside&ahi


def projection(w,h,q):
 a=angle_projection(w,h,q);return a.half_width_lower,a.half_width_upper,a.full


def class_endpoint(data,k,q,cal_support,eval_mult):
 roles=np.where(eval_mult>0,1,0);c=ClassData(data,k,roles)
 return c.endpoint(q,cal_support,eval_mult,c.sufficient(q))


class ModelPipeline:
 def __init__(self,conditions,roles):
  self.conditions=conditions;self.roles=roles;self.n=len(roles);self.models=[];self.classes=[]
  for data in conditions:
   models=[];classes=[]
   for k in range(20):
    ix=(data['tp_class']==k)&(roles[data['tp_source']]==0)
    models.append(ExactWeightedThreshold(data['tp_score'][ix],data['tp_source'][ix],self.n))
    classes.append(ClassData(data,k,roles))
   self.models.append(models);self.classes.append(classes)
  # A bounded per-cell memo: no threshold/RNG/sampling reuse; all entries are
  # deterministic per-source sufficient statistics for an exact q value.
  self.cache=OrderedDict();self.cache_container_reserve=1024**2
  self.cache_bytes=self.cache_container_reserve;self.cache_limit=384*1024**2
  # Per-entry allowance covers3ndarray headers, key/value tuples, q/size
  # objects and OrderedDict nodes;1MiB base reserve covers container overhead.
  self.cache_entry_overhead=2048
  self.cache_peak_charged_bytes=self.cache_bytes
  self.cache_hits=0;self.cache_misses=0

 def summaries(self,ci,k,q):
  key=(ci,k,float(q))
  if key in self.cache:
   self.cache_hits+=1;value,size=self.cache.pop(key);self.cache[key]=(value,size);return value
  self.cache_misses+=1;ep=self.classes[ci][k]
  # Output dimensions are known before construction. Evict first, so new
  # retained entries cannot transiently coexist with an already-full memo.
  payload=len(ep.tp_ids)*(8+7*np.dtype(np.longdouble).itemsize)+len(ep.det_ids)*2*np.dtype(np.longdouble).itemsize
  size=payload+self.cache_entry_overhead
  while self.cache and self.cache_bytes+size>self.cache_limit:
   removed_key,(removed_value,removed_size)=self.cache.popitem(last=False)
   self.cache_bytes-=removed_size
   del removed_key,removed_value,removed_size
  value=ep.sufficient(q)
  require(sum(a.nbytes for a in value)==payload,'Memo dimensions changed')
  if self.cache_bytes+size<=self.cache_limit:
   self.cache[key]=(value,size);self.cache_bytes+=size
   self.cache_peak_charged_bytes=max(self.cache_peak_charged_bytes,self.cache_bytes)
  return value

 def run(self,mult):
  require(mult.shape==(2,self.n),'Multiplicity shape');cm,em=mult
  require(not np.any(cm[self.roles!=0]) and not np.any(em[self.roles!=1]),'Source role mismatch')
  q=np.empty((6,2,20));support=np.zeros((6,2,20,len(SUPPORT_NAMES)),dtype=np.int64)
  for ci,data in enumerate(self.conditions):
   for role,m in enumerate(mult):
    counts=data['counts'];gt,det,tp=(counts[:,:,j] for j in range(3))
    support[ci,role,:,0]=int(m.sum());support[ci,role,:,1]=np.count_nonzero(m)
    support[ci,role,:,2]=m@(gt>0);support[ci,role,:,3]=m@(tp>0)
    support[ci,role,:,4]=m@gt;support[ci,role,:,5]=m@tp;support[ci,role,:,6]=m@det
   for method in range(2):
    for k in range(20):q[ci,method,k]=self.models[ci][k].fit(cm,method)
  values=np.full((len(ARMS),2,20,len(METRICS)),np.nan)
  endpoint_cache={}
  for ai,arm in enumerate(ARMS):
   tick('fitting')
   for method in range(2):
    for k in range(20):
     threshold=float(q[arm['cal'],method,k]);cs=support[arm['cal'],0,k]
     key=(arm['eval'],k,threshold,int(cs[3]),int(cs[5]))
     if key not in endpoint_cache:
      ep=self.classes[arm['eval']][k]
      endpoint_cache[key]=ep.endpoint(threshold,(cs[3],cs[5]),em,self.summaries(arm['eval'],k,threshold))
     values[ai,method,k]=endpoint_cache[key]
  all_source=np.full((len(ARMS),2,len(ALL_SOURCE_METRICS)),np.nan)
  for ai,arm in enumerate(ARMS):
   data=self.conditions[arm['eval']];gt,det,tp=data['counts'].sum(axis=1).T
   for method in range(2):
    covered=data['tp_score']<=q[arm['cal'],method,data['tp_class']]
    hits=np.bincount(data['tp_source'][covered],minlength=self.n)
    tg=int(em@gt);tt=int(em@tp);dd=int(em@det);hh=int(em@hits)
    all_source[ai,method]=[int(em.sum()),int(em@(gt>0)),int(em@(tp>0)),int(em@(tp==0)),int(em@((gt>0)&(tp==0))),tg,tt,dd-tt,tg-tt,dd,
      scene_rate(tp,gt,em),ratio(tt,tg),scene_rate(hits,gt,em),ratio(hh,tg),scene_rate(hits,tp,em),ratio(hh,tt)]
  macro=np.mean(values,axis=2)
  with np.errstate(invalid='ignore'):
   contrasts=np.stack([values[c['left']]-values[c['right']] for c in CONTRASTS])
   macro_contrasts=np.stack([macro[c['left']]-macro[c['right']] for c in CONTRASTS])
  return dict(values=values,macro=macro,contrasts=contrasts,macro_contrasts=macro_contrasts,q=q,support=support,all_source=all_source)

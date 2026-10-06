"""Literal independent endpoint reconstruction, no statistics_core imports.

Scalar angle enclosures are recomputed with the immutable approved contracts
oracle. This checks application/aggregation, not an independent theorem proof.
The joint event deliberately replays the declared float-pi orientation convention;
its bounds are NOT claimed to enclose exact transcendental angle arithmetic.
"""
from config import *
from io_s4 import require,tick
from fractions import Fraction as F
from functools import lru_cache
from rotcert_contracts.geometry import angle_projection
import math
import numpy as np

TP_METRICS={'scene_coverage','object_coverage','scene_all_tp_coverage','normalized_center_radius',
 'angle_informative_rate','angle_full_rate','angle_halfwidth_lower_rad','angle_halfwidth_upper_rad',
 'joint_center_angle_coverage_lower','joint_center_angle_coverage_upper'}
GT_METRICS={'all_gt_scene_recall','all_gt_object_recall','all_gt_scene_e2e','all_gt_object_e2e'}
DET_METRICS={'all_detection_finite_q_rate','all_detection_normalized_center_radius','all_detection_informative_angle_rate'}
COST_METRICS={'q_px','normalized_center_radius','all_detection_normalized_center_radius'}
ANGLE_METRICS={'angle_halfwidth_lower_rad','angle_halfwidth_upper_rad'}
COUNT_METRICS={n for n in METRICS if n.endswith('_count')}
RATE_METRICS=set(METRICS)-COST_METRICS-ANGLE_METRICS-COUNT_METRICS
AGG_ATOL=4e-15

def validate_prepared(data,n):
 for d in data:
  count=d['counts'];require(count.shape==(n,20,3) and count.dtype.kind in 'iu' and np.all(count>=0),'Prepared counts domain')
  require(np.all(count[:,:,2]<=count[:,:,0]) and np.all(count[:,:,2]<=count[:,:,1]),'Prepared matching conservation')
  for prefix in ['tp','det']:
   sc=d[prefix+'_source'];cl=d[prefix+'_class'];p=d['tp_pred'] if prefix=='tp' else d['det_box']
   require(sc.dtype.kind in 'iu' and cl.dtype.kind in 'iu' and sc.shape==cl.shape==(len(p),),'Prepared source/class metadata')
   require(np.all((sc>=0)&(sc<n)) and np.all((cl>=0)&(cl<20)),'Prepared index range')
   require(p.shape==(len(sc),5) and np.all(np.isfinite(p)) and np.all(p[:,2:4]>0),'Prepared geometry input')
   reconstructed=np.bincount(sc*20+cl,minlength=n*20).reshape(n,20)
   np.testing.assert_array_equal(reconstructed,count[:,:,2 if prefix=='tp' else 1])
  gt=d['tp_gt'];score=d['tp_score'];iou=d['tp_iou']
  require(gt.shape==d['tp_pred'].shape and np.isfinite(gt).all() and (gt[:,2:4]>0).all(),'Matched GT geometry domain')
  require(score.shape==(len(gt),) and np.isfinite(score).all() and (score>=0).all(),'Score domain')
  require(iou.shape==score.shape and np.isfinite(iou).all() and ((iou>=.5)&(iou<=1)).all(),'Matched IoU domain')

def count_reference(data,w):
 """Independent integer contractions, recomputed from prepared counts per draw."""
 support=np.zeros((len(data),2,20,len(SUPPORT_NAMES)),dtype=np.int64)
 support[...,0]=w.sum(axis=1)[None,:,None]
 support[...,1]=np.count_nonzero(w,axis=1)[None,:,None]
 detection_sources=np.empty((len(data),20),dtype=np.int64)
 global_counts=np.empty((len(data),10),dtype=np.int64);em=w[1];alln=int(em.sum())
 for ci,d in enumerate(data):
  count=d['counts']
  weighted=np.einsum('rn,nkc->rkc',w,count)
  positive=np.einsum('rn,nkc->rkc',w,count>0)
  support[ci,:,:,2]=positive[:,:,0];support[ci,:,:,3]=positive[:,:,2]
  support[ci,:,:,4]=weighted[:,:,0];support[ci,:,:,5]=weighted[:,:,2];support[ci,:,:,6]=weighted[:,:,1]
  detection_sources[ci]=positive[1,:,1]
  gt,det,tp=count.sum(axis=1).T;ng,nd,nt=map(int,weighted[1].sum(axis=0))
  global_counts[ci]=alln,int(em[gt>0].sum()),int(em[tp>0].sum()),int(em[tp==0].sum()),int(em[(gt>0)&(tp==0)].sum()),ng,nt,nd-nt,ng-nt,nd
 return support,detection_sources,global_counts


def domains(r,data,w,counts=None):
 """All saved endpoints, every draw; finite/NA/support and exact count contracts."""
 expected_shapes={'q':(6,2,20),'support':(6,2,20,7),'values':(11,2,20,33),'macro':(11,2,33),
                  'contrasts':(10,2,20,33),'macro_contrasts':(10,2,33),'all_source':(11,2,16)}
 for name,shape in expected_shapes.items():require(r[name].shape==shape,'Endpoint shape '+name)
 q=r['q'];require(np.all(~np.isnan(q)) and np.all(q>=0) and not np.any(np.isneginf(q)),'All threshold domain')
 missing=r['support'][:,0,:,3]==0
 require(np.all(~missing[:,None,:]|np.isposinf(q)),'Empty calibration must retain infinity')
 support,detection_sources,global_counts=count_reference(data,w) if counts is None else counts
 cal=[a['cal'] for a in ARMS];evaluation=[a['eval'] for a in ARMS]
 cs=r['support'][cal,0];es=support[evaluation,1];alln=int(w[1].sum())
 # Match the frozen scalar float(...) conversion before arithmetic, including
 # float32 storage: its rounded complement must not mask a float64 violation.
 threshold=np.asarray(q[cal],dtype=float);v=np.asarray(r['values'],dtype=float);ix={name:i for i,name in enumerate(METRICS)}
 ref_counts={'tp_count':es[:,:,5],'fp_count':es[:,:,6]-es[:,:,5],'fn_count':es[:,:,4]-es[:,:,5],
  'gt_count':es[:,:,4],'detection_count':es[:,:,6],'tp_source_count':es[:,:,3],'gt_positive_source_count':es[:,:,2],
  'zero_tp_source_count':alln-es[:,:,3],'cal_tp_source_count':cs[:,:,3],'cal_tp_object_count':cs[:,:,5]}
 has_tp=es[:,None,:,3]>0;has_gt=es[:,None,:,2]>0;has_det=detection_sources[evaluation,None,:]>0
 for j,name in enumerate(METRICS):
  x=v[...,j];has=has_tp if name in TP_METRICS else has_gt if name in GT_METRICS else has_det if name in DET_METRICS else True
  require(np.all(has|np.isnan(x)),'Missing support must yield NA: '+name)
  require(np.all(~np.asarray(has)|(~np.isnan(x)&(x!=-math.inf))),'Supported metric NA/negative infinity: '+name)
  if name in COST_METRICS:valid=(x>=0)&(np.isinf(x)==np.isinf(threshold));message='Radius/threshold infinity contract: '
  elif name in ANGLE_METRICS:valid=np.isfinite(x)&(x>=-AGG_ATOL)&(x<=math.pi/2+AGG_ATOL);message='Angle range: '
  elif name in COUNT_METRICS:
   # Python float/int equality does not round integer counts above 2**53.
   valid=x.astype(object)==ref_counts[name][:,None,:].astype(object);message='Exact count differs: '
  else:valid=np.isfinite(x)&(x>=-AGG_ATOL)&(x<=1+AGG_ATOL);message='Rate range: '
  require(np.all(~np.asarray(has)|valid),message+name)
 def x(name):return v[...,ix[name]]
 # Frozen verifier checked these identities on the raw stored scalar dtype.
 # Preserve extra precision rather than rounding a corrupt flag to float64.
 raw=r['values']
 require(np.all(raw[...,ix['q_px']]==threshold),'Endpoint q differs from fit')
 require(np.all(raw[...,ix['finite_q_rate']]==np.isfinite(threshold)),'Finite q flag')
 require(np.all(raw[...,ix['cal_class_absent']]==(cs[:,None,:,3]==0)),'Empty class flag')
 require(np.all(~has_tp|(x('angle_halfwidth_lower_rad')<=x('angle_halfwidth_upper_rad')+AGG_ATOL)),'Reversed angle brackets')
 require(np.all(~has_tp|(x('joint_center_angle_coverage_lower')<=x('joint_center_angle_coverage_upper')+AGG_ATOL)),'Reversed joint brackets')
 require(np.all(~has_tp|(np.abs(x('angle_informative_rate')+x('angle_full_rate')-1)<=AGG_ATOL)),'Full/informative complement')
 allspace=has_tp&np.isinf(threshold)
 require(np.all(~allspace|((x('angle_full_rate')==1)&(x('angle_informative_rate')==0))),'All-space orientation flags')
 require(np.all(~allspace|(np.abs(x('angle_halfwidth_upper_rad')-math.pi/2)<=AGG_ATOL)),'All-space angular cost')
 require(np.all(~has_gt|((x('all_gt_scene_e2e')<=x('all_gt_scene_recall')+AGG_ATOL)&(x('all_gt_object_e2e')<=x('all_gt_object_recall')+AGG_ATOL))),'E2E exceeds recall')
 g=r['all_source'];gc=global_counts[evaluation]
 require(np.isfinite(g[...,:10]).all() and np.all(g[...,:10]>=0) and np.all(g[...,:10]==np.floor(g[...,:10])),'Global count range')
 for j in range(10,16):
  supported=gc[:,None,5 if j<14 else 6]>0;ratio=np.asarray(g[...,j],dtype=float)
  valid=np.isfinite(ratio)&(ratio>=-AGG_ATOL)&(ratio<=1+AGG_ATOL)
  require(np.all(np.where(supported,valid,np.isnan(ratio))),'Global ratio range/NA: '+ALL_SOURCE_METRICS[j])
 require(np.all((gc[:,None,5]==0)|((g[...,12]<=g[...,10]+AGG_ATOL)&(g[...,13]<=g[...,11]+AGG_ATOL))),'Global E2E exceeds recall')

@lru_cache(maxsize=65536)
def approved_projection(w,h,q):
 a=angle_projection(w,h,q)
 wf,hf=F(w),F(h);qf=None if math.isinf(q) else F(q)
 full=math.isinf(q) or qf*qf>=(wf-hf)**2/8
 require(full==a.full,'Independent rational full-orientation boundary')
 require(0<=a.half_width_lower<=a.half_width_upper<=math.pi/2,'Approved angular bracket domain')
 if full:require(a.half_width_lower==a.half_width_upper==math.pi/2,'Full orientation oracle')
 return a.half_width_lower,a.half_width_upper,full

def canonical_angle(box):
 w,h,t=map(float,box[2:]);return (w,h,math.remainder(t,math.pi)) if w>=h else (h,w,math.remainder(t+math.pi/2,math.pi))

def geometry_rows(boxes,truth,q):
 out=[]
 for i,p in enumerate(boxes):
  tick('verification')
  lo,hi,full=approved_projection(float(p[2]),float(p[3]),float(q))
  norm=q/math.sqrt(float(p[2])*float(p[3]))
  jlo=jhi=math.nan
  if truth is not None:
   g=truth[i];dx=F(float(p[0]))-F(float(g[0]));dy=F(float(p[1]))-F(float(g[1]))
   center=math.isinf(q) or dx*dx+dy*dy<=F(q)**2
   pw,ph,pt=canonical_angle(p);gw,gh,gt=canonical_angle(g)
   d=abs(math.remainder(math.remainder(pt,math.pi)-math.remainder(gt,math.pi),math.pi))
   jlo=float(center and (full or (gw!=gh and d<=lo)));jhi=float(center and (full or (gw!=gh and d<=hi)))
  out.append((norm,float(not full),float(full),lo,hi,jlo,jhi))
 return out

def average(values):
 if not values:return math.nan
 if any(math.isnan(x) for x in values):return math.nan
 if any(x==math.inf for x in values):return math.inf
 return math.fsum(values)/len(values)

def mean_ratios(num,den,copies):
 vals=[F(int(num[s]),int(den[s])) for s in copies if den[s]>0]
 return float(sum(vals,F())/len(vals)) if vals else math.nan

def div(a,b):return float(F(int(a),int(b))) if b else math.nan

def literal_class(data,k,q,cal_data,cm,em):
 """Copies enumerated explicitly; ratios use exact integer/Fraction sums."""
 gt,det,tp=data['counts'][:,k].T;copies=np.repeat(np.arange(len(em)),em)
 rows=np.flatnonzero(data['tp_class']==k);s=data['tp_source'][rows];score=data['tp_score'][rows]
 hit=np.zeros(len(em),dtype=np.int64)
 for src,value in zip(s,score):
  if value<=q:hit[int(src)]+=1
 nt=sum(int(tp[i]) for i in copies);ng=sum(int(gt[i]) for i in copies);nd=sum(int(det[i]) for i in copies);nh=sum(int(hit[i]) for i in copies)
 tp_copies=[int(i) for i in copies if tp[i]>0];gt_copies=[int(i) for i in copies if gt[i]>0]
 caltp=cal_data['counts'][:,k,2]
 vals={'scene_coverage':mean_ratios(hit,tp,copies),'object_coverage':div(nh,nt),'scene_all_tp_coverage':div(sum(hit[i]==tp[i] for i in tp_copies),len(tp_copies)),
       'q_px':q,'finite_q_rate':float(math.isfinite(q)),'all_gt_scene_recall':mean_ratios(tp,gt,copies),'all_gt_object_recall':div(nt,ng),
       'all_gt_scene_e2e':mean_ratios(hit,gt,copies),'all_gt_object_e2e':div(nh,ng),'tp_source_rate':div(len(tp_copies),len(copies)),
       'zero_tp_source_rate':div(len(copies)-len(tp_copies),len(copies)),'gt_positive_zero_tp_source_rate':div(sum(gt[i]>0 and tp[i]==0 for i in copies),len(copies)),
       'tp_count':nt,'fp_count':nd-nt,'fn_count':ng-nt,'gt_count':ng,'detection_count':nd,'tp_source_count':len(tp_copies),'gt_positive_source_count':len(gt_copies),
       'zero_tp_source_count':len(copies)-len(tp_copies),'cal_tp_source_count':int(cm@(caltp>0)),'cal_tp_object_count':int(cm@caltp),'cal_class_absent':float(cm@(caltp>0)==0)}
 active=em[s]>0;rows=rows[active];s=s[active]
 geom=geometry_rows(data['tp_pred'][rows],data['tp_gt'][rows],q)
 labels=['normalized_center_radius','angle_informative_rate','angle_full_rate','angle_halfwidth_lower_rad','angle_halfwidth_upper_rad','joint_center_angle_coverage_lower','joint_center_angle_coverage_upper']
 by_source={int(i):[] for i in np.unique(s)}
 for src,row in zip(s,geom):by_source[int(src)].append(row)
 means={si:[average([a[j] for a in rr]) for j in range(7)] for si,rr in by_source.items()}
 for j,label in enumerate(labels):vals[label]=average([means[i][j] for i in tp_copies])
 dr=np.flatnonzero(data['det_class']==k);ds=data['det_source'][dr];act=em[ds]>0;dr=dr[act];ds=ds[act]
 dg=geometry_rows(data['det_box'][dr],None,q);dd={int(i):[] for i in np.unique(ds)}
 for src,row in zip(ds,dg):dd[int(src)].append(row)
 dcp=[int(i) for i in copies if det[i]>0]
 dmean={si:[average([a[j] for a in rr]) for j in [0,1]] for si,rr in dd.items()}
 vals['all_detection_finite_q_rate']=float(math.isfinite(q)) if dcp else math.nan
 vals['all_detection_normalized_center_radius']=average([dmean[i][0] for i in dcp])
 vals['all_detection_informative_angle_rate']=average([dmean[i][1] for i in dcp])
 return np.asarray([vals[n] for n in METRICS])

def literal_global(data,qs,em):
 gt,det,tp=data['counts'].sum(axis=1).T;copies=np.repeat(np.arange(len(em)),em);hit=np.zeros(len(em),dtype=np.int64)
 for src,k,s in zip(data['tp_source'],data['tp_class'],data['tp_score']):
  if s<=qs[k]:hit[src]+=1
 nt=sum(int(tp[i]) for i in copies);ng=sum(int(gt[i]) for i in copies);nd=sum(int(det[i]) for i in copies);nh=sum(int(hit[i]) for i in copies)
 return np.asarray([len(copies),sum(gt[i]>0 for i in copies),sum(tp[i]>0 for i in copies),sum(tp[i]==0 for i in copies),sum(gt[i]>0 and tp[i]==0 for i in copies),ng,nt,nd-nt,ng-nt,nd,
 mean_ratios(tp,gt,copies),div(nt,ng),mean_ratios(hit,gt,copies),div(nh,ng),mean_ratios(hit,tp,copies),div(nh,nt)],float)

def reconstruct(r,data,w):
 cm,em=w;memo={};comparisons=0
 for ai,a in enumerate(ARMS):
  tick('verification')
  for method in range(2):
   qs=r['q'][a['cal'],method]
   for k,q in enumerate(qs):
    key=(a['eval'],a['cal'],k,float(q))
    if key not in memo:memo[key]=literal_class(data[a['eval']],k,float(q),data[a['cal']],cm,em)
    # Aggregation tolerance only; calibration/full-geometry decisions exact.
    np.testing.assert_allclose(r['values'][ai,method,k],memo[key],rtol=4e-15,atol=AGG_ATOL,equal_nan=True,
                               err_msg=f'literal class endpoint arm={ai} method={method} class={k}')
    comparisons+=len(METRICS)
   np.testing.assert_allclose(r['all_source'][ai,method],literal_global(data[a['eval']],qs,em),rtol=4e-15,atol=AGG_ATOL,equal_nan=True,err_msg='literal all-source endpoint')
   comparisons+=len(ALL_SOURCE_METRICS)
 return comparisons

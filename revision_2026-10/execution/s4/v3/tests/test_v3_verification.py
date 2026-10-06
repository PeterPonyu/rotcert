"""Bounded verifier equivalence/mutation tests; no bootstrap job or inference.

The frozen v2 implementation supplies the before-behavior oracle. Synthetic
endpoints are constructed by its literal Fraction/fsum reference, independently
of candidate statistics_core and fast_geometry. Real checkpoints are read-only.
"""
import ast
import copy
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import config
import endpoint_reference as reference
import verify_s4 as verifier
from inputs import roles_from_ids
from intervals import interval

ROOT=Path(__file__).resolve().parents[1]
TARGET=Path(os.environ.get('ROTCERT_S4_RECORDS_ROOT',ROOT.parent/'records'))
ORIGINAL=ROOT.parent/'v2'


def frozen_module(name):
 spec=importlib.util.spec_from_file_location('frozen_v2_'+name,ORIGINAL/(name+'.py'))
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module


old_reference=frozen_module('endpoint_reference')
old_verifier=frozen_module('verify_s4')
old_verifier.domains=old_reference.domains
old_verifier.reconstruct=old_reference.reconstruct


def fixture_data():
 """32 sources: finite/inf fits, absent cal, zeroGT, zeroTP, unequal sizes."""
 inputs=SimpleNamespace(ids=[f'verifier-source-{i:02}' for i in range(32)])
 roles=roles_from_ids(inputs.ids)
 data=[]
 for ci in range(6):
  counts=np.zeros((32,20,3),dtype=np.int64)
  source=[];classes=[];pred=[];truth=[];scores=[];ds=[];dc=[];db=[]
  for si,sid in enumerate(inputs.ids):
   calibration=roles[sid]=='calibration'
   # Class 0 has source-size heterogeneity and guaranteed finite calibration.
   ct=1+si%3;counts[si,0]=[ct+int(si%5==0),ct+int(si%4==0),ct]
   if not calibration:
    if si%7==0:counts[si,0]=[0,1,0]
    elif si%5==0:counts[si,0]=[2,1,0]
    counts[si,1]=[1,1,1] # absent calibration, supported infinite-q evaluation
    counts[si,2]=[2,1,0] # GT/detections supported, no TP
    counts[si,3]=[0,1,0] # detections only, zero GT/TP
   else:counts[si,4]=[1,1,1] # calibration only, evaluation NA
   if ci==3:counts[si,:,2]=0 # global GT/detection support without TP
   if ci==4:counts[si,:,0]=0;counts[si,:,2]=0 # global detections without GT
   if ci==5:counts[si]=0 # no global or class support, including calibration
   for k in range(5):
    for j in range(int(counts[si,k,2])):
     p=[si*.01,j*.03,12.+si%3,3.+j%2,.01*(si%4)]
     g=[p[0]+.125,p[1],p[2],p[3],p[4]+.025]
     source.append(si);classes.append(k);pred.append(p);truth.append(g)
     scores.append(float(1+(si+j+ci)%3))
    for j in range(int(counts[si,k,1])):
     ds.append(si);dc.append(k);db.append([si*.01,j*.03,12.+si%3,3.+j%2,.01*(si%4)])
  data.append(dict(counts=counts,tp_source=np.asarray(source,dtype=np.int64),tp_class=np.asarray(classes,dtype=np.int16),
   tp_pred=np.asarray(pred,float).reshape(-1,5),tp_gt=np.asarray(truth,float).reshape(-1,5),tp_score=np.asarray(scores),
   tp_iou=np.full(len(scores),.8),det_source=np.asarray(ds,dtype=np.int64),det_class=np.asarray(dc,dtype=np.int16),
   det_box=np.asarray(db,float).reshape(-1,5)))
 return inputs,data


def literal_record(inputs,data,draw):
 w=old_verifier.expected_weights(inputs,draw)
 q=np.empty((6,2,20));support=np.zeros((6,2,20,7),dtype=np.int64)
 for ci,d in enumerate(data):
  for ri,m in enumerate(w):
   for k in range(20):
    gt,det,tp=d['counts'][:,k].T
    support[ci,ri,k]=sum(m),sum(m>0),m[gt>0].sum(),m[tp>0].sum(),m@gt,m@tp,m@det
  for method in range(2):
   for k in range(20):q[ci,method,k]=old_verifier.ref_fit(d,k,w[0],method)
 values=np.empty((11,2,20,33));all_source=np.empty((11,2,16))
 for ai,a in enumerate(config.ARMS):
  for method in range(2):
   qs=q[a['cal'],method]
   for k,threshold in enumerate(qs):
    values[ai,method,k]=old_reference.literal_class(data[a['eval']],k,float(threshold),data[a['cal']],*w)
   all_source[ai,method]=old_reference.literal_global(data[a['eval']],qs,w[1])
 r=dict(q=q,support=support,values=values,all_source=all_source,multiplicity=w)
 refresh_summaries(r)
 return r


def refresh_summaries(r):
 with np.errstate(invalid='ignore'):
  r['macro']=r['values'].mean(axis=2)
  r['contrasts']=np.stack([r['values'][c['left']]-r['values'][c['right']] for c in config.CONTRASTS])
  r['macro_contrasts']=np.stack([r['macro'][c['left']]-r['macro'][c['right']] for c in config.CONTRASTS])


class VerificationTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.inputs,cls.data=fixture_data()
  reference.validate_prepared(cls.data,len(cls.inputs.ids))
  cls.records={draw:literal_record(cls.inputs,cls.data,draw) for draw in [-1,0,1,2499,4999]}

 def rejected_by_both(self,r,deep=False):
  for verify in (old_verifier.verify_one,verifier.verify_one):
   with self.subTest(verifier=verify.__module__):
    with self.assertRaises((AssertionError,ValueError)):
     verify(r,self.inputs,self.data,-1,deep)

 def test_complete_selected_draws_match_frozen_deep_reference(self):
  for draw,r in self.records.items():
   with self.subTest(draw=draw):
    expected=old_verifier.verify_one(r,self.inputs,self.data,draw,True)
    actual=verifier.verify_one(r,self.inputs,self.data,draw,True)
    self.assertEqual(expected,actual)
    self.assertEqual(actual['literal_class_and_global_endpoint_scalars'],11*2*(20*33+16))

 def test_roles_and_rng_cached_by_complete_ordered_identity(self):
  verifier.role_indices.cache_clear()
  with mock.patch.object(verifier,'roles_from_ids',wraps=roles_from_ids) as role_hash:
   for draw in [-1,0,1,2499,4999]:
    np.testing.assert_array_equal(verifier.expected_weights(self.inputs,draw),old_verifier.expected_weights(self.inputs,draw))
   self.assertEqual(role_hash.call_count,1)
   for ids in [list(reversed(self.inputs.ids)),self.inputs.ids[:-1]+['new-source']]:
    changed=SimpleNamespace(ids=ids,binding_hash='same-untrusted-binding')
    np.testing.assert_array_equal(verifier.expected_weights(changed,0),old_verifier.expected_weights(changed,0))
   self.assertEqual(role_hash.call_count,3)
   changed=SimpleNamespace(ids=self.inputs.ids.copy())
   changed.ids[0]='another-source'
   np.testing.assert_array_equal(verifier.expected_weights(changed,1),old_verifier.expected_weights(changed,1))
   self.assertEqual(role_hash.call_count,4)
  for indices in verifier.role_indices(config.PROTOCOL_ID,tuple(self.inputs.ids)):
   with self.assertRaises(ValueError):indices.flags.writeable=True
  with self.assertRaises(ValueError):verifier.expected_weights(SimpleNamespace(ids=['duplicate']*2),-1)
  with self.assertRaises(ValueError):verifier.role_indices('different-protocol',tuple(self.inputs.ids))

 def test_vector_counts_equal_scalar_for_every_fixture_draw(self):
  for r in self.records.values():
   support,nd,glob=reference.count_reference(self.data,r['multiplicity'])
   np.testing.assert_array_equal(support,r['support'])
   for ci,d in enumerate(self.data):
    em=r['multiplicity'][1]
    np.testing.assert_array_equal(nd[ci],[int(em@(d['counts'][:,k,1]>0)) for k in range(20)])
   np.testing.assert_array_equal(glob[[a['eval'] for a in config.ARMS]],r['all_source'][:,0,:10])

 def test_ref_fit_literal_source_copy_groups_and_data_mutations(self):
  rng=np.random.default_rng(174)
  for n in [0,8,9,11,32]:
   sources=np.repeat(np.arange(n),rng.integers(1,5,size=n));order=rng.permutation(len(sources))
   d=dict(tp_source=sources[order],tp_class=np.zeros(len(sources),dtype=np.int16),tp_score=rng.integers(0,8,size=len(sources)).astype(float))
   for cm in [np.ones(n,dtype=np.int64),rng.integers(0,4,size=n),np.zeros(n,dtype=np.int64)]:
    for method in range(2):self.assertEqual(verifier.ref_fit(d,0,cm,method),old_verifier.ref_fit(d,0,cm,method))
   if n:
    d['tp_score'][:]+=19;d['tp_source'][:]=d['tp_source'][::-1]
    for method in range(2):self.assertEqual(verifier.ref_fit(d,0,np.ones(n,dtype=np.int64),method),old_verifier.ref_fit(d,0,np.ones(n,dtype=np.int64),method))

 def test_every_metric_domain_na_infinity_and_counts_mutations(self):
  base=self.records[-1]
  for k in (0,1,2,3,4,5):
   for j,name in enumerate(config.METRICS):
    old=float(base['values'][0,0,k,j])
    replacement=(0. if math.isnan(old) else 1. if math.isinf(old) else old+1 if name in reference.COUNT_METRICS else math.nan)
    r=copy.deepcopy(base);r['values'][0,0,k,j]=replacement;refresh_summaries(r)
    with self.subTest(metric=name,class_index=k):self.rejected_by_both(r)
  for j,name in enumerate(config.METRICS):
   for value in [-math.inf,math.inf] if name not in reference.COST_METRICS else [-math.inf]:
    r=copy.deepcopy(base);r['values'][0,0,0,j]=value;refresh_summaries(r)
    with self.subTest(metric=name,invalid_infinity=value):self.rejected_by_both(r)
  for name in reference.COST_METRICS:
   for value in [-1.,-math.inf,math.inf]:
    r=copy.deepcopy(base);r['values'][0,0,0,config.METRICS.index(name)]=value;refresh_summaries(r)
    with self.subTest(metric=name,cost=value):self.rejected_by_both(r)

 def test_exact_agg_tolerance_rate_and_angle_boundaries(self):
  self.assertEqual(reference.AGG_ATOL,4e-15)
  base=self.records[-1]
  for name in ['scene_coverage','all_detection_finite_q_rate']:
   for accepted,value in [(True,1+4e-15),(False,np.nextafter(1+4e-15,math.inf)),(False,1.0000000000000042),(True,-4e-15),(False,np.nextafter(-4e-15,-math.inf))]:
    r=copy.deepcopy(base);r['values'][0,0,0,config.METRICS.index(name)]=value;refresh_summaries(r)
    with self.subTest(metric=name,value=value):
     if accepted:
      for verify in (old_verifier.verify_one,verifier.verify_one):verify(r,self.inputs,self.data,-1,False)
     else:self.rejected_by_both(r)
  r=copy.deepcopy(base);r['values'][0,0,0,config.METRICS.index('angle_halfwidth_upper_rad')]=np.nextafter(math.pi/2+4e-15,math.inf);refresh_summaries(r)
  self.rejected_by_both(r)

 def test_scalar_float_conversion_and_large_exact_count_equality(self):
  r=copy.deepcopy(self.records[-1]);r['values']=r['values'].astype(np.float32)
  r['values'][0,0,0,config.METRICS.index('angle_informative_rate')]=.1
  r['values'][0,0,0,config.METRICS.index('angle_full_rate')]=.9
  for check in (old_reference.domains,reference.domains):
   with self.assertRaises(ValueError):check(r,self.data,r['multiplicity'])
  # The scalar frozen float/int comparison rejects a rounded count even when
  # NumPy's mixed float64/int64 comparison would round both to the same value.
  r=copy.deepcopy(self.records[-1]);data=copy.deepcopy(self.data)
  active=int(np.flatnonzero(r['multiplicity'][1])[0]);ng=int(r['multiplicity'][1]@data[0]['counts'][:,0,0])
  target=2**53+1;data[0]['counts'][active,0,0]+=target-ng
  nt=int(r['multiplicity'][1]@data[0]['counts'][:,0,2])
  r['values'][0,:,0,config.METRICS.index('gt_count')]=float(target)
  r['values'][0,:,0,config.METRICS.index('fn_count')]=float(target-nt)
  for check in (old_reference.domains,reference.domains):
   with self.assertRaisesRegex(ValueError,'Exact count differs:'):check(r,data,r['multiplicity'])

 def test_raw_longdouble_identity_corruption_not_rounded_away(self):
  self.assertGreater(np.finfo(np.longdouble).nmant,np.finfo(np.float64).nmant)
  for name,k,delta in [('finite_q_rate',0,np.ldexp(np.longdouble(1),-60)),('cal_class_absent',0,np.ldexp(np.longdouble(1),-1075)),('q_px',0,np.ldexp(np.longdouble(1),-60))]:
   r=copy.deepcopy(self.records[-1]);r['values']=r['values'].astype(np.longdouble);j=config.METRICS.index(name)
   r['values'][0,0,k,j]+=delta;refresh_summaries(r)
   with self.subTest(name=name):self.rejected_by_both(r)
 def test_brackets_complements_flags_and_e2e_mutations(self):
  changes=[{'angle_halfwidth_lower_rad':1.,'angle_halfwidth_upper_rad':.2},
   {'joint_center_angle_coverage_lower':1.,'joint_center_angle_coverage_upper':0.},
   {'angle_informative_rate':.25,'angle_full_rate':.25},
   {'all_gt_scene_recall':0.,'all_gt_scene_e2e':1.},
   {'all_gt_object_recall':0.,'all_gt_object_e2e':1.},
   {'finite_q_rate':0.},{'cal_class_absent':1.},{'q_px':.25}]
  for change in changes:
   r=copy.deepcopy(self.records[-1])
   for name,value in change.items():r['values'][0,0,0,config.METRICS.index(name)]=value
   refresh_summaries(r)
   with self.subTest(change=change):self.rejected_by_both(r)
  for change in [{'angle_full_rate':0.,'angle_informative_rate':1.},{'angle_halfwidth_lower_rad':.5,'angle_halfwidth_upper_rad':.5}]:
   r=copy.deepcopy(self.records[-1])
   for name,value in change.items():r['values'][0,0,1,config.METRICS.index(name)]=value
   refresh_summaries(r);self.rejected_by_both(r)

 def test_support_rng_macro_contrasts_and_q_mutations(self):
  for key in ['multiplicity','support','macro','contrasts','macro_contrasts','q']:
   r=copy.deepcopy(self.records[-1]);pos=tuple(np.argwhere(np.isfinite(r[key]))[0]);r[key][pos]+=1
   with self.subTest(key=key):self.rejected_by_both(r)
  for value in [math.nan,-1.,-math.inf]:
   r=copy.deepcopy(self.records[-1]);r['q'][0,0,0]=value;self.rejected_by_both(r)
  r=copy.deepcopy(self.records[-1]);r['q'][0,0,5]=0.;self.rejected_by_both(r)
  r=copy.deepcopy(self.records[-1]);r['macro'][0,0,0]=0.;self.rejected_by_both(r) # no nanmean
  for key in ['q','support','values','macro','contrasts','macro_contrasts','all_source']:
   r=copy.deepcopy(self.records[-1]);r[key]=r[key][:-1]
   with self.subTest(shape=key):self.rejected_by_both(r)

 def test_all_global_count_and_rate_mutations(self):
  for ai in (0,6,8,10):
   for j,name in enumerate(config.ALL_SOURCE_METRICS):
    r=copy.deepcopy(self.records[-1]);old=float(r['all_source'][ai,0,j])
    r['all_source'][ai,0,j]=(old+1 if j<10 else 0. if math.isnan(old) else math.nan)
    with self.subTest(metric=name,arm=ai):self.rejected_by_both(r)
  for index,value in [(0,-1.),(0,.5),(0,math.inf),(10,1.0000000000000042),(10,-4.1e-15),(10,math.inf)]:
   r=copy.deepcopy(self.records[-1]);r['all_source'][0,0,index]=value;self.rejected_by_both(r)
  for recall,e2e in [(10,12),(11,13)]:
   r=copy.deepcopy(self.records[-1]);r['all_source'][0,0,recall]=0.;r['all_source'][0,0,e2e]=1.;self.rejected_by_both(r)

 def test_deep_rejects_in_domain_endpoint_corruption(self):
  for key,pos in [('values',(0,0,0,config.METRICS.index('object_coverage'))),('values',(0,0,0,config.METRICS.index('normalized_center_radius'))),('all_source',(0,0,15))]:
   r=copy.deepcopy(self.records[-1]);r[key][pos]=float(r[key][pos])*.8+.001;refresh_summaries(r)
   for verify in (old_verifier.verify_one,verifier.verify_one):verify(r,self.inputs,self.data,-1,False)
   self.rejected_by_both(r,True)
  r=copy.deepcopy(self.records[-1]);r['q'][0,0,0]+=.1
  with self.assertRaisesRegex(ValueError,'Independent exact recalibration failed'):
   # Isolate exact fitting to avoid the deliberately inconsistent q endpoint.
   with mock.patch.object(verifier,'domains'),mock.patch.object(verifier,'reconstruct'):
    verifier.verify_one(r,self.inputs,self.data,-1,True)

 def test_literal_reconstruction_is_unchanged_and_independent(self):
  old=ast.parse((ORIGINAL/'endpoint_reference.py').read_text());new=ast.parse((ROOT/'endpoint_reference.py').read_text())
  names=['approved_projection','canonical_angle','geometry_rows','average','mean_ratios','div','literal_class','literal_global','reconstruct']
  for name in names:
   before=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
   after=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
   self.assertEqual(ast.dump(before),ast.dump(after),name)
  for tree in (new,ast.parse((ROOT/'verify_s4.py').read_text())):
   for node in ast.walk(tree):
    if isinstance(node,ast.Import):self.assertTrue(all(a.name not in ['statistics_core','fast_geometry'] for a in node.names))
    if isinstance(node,ast.ImportFrom):self.assertNotIn(node.module,['statistics_core','fast_geometry'])

 def test_full_B_independent_intervals_all_endpoint_numeric_classes(self):
  finite=np.sin(np.arange(5000)/13)+np.arange(5000)/5000
  cases=[finite,np.zeros(5000),np.full(5000,math.inf),np.full(5000,-math.inf),np.r_[np.full(2500,-math.inf),np.full(2500,math.inf)],np.r_[math.nan,finite[1:]]]
  for x in cases:
   for level in [.95,.996875]:
    actual=verifier.ref_interval(x,level);expected=old_verifier.ref_interval(x,level)
    verifier.check_interval(expected,actual);verifier.check_interval(actual,interval(x,level))
    if actual['percentile'] is not None:
     self.assertEqual(len(actual['five_blocks1000']),5)
     mutated=copy.deepcopy(actual);mutated['tail_MC'][0]['ranks'][0]+=1
     with self.assertRaises(ValueError):verifier.check_interval(actual,mutated)
  with self.assertRaises(ValueError):interval(finite[:2])

 def test_deep_draw_selection_and_all_B_interval_loop_unchanged(self):
  old=ast.parse((ORIGINAL/'verify_s4.py').read_text());new=ast.parse((ROOT/'verify_s4.py').read_text())
  for name in ['verify_run','ref_quantile','independent_mc_ranks','ref_interval','check_interval']:
   before=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
   after=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
   self.assertEqual(ast.dump(before),ast.dump(after),name)

 def test_saved_real_failed_checkpoint_remains_rejected(self):
  receipt=TARGET/'LOCAL-STATUS-20260929T120155Z/FIRST-FAILURE-SCALAR-REPRODUCTION.json'
  import json
  failure=json.loads(receipt.read_text());checkpoint=Path(failure['checkpoint'])
  # Keep the recorded checkpoint identity while relocating its archive root.
  p=TARGET.joinpath(*checkpoint.parts[checkpoint.parts.index('cpu-runtime-binding-v1'):])
  self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),failure['checkpoint_sha256'])
  with np.load(p,allow_pickle=False) as z:r={k:z[k] for k in z.files}
  prepared=p.parents[2]/'prepared';data=[]
  for condition in config.CONDITIONS:
   with np.load(prepared/f'dior-orcnn-s0__{condition}__iou0.5.npz',allow_pickle=False) as z:data.append({k:z[k] for k in z.files})
  value=r['values'][5,0,15,config.METRICS.index('all_detection_finite_q_rate')]
  self.assertEqual(value,1.0000000000000042)
  for check in (old_reference.domains,reference.domains):
   with self.assertRaisesRegex(ValueError,'Rate range: all_detection_finite_q_rate'):check(r,data,r['multiplicity'])


if __name__=='__main__':unittest.main(verbosity=2)

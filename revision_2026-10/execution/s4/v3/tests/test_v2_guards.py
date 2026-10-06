"""Bounded real-risk faults: resource refusal and plausible wrong endpoints."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import *
from io_s4 import *
from resources import Lifecycle,BudgetStop,tick
from capacity import shape_plan
from inputs import InputSet
from prepare import match_condition,prepare_all
from statistics_core import ModelPipeline,multiplicities
from verify_s4 import verify_one,scratch_allocate,scratch_release
from endpoint_reference import validate_prepared,approved_projection
from fixture import generate
import unittest,tempfile,copy,time
from unittest.mock import patch


def reseal(r):
 r['macro']=r['values'].mean(axis=2)
 with np.errstate(invalid='ignore'):
  r['contrasts']=np.stack([r['values'][a['left']]-r['values'][a['right']] for a in CONTRASTS])
  r['macro_contrasts']=np.stack([r['macro'][a['left']]-r['macro'][a['right']] for a in CONTRASTS])
 return r


def limits(**changes):
 a=dict(max_cpu_seconds=300.,max_wall_seconds=300.,max_output_bytes=2**28,minimum_free_disk_bytes=1024,minimum_available_memory_bytes=1024)
 a.update(changes);return a

class V2Guards(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  (ROOT/'fixtures').mkdir(exist_ok=True)
  cls.fixture_dir=tempfile.TemporaryDirectory(dir=ROOT/'fixtures',prefix='test-s4-guards-')
  cls.addClassCleanup(cls.fixture_dir.cleanup)
  cls.inputs=InputSet(generate(Path(cls.fixture_dir.name)/'synthetic-v1'),True)
  cls.data=[match_condition(cls.inputs,MODELS[0],c,.5) for c in CONDITIONS]
  cls.r=ModelPipeline(cls.data,cls.inputs.roles).run(multiplicities(cls.inputs.roles,-1))
  cls.r['multiplicity']=multiplicities(cls.inputs.roles,-1)
 def assert_rejected(self,r,deep=True):
  with self.assertRaises((ValueError,AssertionError)):verify_one(r,self.inputs,self.data,-1,deep)
 def test_R2_review_123_counterexample_rejected_even_shallow(self):
  r=copy.deepcopy(self.r);r['values'][0,1,0,METRICS.index('all_gt_scene_e2e')]=123
  self.assert_rejected(reseal(r),False)
 def test_R2_in_range_class_mutations_rejected_deep(self):
  for name in ['all_gt_scene_e2e','all_gt_scene_recall','scene_all_tp_coverage','all_gt_object_e2e','all_gt_object_recall']:
   with self.subTest(metric=name):
    r=copy.deepcopy(self.r);j=METRICS.index(name);x=r['values'][0,1,0,j];r['values'][0,1,0,j]=max(0.,x-.03125)
    self.assertNotEqual(x,r['values'][0,1,0,j]);self.assert_rejected(reseal(r))
 def test_R2_each_global_ratio_mutation_rejected(self):
  for j in range(10,16):
   with self.subTest(metric=ALL_SOURCE_METRICS[j]):
    r=copy.deepcopy(self.r);r['all_source'][0,1,j]=max(0.,r['all_source'][0,1,j]-.03125);self.assert_rejected(r)
 def test_R2_geometric_cost_flag_bracket_and_joint_faults(self):
  for name in ['normalized_center_radius','all_detection_normalized_center_radius','angle_halfwidth_upper_rad','angle_informative_rate','all_detection_informative_angle_rate','joint_center_angle_coverage_lower']:
   with self.subTest(metric=name):
    r=copy.deepcopy(self.r);j=METRICS.index(name);x=float(r['values'][0,1,0,j]);r['values'][0,1,0,j]=x+.013 if x<.9 else x-.013;self.assert_rejected(reseal(r))
 def test_R2_NA_infinity_and_count_faults(self):
  for name,value in [('all_gt_scene_recall',math.nan),('scene_coverage',math.inf),('tp_count',1.5),('q_px',-1),('finite_q_rate',.5)]:
   r=copy.deepcopy(self.r);r['values'][0,1,0,METRICS.index(name)]=value;self.assert_rejected(reseal(r),False)
  ai=next(i for i,a in enumerate(ARMS) if a['name']=='blue_cast_115__recalibrated')
  r=copy.deepcopy(self.r);r['values'][ai,1,19,0]=0;self.assert_rejected(reseal(r),False)
  ai=next(i for i,a in enumerate(ARMS) if a['name']=='saturation_035__recalibrated')
  r=copy.deepcopy(self.r);r['values'][ai,1,18,METRICS.index('normalized_center_radius')]=0;self.assert_rejected(reseal(r),False)
 def test_R2_geometry_source_corruption_and_exact_full_boundary(self):
  d=copy.deepcopy(self.data);d[0]['tp_pred'][0,2]=0
  with self.assertRaises(ValueError):validate_prepared(d,len(self.inputs.ids))
  from fractions import Fraction
  for w,h,q in [(10.,np.nextafter(10.,0.),6.280369834735101e-16),(123.456,123.455,.00035355339059496),(10.,10.,0),(10.,2.,math.inf)]:
   lo,hi,full=approved_projection(float(w),float(h),q)
   expected=math.isinf(q) or Fraction(q)**2>=(Fraction(w)-Fraction(h))**2/8
   self.assertEqual(full,expected)
 def test_R1_prepare_budget_refusal_preserves_prior_checkpoint(self):
  with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
   out=Path(tmp);marker=out/'base.npz';marker.write_bytes(b'committed');h=sha(marker);clock=[0.];cpu=[0.]
   with self.assertRaises(BudgetStop):
    with Lifecycle(out,limits(max_cpu_seconds=1),{'test':'prepare'},monitor=False,clock=lambda:clock[0],cpu_clock=lambda:cpu[0],disk=lambda:2**30,memory=lambda:2**30):
     cpu[0]=2.;clock[0]=2.;prepare_all(self.inputs,out/'prepared','test-source')
   self.assertEqual(sha(marker),h);self.assertFalse(list((out/'prepared').glob('*.npz')))
 def test_R1_prepare_disk_refusal_before_new_npz(self):
  with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
   out=Path(tmp);free=[2**30]
   with self.assertRaises(BudgetStop):
    with Lifecycle(out,limits(),{'test':'disk-prep'},monitor=False,disk=lambda:free[0],memory=lambda:2**30) as life:
     free[0]=1024;life.last_check=-math.inf;prepare_all(self.inputs,out/'prepared','test-source')
   self.assertFalse(list((out/'prepared').glob('*.npz')))
 def test_R1_scratch_preallocation_refuses_and_no_draw_changed(self):
  with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
   out=Path(tmp);marker=out/'draw.npz';marker.write_bytes(b'original');old=sha(marker)
   with self.assertRaises(BudgetStop):
    with Lifecycle(out,limits(),{'test':'scratch'},monitor=False,disk=lambda:4096,memory=lambda:2**30):scratch_allocate(out,5000,self.r)
   self.assertEqual(sha(marker),old);self.assertFalse(list(out.glob('verify-*.npy')))
 def test_R1_verification_budget_and_clean_resume_are_cumulative(self):
  with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
   out=Path(tmp);cpu=[0.];wall=[0.];ident={'test':'resume'};opts=limits(max_cpu_seconds=10,max_wall_seconds=10)
   with self.assertRaises(BudgetStop):
    with Lifecycle(out,opts,ident,False,lambda:wall[0],lambda:cpu[0],lambda:2**30,lambda:2**30):
     cpu[0]=11.;wall[0]=2.;tick('verification')
   # New process does not get a fresh10-second allowance.
   with self.assertRaises(BudgetStop):Lifecycle(out,opts,ident,False,lambda:0.,lambda:0.,lambda:2**30,lambda:2**30)
 def test_R1_verification_wall_fault_and_unclean_resume_refused(self):
  with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
   out=Path(tmp);clock=[0.]
   with self.assertRaises(BudgetStop):
    with Lifecycle(out,limits(max_wall_seconds=1),{'test':'wall'},False,lambda:clock[0],lambda:0.,lambda:2**30,lambda:2**30):
     clock[0]=2;verify_one(self.r,self.inputs,self.data,-1,True)
   j=json.loads((out/'LIFECYCLE.json').read_text());j['status']='ACTIVE';(out/'LIFECYCLE.json').write_text(json.dumps(j))
   with self.assertRaises(BudgetStop):Lifecycle(out,limits(max_wall_seconds=1),{'test':'wall'},monitor=False)
 def test_R1_exact_scratch_capacity_release_only_after_receipt(self):
  p=shape_plan(11738);self.assertEqual(p['per_cell_scratch_bytes'],1178320000+5120)
  self.assertEqual(p['maximum_live_scratch_cells'],1);self.assertEqual(p['all12_retained_scratch_bytes'],0)
  with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
   out=Path(tmp);st,paths=scratch_allocate(out,2,self.r)
   with self.assertRaises(ValueError):scratch_release(st,paths,out/'absent.json')
   write_json(out/'INTERVALS.json',{'synthetic':'committed'});receipt_sha=sha(out/'INTERVALS.json')
   hashes=scratch_release(st,paths,out/'INTERVALS.json');self.assertEqual(len(hashes),5)
   self.assertFalse(any(p.exists() for p in paths));self.assertEqual(sha(out/'INTERVALS.json'),receipt_sha)
 def test_R1_full_preflight_includes_future_verification(self):
  with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
   with Lifecycle(Path(tmp),limits(),{'test':'preflight'},monitor=False,disk=lambda:2**30,memory=lambda:2**30) as b:
    with self.assertRaises(BudgetStop):b.preflight(dict(cpu_seconds=1e5,wall_seconds=1,peak_additional_bytes=1))
    with self.assertRaises(BudgetStop):b.preflight(dict(cpu_seconds=1,wall_seconds=1,peak_additional_bytes=2**31))
 def test_R1_long_step_watchdog_requests_stop_no_signal(self):
  with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
   opts=limits(max_wall_seconds=.2)
   with self.assertRaises(BudgetStop):
    with Lifecycle(Path(tmp),opts,{'test':'watchdog'},monitor=True) as b:
     time.sleep(.9);self.assertTrue(b.stop_file.exists());tick('verification')
   self.assertTrue(list(Path(tmp).glob('HEALTH-*.jsonl')))

 def test_tail_MC_zero_envelope_is_unresolved_no_block_selection(self):
  from intervals import interval,contrast_numerical_status
  # Threshold chosen from a synthetic vector to construct a numerical fault,
  # never used to tune real results or frozen B/alpha.
  x=np.arange(5000,dtype=float)-5.
  a=interval(x,.996875);d=contrast_numerical_status(a)
  self.assertEqual(a['tail_MC'][0]['ranks'],[3,15]);self.assertEqual(a['tail_MC'][1]['ranks'],[4986,4998])
  self.assertAlmostEqual(a['tail_MC'][0]['expected_tail_draws'],7.8125)
  self.assertGreater(a['percentile'][0],0)
  self.assertEqual(d['status'],'numerically_unresolved');self.assertFalse(d['sign_claim_allowed']);self.assertEqual(len(d['five_block_signs']),5)
  b=interval(np.ones(5000),.996875);e=contrast_numerical_status(b)
  self.assertEqual(e['status'],'positive');self.assertTrue(e['sign_claim_allowed'])

if __name__=='__main__':unittest.main(verbosity=2)

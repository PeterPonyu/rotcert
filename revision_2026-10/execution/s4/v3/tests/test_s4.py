import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import *
from io_s4 import *
from inputs import InputSet,roles_from_ids
from prepare import match_condition
from statistics_core import ExactWeightedThreshold,multiplicities,ModelPipeline,projection
from intervals import interval,coverage_level
from verify_s4 import ref_fit,verify_one,ref_interval,check_interval
from rotcert_contracts.exact import hcp_threshold,pooled_threshold
from rotcert_contracts.geometry import critical_radius
from fractions import Fraction
from fixture import generate
import tempfile,unittest,shutil,subprocess


class S4Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  (ROOT/'fixtures').mkdir(exist_ok=True)
  cls.fixture_dir=tempfile.TemporaryDirectory(dir=ROOT/'fixtures',prefix='test-s4-fixture-')
  cls.addClassCleanup(cls.fixture_dir.cleanup)
  cls.fixture_root=Path(cls.fixture_dir.name)/'synthetic-v1'
  cls.inputs=InputSet(generate(cls.fixture_root),True)
  cls.data=[match_condition(cls.inputs,MODELS[0],c,.5) for c in CONDITIONS]
 def test_script_compilation_and_imports(self):
  for p in list(ROOT.glob('*.py'))+list((ROOT/'tests').glob('*.py')):compile(p.read_text(),str(p),'exec')
  import run_s4,verify_s4
  self.assertTrue(callable(run_s4.main) and callable(verify_s4.verify_run))
 def test_fixed_shared_rng_and_roles(self):
  roles=self.inputs.roles
  for d in [-1,0,1,4999]:
   w=multiplicities(roles,d)
   for role in range(2):
    self.assertEqual(w[role].sum(),np.sum(roles==role));self.assertTrue(np.all(w[role,roles!=role]==0))
   from verify_s4 import expected_weights
   np.testing.assert_array_equal(w,expected_weights(self.inputs,d))
  self.assertFalse(np.array_equal(multiplicities(roles,0),multiplicities(roles,1)))
 def test_exact_weighted_calibration_adversarial(self):
  rng=np.random.default_rng(202609284)
  for n in [0,8,9,10,19,49]:
   cnt=rng.integers(1,10,n);sources=np.repeat(np.arange(n),cnt);v=rng.integers(0,30,len(sources)).astype(float)
   mult=rng.integers(0,4,n);m=ExactWeightedThreshold(v,sources,n)
   groups=[]
   for i in range(n):groups.extend([v[sources==i]]*int(mult[i]))
   self.assertEqual(m.fit(mult,0),pooled_threshold(groups,'0.1'))
   self.assertEqual(m.fit(mult,1),hcp_threshold(groups,'0.1'))
  for n,expected in [(8,math.inf),(9,9.)]:
   m=ExactWeightedThreshold(np.arange(1,n+1,dtype=float),np.arange(n),n)
   self.assertEqual(m.fit(np.ones(n,dtype=np.int64),1),expected)
 def test_every_selected_draw_recalibrates(self):
  pipeline=ModelPipeline(self.data,self.inputs.roles)
  a=pipeline.run(multiplicities(self.inputs.roles,0));b=pipeline.run(multiplicities(self.inputs.roles,1))
  self.assertFalse(np.array_equal(a['q'],b['q']))
  for d,r in [(0,a),(1,b)]:
   r['multiplicity']=multiplicities(self.inputs.roles,d)
   verify_one(r,self.inputs,self.data,d,True)
 def test_selection_changes_and_all_source_conservation(self):
  base,shift=self.data[:2]
  self.assertFalse(np.array_equal(base['counts'][:,:,2],shift['counts'][:,:,2]))
  self.assertTrue(np.all(base['counts'][:,:,0]==shift['counts'][:,:,0]))
  pipeline=ModelPipeline(self.data,self.inputs.roles);w=multiplicities(self.inputs.roles,-1);r=pipeline.run(w)
  allr=r['all_source'][0,1]
  self.assertEqual(allr[0],200)
  self.assertGreater(allr[3],0) # zero TP sources kept
  self.assertGreater(allr[4],0) # GT positive yet zero TP
  self.assertLess(allr[1],allr[0]) # actual zero GT source kept
  self.assertEqual(allr[5],allr[6]+allr[8])
  self.assertEqual(allr[9],allr[6]+allr[7])
 def test_empty_class_no_drop_and_infinite_cost(self):
  r=ModelPipeline(self.data,self.inputs.roles).run(multiplicities(self.inputs.roles,-1))
  # blue-cast eval lacks class19; fixed class macro is NA, not nanmean.
  ai=next(i for i,a in enumerate(ARMS) if a['name']=='blue_cast_115__recalibrated')
  self.assertTrue(np.isnan(r['values'][ai,1,19,0]))
  self.assertTrue(np.isnan(r['macro'][ai,1,0]))
  # saturation cal lacks class18; q infinity retained, evaluation still defined.
  ci=CONDITIONS.index('saturation_035');self.assertTrue(np.isinf(r['q'][ci,:,18]).all())
  ai=next(i for i,a in enumerate(ARMS) if a['name']=='saturation_035__recalibrated')
  self.assertTrue(np.isinf(r['values'][ai,1,18,METRICS.index('normalized_center_radius')]))
  self.assertEqual(r['values'][ai,1,18,METRICS.index('finite_q_rate')],0)
 def test_sensitivity_rematches_without_inference_mutation(self):
  a=self.data[1];b=match_condition(self.inputs,MODELS[0],CONDITIONS[1],.7)
  np.testing.assert_array_equal(a['counts'][:,:,:2],b['counts'][:,:,:2])
  self.assertTrue((b['counts'][:,:,2]<=a['counts'][:,:,2]).all())
 def test_contract_critical_boundaries_and_full_cost(self):
  for w,h in [(10.,np.nextafter(10.,0.)),(123.456,123.455),(10.,2.),(10.,10.)]:
   q=critical_radius(w,h);lo,hi,full=projection(w,h,q)
   self.assertTrue(full);self.assertEqual(hi,math.pi/2)
   self.assertTrue(projection(w,h,math.inf)[2])
 def test_interval_family_and_all_5000_missing_rule(self):
  count=sum(coverage_level(tau,m,c)==.996875 for _ in MODELS for tau in TAUS for m in range(2) for c in CONTRASTS)
  self.assertEqual(count,16)
  rng=np.random.default_rng(71);x=rng.normal(size=5000)
  for level in [.95,.996875]:
   a=interval(x,level);check_interval(ref_interval(x,level),a)
   np.testing.assert_allclose(a['percentile'],np.quantile(x,[(1-level)/2,(1+level)/2],method='linear'),rtol=0,atol=2e-15)
   self.assertEqual(len(a['five_blocks1000']),5)
  x[42]=math.nan;self.assertEqual(interval(x)['status'],'UNAVAILABLE_REQUIRED_DRAW_NA')
  with self.assertRaises(ValueError):interval(x[:2])
  y=np.full(5000,math.inf);a=interval(y);self.assertEqual(a['percentile'],[math.inf,math.inf])
 def test_input_tamper_role_and_GT_rejected(self):
  # Copy only owned synthetic fixture. Never touch shared/real inputs.
  with tempfile.TemporaryDirectory(dir=self.fixture_root.parent) as tmp:
   dst=Path(tmp)/'copy';shutil.copytree(self.fixture_root,dst)
   idx=dst/'index.json';doc=json.loads(idx.read_text())
   p=dst/doc['runs'][0]['path']/'ground_truth.jsonl';p.write_text(p.read_text()+'\n')
   with self.assertRaises(ValueError):InputSet(idx,True)
   # Resealing an altered receipt does not bypass fixed GT/bundle invariant.
   work=p.parent;r=json.loads((work/'run.json').read_text());r['output_hashes']['ground_truth.jsonl']=sha(p)
   (work/'run.json').write_text(json.dumps(r));doc['runs'][0]['receipt_sha256']=sha(work/'run.json');idx.write_text(json.dumps(doc))
   with self.assertRaises(ValueError):InputSet(idx,True)
 def test_input_duplicate_and_synthetic_scope_guard(self):
  with self.assertRaises(ValueError):InputSet(self.fixture_root/'index.json',False)
  self.assertEqual(len(self.inputs.runs),24);self.assertEqual(len(self.inputs.ids),400)
  with self.assertRaises(ValueError):roles_from_ids(['x','x'])
 def test_lock_checkpoint_identity_and_no_overwrite(self):
  with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
   p=Path(tmp)
   with lock(p/'lock'):
    with self.assertRaises(RuntimeError):
     with lock(p/'lock'):pass
   save_npz(p/'base.npz',dict(a=np.array([math.inf,math.nan])),dict(draw_id=-1,cell='a'))
   with self.assertRaises(ValueError):load_npz(p/'base.npz',dict(draw_id=0))
   with self.assertRaises(ValueError):save_npz(p/'base.npz',{},dict(draw_id=-1))
   from run_s4 import existing_prefix
   self.assertEqual(existing_prefix(p,2),[-1])
   save_npz(p/'draw-0001.npz',{},dict(draw_id=1))
   with self.assertRaises(ValueError):existing_prefix(p,2)
 def test_input_guard_stat_mutation_and_portability(self):
  with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
   p=Path(tmp)/'x';p.write_text('abc');g=InputGuard({str(p):sha(p)});g.check(True)
   p.write_text('abc')
   with self.assertRaises(ValueError):g.check()
  i=json.loads((self.fixture_root/'index.json').read_text());self.assertFalse(Path(i['bundle']).is_absolute())
  self.assertTrue(all(not Path(r['path']).is_absolute() for r in i['runs']))

if __name__=='__main__':unittest.main(verbosity=2)

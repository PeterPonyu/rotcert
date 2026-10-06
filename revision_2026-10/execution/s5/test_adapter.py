import unittest
import numpy as np
from run_points import evaluated, fast_halfwidth, support
from rotcert_contracts.geometry import angle_projection
class AdapterRiskTests(unittest.TestCase):
 def test_zero_tp_and_misses_remain_in_all_gt(self):
  c={'scene_names':np.array(['a','b','c']),'class_names':np.array(['ship']),
     'tp_scene':np.array([0,0,2]),'tp_cls':np.zeros(3,int),'fn_scene':np.array([0,1,1]),'fn_cls':np.zeros(3,int),
     'fp_scene':np.array([1]),'fp_cls':np.zeros(1,int),'tp_pred':np.tile([0.,0.,8.,4.,0.],(3,1)),
     'tp_gt':np.tile([0.,0.,8.,4.,0.],(3,1))}
  out=evaluated(c,np.array([0.,2.,0.]),np.arange(3),0,1.)
  self.assertEqual(out['n_TP_positive_eval'],2);self.assertEqual(out['n_GT_positive_eval'],3)
  self.assertAlmostEqual(out['TP_scene_risk'],.25)
  self.assertAlmostEqual(out['GT_scene_FNR'],(1/3+1+0)/3)
  self.assertAlmostEqual(out['GT_scene_E2E_risk'],(2/3+1+0)/3)
  inf=evaluated(c,np.array([0.,2.,0.]),np.arange(3),0,float('inf'))
  self.assertAlmostEqual(inf['GT_scene_E2E_risk'],inf['GT_scene_FNR'])
  self.assertEqual(inf['infinite_fraction'],1)
  self.assertEqual(support(c,np.arange(3))[0]['GTs'],6)
 def test_geometry_vector_reference(self):
  w=np.array([4.,16.,64.,256.,4.,16.]);h=np.array([4.,15.99,12.,2.,2.,8.])
  for q in [0.,.001,.25,1.,8.,np.inf]:
   x=fast_halfwidth(w,h,q)
   y=np.array([angle_projection(float(a),float(b),q).half_width_upper for a,b in zip(w,h)])
   np.testing.assert_allclose(x,y,rtol=0,atol=1e-12)
if __name__=='__main__':unittest.main()

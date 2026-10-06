"""Exercise memo pre-eviction at the real384MiB limit and exact fallback."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,unittest,resource,json,time,weakref
from collections import OrderedDict
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from statistics_core import ModelPipeline,_centers_inside
from io_s4 import cap
class MemoTests(unittest.TestCase):
 def test_actual_capacity_churn_and_pre_eviction(self):
  cap();pipe=ModelPipeline.__new__(ModelPipeline);pipe.cache=OrderedDict();pipe.cache_container_reserve=1024**2;pipe.cache_bytes=pipe.cache_container_reserve;pipe.cache_limit=384*1024**2;pipe.cache_entry_overhead=2048;pipe.cache_peak_charged_bytes=pipe.cache_bytes;pipe.cache_hits=pipe.cache_misses=0
  # Shapes exceed any real class source support while remaining cheap to build.
  n=11738;payload=n*(8+9*np.dtype(np.longdouble).itemsize);charged=payload+2048
  owner=self;weak_entries=[]
  max_entries=(pipe.cache_limit-pipe.cache_container_reserve)//charged
  class Cell:
   tp_ids=det_ids=np.arange(n)
   def sufficient(self,q):
    owner.assertLessEqual(pipe.cache_bytes+charged,pipe.cache_limit)
    if int(q)>=max_entries:owner.assertIsNone(weak_entries[int(q)-max_entries]())
    summary=(np.full(n,int(q),dtype=np.int64),np.full((7,n),q,dtype=np.longdouble),np.full((2,n),q,dtype=np.longdouble))
    weak_entries.append(weakref.ref(summary[0]));return summary
  pipe.classes=[[Cell()]]
  for q in range(300):
   result=pipe.summaries(0,0,float(q));self.assertEqual(result[0][0],q);del result
  self.assertGreater(pipe.cache_misses,len(pipe.cache));self.assertLessEqual(pipe.cache_peak_charged_bytes,pipe.cache_limit)
  # Conservative overhead charge dominates actual Python object ownership.
  actual=sys.getsizeof(pipe.cache)
  for key,(value,size) in pipe.cache.items():
   actual+=sys.getsizeof(key)+sum(sys.getsizeof(x) for x in key)+sys.getsizeof(value)+sum(sys.getsizeof(a) for a in value)+sys.getsizeof(size)+sys.getsizeof((value,size))
  self.assertLessEqual(actual,pipe.cache_bytes)
  peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;vm=int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmPeak:')))*1024
  print(json.dumps(dict(scope='synthetic cache stress only no endpoint scientific computation',entries=len(pipe.cache),misses=pipe.cache_misses,charged_bytes=pipe.cache_bytes,measured_owned_bytes=actual,peak_rss_bytes=peak,vm_peak_bytes=vm,address_cap=4*1024**3)),flush=True)
  self.assertLess(vm,4*1024**3)
 def test_unsupported_platform_uses_exact_fraction(self):
  pred=np.array([[0.,0.,1,1,0],[0.,0.,1,1,0],[1e250,0,1,1,0]])
  truth=np.array([[3.,4.,1,1,0],[3.,np.nextafter(4.,np.inf),1,1,0],[0.,0.,1,1,0]])
  with patch('fast_geometry._platform_supported',return_value=False):np.testing.assert_array_equal(_centers_inside(pred,truth,5.),[True,False,False])
if __name__=='__main__':unittest.main(verbosity=2)

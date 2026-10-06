"""Explicit full-lifecycle projections measured from the same-input short pilot."""
from config import *
from io_s4 import require
from pathlib import Path
import math

STACK_KEYS=('values','macro','contrasts','macro_contrasts','all_source')
# Strict schema upper allowance, per endpoint, including two tails/five blocks.
# Runtime write checks remain mandatory if future serialization is larger.
INTERVAL_ENTRY_ALLOWANCE=16384
CONTROL_ALLOWANCE=64*1024**2

def shape_plan(n_sources,count=5000):
 shapes={'values':(11,2,20,33),'macro':(11,2,33),'contrasts':(10,2,20,33),'macro_contrasts':(10,2,33),'all_source':(11,2,16)}
 elements={k:math.prod(v) for k,v in shapes.items()}
 scratch=count*sum(elements.values())*8+len(shapes)*1024
 per_draw=(sum(elements.values())+6*2*20+6*2*20*7+2*n_sources+1)*8+32768
 endpoints=sum(elements.values())
 return dict(shapes=shapes,per_cell_scratch_bytes=scratch,maximum_live_scratch_cells=1,all12_retained_scratch_bytes=0,
  per_checkpoint_uncompressed_upper_bytes=per_draw,endpoint_count_per_cell=endpoints,
  full_intervals_upper_bytes=12*(endpoints*INTERVAL_ENTRY_ALLOWANCE+65536),
  full_checkpoints_upper_bytes=12*(count+1)*(per_draw+16384))

def future_capacity(n_sources,prepared_bytes,timings,verification,preparation_seconds,input_seconds,interval_benchmark):
 p=shape_plan(n_sources)
 means=[sum(x['fit_check_save_cpu_seconds'] for x in t['draws'] if x['draw']>=0)/2 for t in timings]
 # Non-scientific interval microbenchmark processes full-length scalar vectors.
 interval_cpu=interval_benchmark['max_cpu_per_endpoint']*p['endpoint_count_per_cell']*12
 light=verification['max_light_cpu_per_draw'];deep=verification['max_deep_cpu_per_draw']
 verify_cpu=light*12*5001+max(0.,deep-light)*12*5+interval_cpu
 compute_cpu=sum(means)*5000+sum(t['draws'][0]['fit_check_save_cpu_seconds'] for t in timings)
 total=(input_seconds+preparation_seconds+compute_cpu+verify_cpu)*1.5
 wall_factor=max(1.,verification['wall_to_cpu_ratio'])
 # Retained checkpoint allowance is conservative, not a compression promise.
 prepared_allowance=int(prepared_bytes*1.5)+CONTROL_ALLOWANCE
 peak=p['full_checkpoints_upper_bytes']+p['full_intervals_upper_bytes']+p['per_cell_scratch_bytes']+prepared_allowance+CONTROL_ALLOWANCE
 return dict(cpu_seconds=total,wall_seconds=total*wall_factor,peak_additional_bytes=peak,
  component_cpu_nominal=dict(input=input_seconds,preparation=preparation_seconds,compute=compute_cpu,verification=verify_cpu,interval_reduction=interval_cpu),
  shape_plan=p,prepared_allowance_bytes=prepared_allowance,safety_factor=1.5,
  scope='same actual input only; includes input/preparation/compute/full verification/scalar interval reduction; pilot projection remains uncertain',
  no_real_data_ETA_from_synthetic=True)

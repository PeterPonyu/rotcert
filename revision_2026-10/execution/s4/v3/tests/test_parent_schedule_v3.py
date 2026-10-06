"""Focused review-regression tests for pilot/full schedule terminal gates."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
from pathlib import Path
from datetime import datetime,timedelta,timezone
import sys,json,tempfile,unittest,time,copy
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_s4 as runner
from resources import Lifecycle,BudgetStop
class ScheduleTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.index=self.root/'index.json';self.index.write_text('{}');self.source='candidate-source'
  self.deadline=(datetime.now(timezone.utc)+timedelta(days=1)).isoformat()
  self.limit=dict(max_cpu_seconds=1000,max_wall_seconds=1000,max_output_bytes=10**8,minimum_free_disk_bytes=2*1024**3,minimum_available_memory_bytes=2*1024**3,deadline_utc=self.deadline)
  self.pilot_schedule=dict(self.limit,approve_pilot=True,source_sha256=self.source,index_sha256=runner.sha(self.index));self.ps=self.root/'PARENT-PILOT-SCHEDULE.json';self.write(self.ps,self.pilot_schedule)
  self.out=self.root/'pilot';self.out.mkdir();self.pilot=self.out/'PILOT.json';self.life=self.out/'LIFECYCLE.json';self.inputs=self.out/'INPUTS.json';self.ver=self.out/'VERIFIED.json'
  self.write(self.ver,dict(status='PASS_DECLARED_NUMERICAL_ENDPOINT_SCOPE'));self.write(self.inputs,dict(source_sha256=self.source,input_binding_sha256='binding'))
  self.receipt=dict(source_sha256=self.source,input_binding_sha256='binding',synthetic=False,status='VERIFIED',verification=dict(status='PASS_DECLARED_NUMERICAL_ENDPOINT_SCOPE',verified_sha256=runner.sha(self.ver)),total_cpu_seconds=10.,wall_seconds=12.,full_lifecycle_projection=dict(includes_complete_lifecycle=True,cpu_seconds=100.,wall_seconds=120.,peak_additional_bytes=0))
  self.write(self.pilot,self.receipt);self.ledger=dict(status='COMPLETE',identity=dict(source_sha256=self.source,index_sha256=runner.sha(self.index),mode='pilot',synthetic=False,schedule_sha256=runner.sha(self.ps)),cpu_seconds=10.5,wall_seconds=12.5);self.write(self.life,self.ledger)
  self.gate=dict(self.limit,approve_full=True,independent_review_passed=True,source_sha256=self.source,index_sha256=runner.sha(self.index),B=5000,real_pilot_receipt=str(self.pilot),real_pilot_sha256=runner.sha(self.pilot),input_binding_sha256='binding',real_pilot_lifecycle=str(self.life),real_pilot_lifecycle_sha256=runner.sha(self.life),real_pilot_inputs_sha256=runner.sha(self.inputs),real_pilot_schedule_sha256=runner.sha(self.ps))
  self.full=self.root/'PARENT-SCHEDULE.json';self.write(self.full,self.gate);self.patch=patch.object(runner,'ROOT',self.root);self.patch.start()
 def tearDown(self):self.patch.stop();self.tmp.cleanup()
 def write(self,p,d):p.write_text(json.dumps(d))
 def test_terminal_ledger_required_and_pinned(self):
  limits,gate,h=runner.schedule(self.source,self.index,'full');self.assertIn('deadline_unix_seconds',limits)
  for status in ['ACTIVE','STOPPED']:
   ledger=dict(self.ledger,status=status);self.write(self.life,ledger);g=dict(self.gate,real_pilot_lifecycle_sha256=runner.sha(self.life));self.write(self.full,g)
   with self.assertRaisesRegex(ValueError,'not complete'):runner.schedule(self.source,self.index,'full')
  self.write(self.full,self.gate)
  with self.assertRaisesRegex(ValueError,'lifecycle hash'):runner.schedule(self.source,self.index,'full')
 def test_source_index_binding_and_closeout_cost(self):
  ledger=copy.deepcopy(self.ledger);ledger['identity']['index_sha256']='wrong';self.write(self.life,ledger);g=dict(self.gate,real_pilot_lifecycle_sha256=runner.sha(self.life));self.write(self.full,g)
  with self.assertRaisesRegex(ValueError,'terminal identity'):runner.schedule(self.source,self.index,'full')
  self.write(self.life,self.ledger);g=dict(self.gate,max_cpu_seconds=100.);self.write(self.full,g)
  with self.assertRaisesRegex(ValueError,'closeout cost'):runner.schedule(self.source,self.index,'full')
  self.write(self.full,self.gate);self.write(self.inputs,dict(source_sha256='wrong',input_binding_sha256='binding'))
  with self.assertRaisesRegex(ValueError,'input receipt hash'):runner.schedule(self.source,self.index,'full')
 def test_closeout_reserve_and_already_consumed_budget_are_combined(self):
  ledger=dict(self.ledger,cpu_seconds=12.,wall_seconds=14.);self.write(self.life,ledger)
  g=dict(self.gate,max_cpu_seconds=105.,real_pilot_lifecycle_sha256=runner.sha(self.life));self.write(self.full,g)
  limits,permit,h=runner.schedule(self.source,self.index,'full')
  plan=permit['validated_full_lifecycle_projection'];self.assertEqual(plan['cpu_seconds'],103.);self.assertEqual(plan['wall_seconds'],123.)
  with Lifecycle(self.root/'cumulative',limits,dict(test=True),monitor=False,cpu_clock=lambda:3.,disk=lambda:10**12,memory=lambda:10**12) as life:
   with self.assertRaisesRegex(BudgetStop,'exceeds remaining schedule'):life.preflight(plan)
 def test_pilot_schedule_drift_and_deadline(self):
  limits,gate,h=runner.schedule(self.source,self.index,'pilot');self.assertIsNone(gate);runner.check_schedule('pilot',h)
  self.write(self.ps,dict(self.pilot_schedule,max_cpu_seconds=999.))
  with self.assertRaisesRegex(ValueError,'schedule changed'):runner.check_schedule('pilot',h)
  self.write(self.ps,dict(self.pilot_schedule,deadline_utc='2026-09-01T00:00:00+00:00'))
  with self.assertRaisesRegex(ValueError,'deadline expired'):runner.schedule(self.source,self.index,'pilot')
 def test_projection_cannot_overrun_absolute_deadline(self):
  limits={k:v for k,v in self.limit.items() if k!='deadline_utc'};limits['deadline_unix_seconds']=time.time()+10
  with Lifecycle(self.root/'projection-deadline',limits,dict(test=True),monitor=False,disk=lambda:10**12,memory=lambda:10**12) as life:
   with self.assertRaisesRegex(BudgetStop,'exceeds absolute deadline'):life.preflight(dict(cpu_seconds=1.,wall_seconds=20.,peak_additional_bytes=0))
 def test_lifecycle_absolute_deadline_cooperative_stop(self):
  limits={k:v for k,v in self.limit.items() if k!='deadline_utc'};limits['deadline_unix_seconds']=time.time()-1
  with self.assertRaisesRegex(BudgetStop,'absolute schedule deadline'):
   Lifecycle(self.root/'expired',limits,dict(test=True),monitor=False,disk=lambda:10**12,memory=lambda:10**12)
if __name__=='__main__':unittest.main(verbosity=2)

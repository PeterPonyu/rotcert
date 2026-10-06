#!/usr/bin/env python
"""S4 CPU consumer; measured actual-input pilot and reviewed full lifecycle gate."""
from config import *
from io_s4 import *
from inputs import InputSet
from prepare import prepare_all,load_model
from statistics_core import ModelPipeline,multiplicities
import argparse,time,traceback,shutil
from resources import Lifecycle,tick
from capacity import future_capacity


def existing_prefix(directory,count):
 directory=Path(directory);expected=['base']+[f'draw-{i:04d}' for i in range(count)]
 if not directory.exists():return []
 require(not list(directory.glob('*.tmp-*')),'Orphan temporary checkpoint')
 actual={p.stem for p in directory.glob('*.npz')}|{p.stem for p in directory.glob('*.json') if p.stem=='base' or p.stem.startswith('draw-')}
 require(actual<=set(expected),'Unplanned checkpoint')
 gap=False;found=[]
 for i,name in enumerate(expected):
  a=directory/(name+'.npz');b=directory/(name+'.json')
  require(a.exists()==b.exists(),'Orphan checkpoint '+name)
  if a.exists():require(not gap,'Checkpoint hole '+name);found.append(i-1)
  else:gap=True
 return found


def validate_saved(result,roles,draw):
 require(np.array_equal(result['multiplicity'],multiplicities(roles,draw)),'Scientific RNG drift')
 require(np.array_equal(result['values'].mean(axis=2),result['macro'],equal_nan=True),'Strict20class macro violated')
 with np.errstate(invalid='ignore'):
  for i,c in enumerate(CONTRASTS):
   require(np.array_equal(result['values'][c['left']]-result['values'][c['right']],result['contrasts'][i],equal_nan=True),'Per-class contrast differs')
   require(np.array_equal(result['macro'][c['left']]-result['macro'][c['right']],result['macro_contrasts'][i],equal_nan=True),'Macro contrast differs')


def checkpoint_meta(source,input_hash,model,tau,draw):
 return dict(schema='S4-draw-v1',source_sha256=source,input_binding_sha256=input_hash,model=model,iou=tau,draw_id=draw,
             RNG='PCG64 SeedSequence([20260928,4,0,draw,role]); one shared source draw across all cells')



def schedule_deadline(g):
 from datetime import datetime
 value=datetime.fromisoformat(g['deadline_utc'])
 require(value.tzinfo is not None,'Schedule deadline must include timezone')
 deadline=value.timestamp();require(math.isfinite(deadline) and time.time()<deadline,'Schedule deadline expired')
 return deadline


def check_schedule(mode,expected_hash):
 p=ROOT/('PARENT-PILOT-SCHEDULE.json' if mode=='pilot' else 'PARENT-SCHEDULE.json')
 require(sha(p)==expected_hash,'Parent schedule changed: safe checkpoint stop')
 schedule_deadline(json.loads(p.read_text()))
 require(sha(p)==expected_hash,'Schedule changed during deadline check')


def schedule(source,index,mode):
 # Resolve a budget before archive reads/preparation; same receipt reused on resume.
 if mode=='pilot':
  p=ROOT/'PARENT-PILOT-SCHEDULE.json'
  require(p.exists(),'Measured candidate pilot requires parent resource schedule')
  permit_hash=sha(p);g=json.loads(p.read_text());require(sha(p)==permit_hash,'Pilot schedule changed during read')
  deadline=schedule_deadline(g)
  require(g.get('approve_pilot') is True and g.get('source_sha256')==source and g.get('index_sha256')==sha(index),'Pilot approval/source/index binding')
  keys=['max_cpu_seconds','max_wall_seconds','max_output_bytes','minimum_free_disk_bytes','minimum_available_memory_bytes']
  for k in keys:require(type(g.get(k)) in (int,float) and math.isfinite(g[k]) and g[k]>0,'Pilot finite limit '+k)
  require(g['minimum_free_disk_bytes']>=2*1024**3 and g['minimum_available_memory_bytes']>=2*1024**3,'Pilot reserve below2GiB')
  return dict({k:g[k] for k in keys},deadline_unix_seconds=deadline),None,permit_hash
 p=ROOT/'PARENT-SCHEDULE.json';require(p.exists(),'Formal B5000 blocked: parent launch gate absent')
 permit_hash=sha(p);g=json.loads(p.read_text());require(sha(p)==permit_hash,'Full schedule changed during read')
 deadline=schedule_deadline(g)
 require(g.get('approve_full') is True and g.get('independent_review_passed') is True,'Formal launch not reviewed/authorized')
 require(g.get('source_sha256')==source and g.get('B')==5000,'Parent gate source/B mismatch')
 require(g.get('index_sha256')==sha(index),'Schedule must pin exact input index before input acceptance')
 for k in ['max_cpu_seconds','max_wall_seconds','max_output_bytes','minimum_free_disk_bytes','minimum_available_memory_bytes']:
  require(type(g.get(k)) in (float,int) and math.isfinite(g[k]) and g[k]>0,'Finite cumulative lifecycle limit '+k)
 require(g['minimum_free_disk_bytes']>=2*1024**3 and g['minimum_available_memory_bytes']>=2*1024**3,'Reserve below2GiB')
 pilot=Path(g['real_pilot_receipt']);require(sha(pilot)==g.get('real_pilot_sha256'),'Real pilot receipt hash')
 pr=json.loads(pilot.read_text())
 require(pr['source_sha256']==source and pr['input_binding_sha256']==g.get('input_binding_sha256') and pr['synthetic'] is False and pr['status']=='VERIFIED','Real input pilot must be measured and verified')
 require(pr['verification']['status']=='PASS_DECLARED_NUMERICAL_ENDPOINT_SCOPE','Real pilot verification scope')
 # PILOT is published before context-manager closeout. Only the immutable final
 # lifecycle receipt proves that verification AND monitor/accounting finished.
 lp=Path(g['real_pilot_lifecycle']);require(lp==pilot.parent/'LIFECYCLE.json','Pilot lifecycle path')
 require(sha(lp)==g['real_pilot_lifecycle_sha256'],'Final pilot lifecycle hash')
 lifecycle=json.loads(lp.read_text());require(lifecycle['status']=='COMPLETE','Pilot lifecycle not complete')
 expected_identity=dict(source_sha256=source,index_sha256=sha(index),mode='pilot',synthetic=False,schedule_sha256=g['real_pilot_schedule_sha256'])
 require(lifecycle['identity']==expected_identity,'Pilot terminal identity differs')
 require(sha(ROOT/'PARENT-PILOT-SCHEDULE.json')==g['real_pilot_schedule_sha256'],'Pilot schedule hash differs')
 require(lifecycle['cpu_seconds']>=pr['total_cpu_seconds'] and lifecycle['wall_seconds']>=pr['wall_seconds'],'Pilot final accounting regressed')
 binding=pilot.parent/'INPUTS.json';require(sha(binding)==g['real_pilot_inputs_sha256'],'Pilot input receipt hash')
 pd=json.loads(binding.read_text());require(pd['source_sha256']==source and pd['input_binding_sha256']==g['input_binding_sha256'],'Terminal pilot input identity')
 require(sha(pilot.parent/'VERIFIED.json')==pr['verification']['verified_sha256'],'Pilot verified receipt changed')
 residual_cpu=max(0.,lifecycle['cpu_seconds']-pr['total_cpu_seconds'])
 residual_wall=max(0.,lifecycle['wall_seconds']-pr['wall_seconds'])
 plan=pr['full_lifecycle_projection']
 require(plan.get('includes_complete_lifecycle') is True,'Legacy compute-only budget is not accepted')
 require(g['max_cpu_seconds']>=plan['cpu_seconds']+1.5*residual_cpu and g['max_wall_seconds']>=plan['wall_seconds']+1.5*residual_wall,'Schedule omits final pilot closeout cost')
 corrected_plan=dict(plan)
 corrected_plan['cpu_seconds']+=1.5*residual_cpu;corrected_plan['wall_seconds']+=1.5*residual_wall
 corrected_plan['final_pilot_closeout_reserve']=dict(cpu_seconds=1.5*residual_cpu,wall_seconds=1.5*residual_wall)
 # In-memory derived permit only; original hash-pinned schedule stays immutable.
 permit=dict(g,validated_full_lifecycle_projection=corrected_plan)
 return dict({k:g[k] for k in ['max_cpu_seconds','max_wall_seconds','max_output_bytes','minimum_free_disk_bytes','minimum_available_memory_bytes']},deadline_unix_seconds=deadline),permit,permit_hash


def run(index,mode,out,synthetic=False):
 out=owned(out);cap();require(mode in ['pilot','full'],'mode')
 require(not (mode=='full' and synthetic),'Synthetic fixture cannot run formal B5000')
 source=sha(ROOT/'FROZEN.json');limits,permit,permit_hash=schedule(source,index,mode)
 count=2 if mode=='pilot' else B
 identity=dict(source_sha256=source,index_sha256=sha(index),mode=mode,synthetic=synthetic,schedule_sha256=permit_hash)
 with lock(ROOT/'ownership.lock'),Lifecycle(out,limits,identity) as life:
  source_checked=verify();require(source_checked==source,'Freeze changed before launch');check_schedule(mode,permit_hash)
  if permit:
   life.preflight(permit['validated_full_lifecycle_projection'])
  tick('input_acceptance');input_start=time.process_time();inputs=InputSet(index,synthetic=synthetic);input_cpu=time.process_time()-input_start
  if permit:require(inputs.binding_hash==permit['input_binding_sha256'],'Parent input binding differs')
  out.mkdir(parents=True,exist_ok=True);require(not (out/'VERIFIED.json').exists(),'Already verified: no duplicate launch')
  binding=out/'INPUTS.json';doc=dict(source_sha256=source,binding=inputs.binding,input_binding_sha256=inputs.binding_hash,files=inputs.files)
  if binding.exists():require(json.loads(binding.read_text())==doc,'Output bound to different input')
  else:write_json(binding,doc)
  started=time.monotonic();cpu=time.process_time();prepared=out/'prepared';prep_start=time.process_time();tick('preparation');prepare_all(inputs,prepared,source);prep_cpu=time.process_time()-prep_start
  prepared_guard=InputGuard({str(p):sha(p) for p in prepared.glob('*') if p.is_file()})
  timings=[];last_health=0.;setup_cpu_total=0.
  # Cell-major execution replays exact identical draw keys across all12
  # model/tau analyses; no model-specific sampling. One cell cache at a time.
  for mi,model in enumerate(MODELS):
   for tau in TAUS:
    directory=out/model/f'iou{tau:.1f}';prefix=existing_prefix(directory,count);directory.mkdir(parents=True,exist_ok=True)
    tick('fitting');setup_begin=time.process_time();data=load_model(inputs,prepared,source,model,tau);pipeline=ModelPipeline(data,inputs.roles);setup_cpu_total+=time.process_time()-setup_begin
    draw_cost=[]
    for draw in [-1]+list(range(count)):
     iteration_start=time.process_time()
     tick('fitting');inputs.guard.check();prepared_guard.check()
     check_schedule(mode,permit_hash)
     path=directory/('base.npz' if draw<0 else f'draw-{draw:04d}.npz');meta=checkpoint_meta(source,inputs.binding_hash,model,tau,draw)
     dt=time.process_time()
     if draw in prefix:result=load_npz(path,meta)
     else:
      mult=multiplicities(inputs.roles,draw);result=pipeline.run(mult);result['multiplicity']=mult
      result['pipeline_cpu_seconds']=np.asarray(time.process_time()-dt)
      validate_saved(result,inputs.roles,draw);inputs.guard.check();save_npz(path,result,meta)
     validate_saved(result,inputs.roles,draw)
     if draw>=0 and draw%25==0:verify()
     if draw>=0 and draw%100==0:prepared_guard.check(full=True)
     if time.monotonic()-last_health>=30 or draw==count-1:
      write_json(out/'state.json',dict(status='RUNNING',model=model,iou=tau,draw=draw,target_B=count,source_sha256=source,cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-started,free_disk_bytes=shutil.disk_usage(ROOT).free,**health()),replace=True);last_health=time.monotonic()
     draw_cost.append(dict(draw=draw,pipeline_cpu_seconds=float(result['pipeline_cpu_seconds']),fit_check_save_cpu_seconds=time.process_time()-iteration_start,bytes=path.stat().st_size,reused=draw in prefix,timing_scope='entire iteration including stat/source/resource/checkpoint checks'))
    timings.append(dict(model=model,iou=tau,draws=draw_cost))
    print(json.dumps(dict(model=model,iou=tau,status='CELL_COMPLETE',pipeline_cpu_seconds=sum(r['pipeline_cpu_seconds'] for r in draw_cost))),flush=True)
    del pipeline,data
  inputs.guard.check(full=True);prepared_guard.check(full=True);verify()
  projected=sum(sum(r['fit_check_save_cpu_seconds'] for r in t['draws'] if r['draw']>=0)/count for t in timings)*5000
  report=dict(status='COMPUTED_AWAITING_VERIFICATION',mode=mode,synthetic=synthetic,source_sha256=source,input_binding_sha256=inputs.binding_hash,
              bootstrap_draws_per_cell=count,full_pipeline_fits=len(timings)*(count+1),total_cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-started,
              projected_full_cpu_seconds_safety1p5=projected*1.5,projection_scope='this input size only; synthetic runtime is NOT a real-data/GPU ETA; excludes full independent verification and initial archive hashing/preparation',
              timings=timings,metrics=METRICS,all_source_metrics=ALL_SOURCE_METRICS,arms=ARMS,contrasts=CONTRASTS,**health())
  if not (out/'COMPLETE.json').exists():write_json(out/'COMPLETE.json',report)
  check_schedule(mode,permit_hash)
  from verify_s4 import verify_run
  verified=verify_run(inputs,out,source,count,deep=True)
  report.update(status='VERIFIED',verification=verified)
  if mode=='pilot':
   prepared_bytes=sum(p.stat().st_size for p in prepared.iterdir() if p.is_file())
   projection=future_capacity(len(inputs.ids),prepared_bytes,timings,verified['cost'],prep_cpu,input_cpu,verified['interval_capacity_benchmark'])
   modeled_pilot_cpu=input_cpu+prep_cpu+sum(sum(x['fit_check_save_cpu_seconds'] for x in t['draws']) for t in timings)+verified['cost']['cpu_seconds']
   overhead=max(0.,life.counters()[0]-modeled_pilot_cpu)
   projection['cpu_seconds']+=1.5*overhead;projection['wall_seconds']+=1.5*overhead*max(1.,life.counters()[1]/max(life.counters()[0],1e-9))
   projection['component_cpu_nominal']['measured_misc_lifecycle_overhead']=overhead
   projection['measured_cell_setup_cpu_seconds_in_misc']=setup_cpu_total
   projection['includes_complete_lifecycle']=True
   report['full_lifecycle_projection']=projection
  life_cpu,life_wall=life.counters()
  report.update(**health())
  report.update(total_cpu_seconds=life_cpu,wall_seconds=life_wall,resource_scope='cumulative complete lifecycle including prior clean stopped attempts; input,prep,fits,verification,interval benchmark',projection_scope='full_lifecycle_projection supersedes legacy compute-only number; synthetic numbers are not real-data ETA')
  check_schedule(mode,permit_hash)
  write_json(out/('PILOT.json' if mode=='pilot' else 'FULL.json'),report)
  write_json(out/'state.json',dict(status='VERIFIED',source_sha256=source,**health()),replace=True)
  return report


def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','fixture','pilot','full']);p.add_argument('--index');p.add_argument('--out');p.add_argument('--synthetic',action='store_true');a=p.parse_args()
 if a.command=='freeze':
  with lock(ROOT/'ownership.lock'):print(freeze())
 elif a.command=='fixture':
  from fixture import generate
  with lock(ROOT/'ownership.lock'):print(generate())
 else:
  require(a.index is not None and a.out is not None,'--index and --out required');r=run(a.index,a.command,a.out,a.synthetic)
  print(json.dumps({k:r[k] for k in ['status','mode','synthetic','total_cpu_seconds','wall_seconds','peak_rss_bytes']}))

if __name__=='__main__':
 try:main()
 except Exception as e:
  write_json(ROOT/'errors'/f'{os.getpid()}-{time.time_ns()}.json',dict(error=repr(e),traceback=traceback.format_exc(),utc=now()))
  raise

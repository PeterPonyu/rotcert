#!/usr/bin/env python
"""Publish measured pilot resource forecast without producing scientific CIs."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import json
import math
from pathlib import Path
from s3_io import ROOT,verify,sha,write_json,now,lock


def main():
    frozen=verify();pilot=json.loads((ROOT/'pilot/PILOT.json').read_text())
    verified=json.loads((ROOT/'pilot/VERIFIED.json').read_text())
    if verified['frozen_sha256']!=frozen or verified['status']!='PASS':raise RuntimeError('pilot not verified')
    per=[]
    for c in pilot['cells']:
        rows=[r for r in c['records'] if r['draw']>=0]
        if len(rows)!=2 or any(r['reused_checkpoint'] for r in rows):raise RuntimeError('ETA requires actual two draws')
        avg=sum(r['fit_check_save_cpu_seconds'] for r in rows)/2
        pipeline=sum(r['cpu_seconds'] for r in rows)/2
        compressed=sum(r['compressed_bytes'] for r in rows)/2
        once=max(0,c['total_cpu_seconds']-sum(r['fit_check_save_cpu_seconds'] for r in c['records']))
        base=next(r['fit_check_save_cpu_seconds'] for r in c['records'] if r['draw']==-1)
        per.append(dict(cell=c['cell'],cell_index=c['cell_index'],pipeline_cpu_seconds_per_draw=pipeline,
              fit_check_save_cpu_seconds_per_draw=avg,one_time_prepare_cpu_seconds=once,base_fit_check_save_cpu_seconds=base,
              full_cell_cpu_seconds=once+base+5000*avg,compressed_bytes_per_draw=compressed,
              peak_rss_bytes=c['rss_peak_bytes'],pilot_full_pipeline_fits=3,
              tuned_arms_available=[r['tuned_alpha_available'] for r in c['records']]))
    total=sum(c['full_cell_cpu_seconds'] for c in per)
    makespans={};loads={}
    # Frozen cell-order list scheduling approximates worker CPU loads; no claim of
    # empirical six-way speedup, disk contention or whole-machine quietness.
    for workers in range(1,7):
        slots=[0.]*workers
        for c in per:
            j=min(range(workers),key=lambda j:slots[j]);slots[j]+=c['full_cell_cpu_seconds']
        makespans[str(workers)]=max(slots)*1.5;loads[str(workers)]=slots
    projected=sum(c['compressed_bytes_per_draw'] for c in per)*5000
    gate=dict(approve_full=False,independent_review_passed=False,frozen_sha256=frozen,B=5000,cells='all16',max_workers=6,
              worker_address_space_bytes=4*1024**3,max_cpu_seconds=math.ceil(total*1.5),
              max_wall_seconds=math.ceil(makespans['6']),launch_minimum_free_disk_bytes=100*1024**3,minimum_free_disk_bytes=15*1024**3)
    report=dict(status='PILOT_VERIFIED_AWAITING_INDEPENDENT_LAUNCH_REVIEW',utc=now(),root=str(ROOT),frozen_sha256=frozen,
           v1_frozen_sha256=json.loads((ROOT/'FROZEN.json').read_text())['v1_frozen_sha256'],full_launched=False,
           measured_worker_cpu_seconds=pilot['total_worker_cpu_seconds'],measured_scheduler_wall_seconds=pilot['current_invocation_wall_seconds'],
           measured_peak_worker_rss_bytes=max(c['peak_rss_bytes'] for c in per),measured_workers=1,
           projected_full_cpu_seconds=total,projected_full_cpu_seconds_safety1p5=total*1.5,
           projected_parallel_wall_seconds_safety1p5=makespans,projected_worker_CPU_loads_seconds=loads,
           projected_compressed_draw_bytes=projected,projected_compressed_draw_bytes_safety1p5=projected*1.5,
           no_six_worker_stress_test=True,forecast_limit='Two-draw/cell single-worker extrapolation including fit, RNG/support check and compressed write. Full verifier CPU and system contention add time;1.5 safety is planning margin, not measured speedup or guaranteed ETA.',
           conditions=16,DIOR_cells=14,base_fits=16,bootstrap_fits=32,arms=22,per_cell=per,
           exact_RNG_replay='all48 pilot checkpoints',v1_nongeometry_differences=verified['v1_pilot_comparison'],
           test_log='tests-pre-freeze-final.log',test_count=16,
           required_parent_gate=gate,parent_gate_created_by_worker=False,
           full_launch_command=f'PYTHONDONTWRITEBYTECODE=1 {__import__("sys").executable} -B {ROOT}/run_s3.py full --workers 6',
           full_verification='automatic after all children exit; manually verify_s3.py full under scheduler lock if needed',
           independent_reviewer='Hume; E/review/s3; launch decision remains parent-owned',
           model='requested gpt-6-astra; actual runtime verified by parent')
    write_json(ROOT/'PARENT-READINESS.json',report,immutable=True)
    write_json(ROOT/'PARENT-SCHEDULE.example.json',gate,immutable=True)
    print(json.dumps({k:report[k] for k in ['frozen_sha256','measured_worker_cpu_seconds','measured_scheduler_wall_seconds','measured_peak_worker_rss_bytes','projected_full_cpu_seconds_safety1p5','projected_parallel_wall_seconds_safety1p5','projected_compressed_draw_bytes_safety1p5']}))

if __name__=='__main__':
    with lock(ROOT/'locks/scheduler.lock'):main()

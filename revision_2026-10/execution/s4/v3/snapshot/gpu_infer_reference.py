#!/usr/bin/env python3
"""Operator-only inference. Requires a frozen bundle, staged inputs and explicit GPU lease.
No SSH, installation, training, queue mutations, or model imports at module load.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

from supplement import (HERE, CLASSES, checked, sha, read_json, write_json, write_rows,
                        require, load_bundle, load_prepared, new_output, plain_config, utc, json_lines)


def run_readonly(command):
    return subprocess.check_output(command, text=True, timeout=30, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))

def identity():
    return {'hostname':socket.gethostname(), 'machine_id':Path('/etc/machine-id').read_text().strip(),
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}

def validate_lease(lease, bundle_path, output, now=None, host=None):
    require(lease.get('schema')=='rotcert-gpu-lease-v1', 'Wrong lease schema')
    require(bool(lease.get('approved_by')) and bool(lease.get('allocation_reference')), 'Leader allocation evidence required')
    require(lease.get('identity')==(identity() if host is None else host), 'Lease host/boot identity changed')
    require(lease.get('bundle_sha256')==sha(Path(bundle_path)/'bundle.json'), 'Lease bound to wrong bundle')
    require(lease.get('exclusive_outside_all_existing_queues') is True, 'Leader must reserve GPU outside every existing queue')
    require(lease.get('queue_inventory_reviewed') and lease.get('reviewed_utc'), 'Complete queue inventory evidence required')
    deadline=datetime.fromisoformat(lease['expires_utc'])
    require(deadline.tzinfo is not None, 'Lease expiry needs timezone')
    current=now or datetime.now(timezone.utc)
    require(deadline > current, 'Lease expired')
    require(datetime.fromisoformat(lease['valid_from_utc'])<=current,'Lease not yet valid')
    gpuid=lease['gpu_uuid']; require(gpuid.startswith('GPU-') and all(x.isalnum() or x=='-' for x in gpuid), 'Use full GPU UUID')
    require(gpuid not in lease.get('gpu_uuids_reserved_by_other_queues',[]), 'GPU reserved by another queue')
    root=Path(lease['output_root']).resolve()
    require(Path(output).resolve().is_relative_to(root) and Path(output).resolve()!=root, 'Output outside lease root')
    return gpuid

def gpu_snapshot(gpuid, check_idle=True, allowed_pids=()):
    lines=run_readonly(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu,name','--format=csv,noheader,nounits'])
    all_gpu=list(csv.reader(lines.splitlines())); matches=[r for r in all_gpu if len(r)==5 and r[1].strip()==gpuid]
    require(len(matches)==1, 'GPU UUID absent or ambiguous')
    r=[v.strip() for v in matches[0]]
    processes=run_readonly(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader,nounits'])
    procs=[]
    for row in csv.reader(processes.splitlines()):
        if row and row[0].strip()==gpuid:
            require(len(row)==3 and row[1].strip().isdigit(), 'Unparseable GPU process result')
            procs.append({'pid':int(row[1]),'name':row[2].strip()})
    require(not [p for p in procs if p['pid'] not in allowed_pids], 'Another GPU process is active; do not disturb it')
    if check_idle:
        require(float(r[2])<=512 and float(r[3])<=5, 'GPU not idle; do not launch')
    return {'index':r[0],'uuid':r[1],'memory_used_mib':float(r[2]),'utilization_pct':float(r[3]),'name':r[4],'processes':procs,'utc':utc()}

def build_config(original, work, checkpoint, image_manifest):
    import copy
    cfg=copy.deepcopy(original)
    pipeline=cfg['test_dataloader']['dataset']['pipeline']
    # Annotation-only transforms have no role in frozen image inference; canonical GT stays external.
    pipeline=[x for x in pipeline if x['type'].split('.')[-1] not in ['LoadAnnotations','ConvertBoxType']]
    types=[x['type'].split('.')[-1] for x in pipeline]
    require(types==['LoadImageFromFile','Resize','PackDetInputs'], 'Unexpected preprocessing; review required')
    cfg['test_dataloader']={'batch_size':1,'num_workers':0,'persistent_workers':False,'drop_last':False,
     'sampler':{'type':'DefaultSampler','shuffle':False},
     'dataset':{'type':'RotCertSupplementImages','manifest_path':str(image_manifest),
                'metainfo':{'classes':CLASSES},'test_mode':True,'pipeline':pipeline}}
    cfg.update(work_dir=str(work/'framework'),load_from=str(checkpoint),resume=False,
               train_dataloader=None,train_cfg=None,optim_wrapper=None,param_scheduler=None,
               val_dataloader=None,val_cfg=None,val_evaluator=None,auto_scale_lr=None,
               test_cfg={'type':'TestLoop'},test_evaluator={'type':'DumpResults','out_file_path':str(work/'predictions.pkl')},
               custom_hooks=[],default_hooks={'timer':{'type':'IterTimerHook'},'logger':{'type':'LoggerHook','interval':50}},
               randomness={'seed':0,'deterministic':True,'diff_rank_seed':False},
               launcher='none',visualizer=None)
    cfg.pop('custom_imports',None)
    cfg.pop('tta_model',None);cfg.pop('tta_pipeline',None)
    return cfg

def environment_check(root, model, bundle_path):
    root=Path(root).resolve(); m=model
    expected_python=root/m['python_rel']
    require(Path(sys.executable).resolve()==expected_python.resolve(), 'Run with archived mmrot interpreter')
    for key in ['config','checkpoint','env_manifest','freeze']:
        checked(root/m[key+'_rel'],m[key+'_sha256'])
    frozen=Path(bundle_path)/'mmrot.freeze.txt'
    require(subprocess.check_output([sys.executable,'-m','pip','freeze','--all'],timeout=60)==frozen.read_bytes(), 'Environment freeze differs; no install/repair here')
    source=root/'run/src/mmrot'
    require(run_readonly(['git','-C',str(source),'rev-parse','HEAD']).strip()==m['source_commit'], 'Upstream commit mismatch')
    return source

def science_budget_gate(path,bundle,model,condition,image_count,free_bytes):
    budget=read_json(path)
    require(budget.get('schema')=='rotcert-pilot-budget-v1' and budget.get('bundle_sha256')==sha(bundle/'bundle.json'),'Pilot budget mismatch')
    key=model+'__'+condition
    require(key in budget['cells'], 'Exact model/condition needs pilot measurements')
    c=budget['cells'][key]
    require(c['repeats']>=2 and c['pilot_images']==200,'Two full pilot repeats needed')
    require(c['clean_parity_reviewed'] is True,'Clean baseline parity must be reviewed before science launch')
    needed=int(c['planned_prediction_bytes'])+10*1024**3
    require(free_bytes>=needed,'Insufficient prediction/export space from pilot budget')
    require(c['science_images']==image_count,'Pilot budget uses wrong science population')
    return c

def infer(args):
    import shutil
    import traceback
    created_output=False
    start=time.monotonic(); bp=Path(args.bundle).resolve(); b=load_bundle(bp)
    require(args.model in b['models'] and args.condition in b['conditions'],'Cell outside frozen tier')
    stage=load_prepared(args.prepared,bp);prepared_hash=sha(args.prepared);m=b['catalog']['models'][args.model]
    root=Path(args.legacy_root).resolve(); out=Path(args.out).resolve()
    # The data mount, legacy run root and sealed bundle are all protected input trees.
    for protected in [root,Path(stage['data_root']).resolve(),bp,Path(args.prepared).resolve().parent]:
        require(not out.is_relative_to(protected) and not protected.is_relative_to(out),'Output overlaps protected input tree')
    lease_path=Path(args.lease).resolve(); lease=read_json(lease_path);lease_hash=sha(lease_path)
    gpuid=validate_lease(lease,bp,out)
    lock_root=Path(lease['lock_root']).resolve()
    require(lock_root.is_relative_to(Path(lease['output_root']).resolve()),'Lock directory outside lease output root')
    require(not lock_root.is_relative_to(root),'No locks inside legacy tree')
    lock_root.mkdir(parents=True,exist_ok=True)
    lock=(lock_root/(gpuid+'.lock')).open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # OS lock is held by the inference process, not only a short-lived launcher.
    try:
        first=gpu_snapshot(gpuid); time.sleep(0.2); second=gpu_snapshot(gpuid)
        source=environment_check(root,m,bp)
        require(run_readonly(['git','-C',str(source),'diff','--no-ext-diff','--no-color','HEAD'])==b['catalog']['mmrot_git_diff'],'Source diff changed from archive')
        for name,h in b['catalog']['source_patch_files'].items(): checked(root/name,h)
        checked(root/'data/dior/data_manifest.json',b['catalog']['data_manifest_sha256'])
        rows=list(json_lines(stage['manifests'][args.condition]['path']))
        require(len(rows)==stage['image_count'] and len({r['id'] for r in rows})==len(rows),'Invalid image inventory')
        expected_rows=[r for r in json_lines(bp/'sources.jsonl') if stage['cohort']=='science' or r['pilot']]
        require([r['id'] for r in rows]==[r['id'] for r in expected_rows],'Stage omitted or added source images')
        for r,e in zip(rows,expected_rows):
            require(r['role']==e['role'] and r['source_id']==e['source_id'] and r['pilot']==e['pilot'] and r['condition']==args.condition,'Role/condition drift')
            checked(r['img_path'],r['input_sha256'])
            checked(Path(stage['data_root'])/r['annotation']['file'],r['annotation']['sha256'])
        if stage['cohort']=='science':
            require(b['tier']=='full','Minimum tier is a pilot gate, not reduced science scope')
            require(args.pilot_budget is not None,'Science blocked until two pilot repeats and budget/parity review')
            science_budget_gate(Path(args.pilot_budget),bp,args.model,args.condition,len(rows),shutil.disk_usage(out.parent).free)
        out=new_output(out,[root,bp,Path(args.prepared).parent]); created_output=True; (out/'inputs').mkdir()
        os.environ.update(CUDA_VISIBLE_DEVICES=gpuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONDONTWRITEBYTECODE='1',
                          OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        os.environ['TORCH_HOME']=str(out/'torch-cache')
        os.environ['MPLCONFIGDIR']=str(out/'mpl-cache')
        os.environ['TMPDIR']=str(out/'tmp'); (out/'tmp').mkdir()
        sys.dont_write_bytecode=True
        # Models are imported only beyond all authorization, hash, lease, and idle gates.
        sys.path.insert(0,str(source))
        import mmrotate
        require(Path(mmrotate.__file__).resolve().is_relative_to(source),'Wrong MMRotate namespace')
        from mmengine import Config
        from mmengine.dataset import BaseDataset
        from mmengine.hooks import Hook
        from mmengine.runner import Runner
        from mmdet.utils import register_all_modules as reg_mmdet
        from mmrotate.utils import register_all_modules as reg_mmrot
        from mmrotate.registry import DATASETS
        reg_mmdet(init_default_scope=False);reg_mmrot(init_default_scope=False)
        @DATASETS.register_module()
        class RotCertSupplementImages(BaseDataset):
            def __init__(self, manifest_path, **kwargs):
                self.manifest_path=manifest_path
                super().__init__(**kwargs)
            def load_data_list(self):
                return [{'img_id':r['id'],'img_path':r['img_path'],'width':r['width'],'height':r['height'],'instances':[]} for r in rows]
        class ResourceGuard(Hook):
            def __init__(self): self.seconds=[];self.last_check=0
            def before_test_iter(self,runner,batch_idx,data_batch=None):
                checked(lease_path,lease_hash);validate_lease(lease,bp,out)
                require(shutil.disk_usage(out).free>=10*1024**3,'Free disk below reserve; own job stops')
                if time.monotonic()-self.last_check>60:
                    gpu_snapshot(gpuid,check_idle=False,allowed_pids=[os.getpid()]); self.last_check=time.monotonic()
                self.t=time.monotonic()
            def after_test_iter(self,runner,batch_idx,data_batch=None,outputs=None):
                self.seconds.append(time.monotonic()-self.t)
        config=build_config(plain_config(bp/'configs'/f'{args.model}.py'),out,root/m['checkpoint_rel'],stage['manifests'][args.condition]['path'])
        Config(config).dump(str(out/'inputs/config.py')); config_hash=sha(out/'inputs/config.py')
        guard=ResourceGuard(); runner=Runner.from_cfg(Config(config));runner.register_hook(guard,priority='VERY_HIGH')
        inference_start=time.monotonic();runner.test();inference_seconds=time.monotonic()-inference_start
        checked(out/'inputs/config.py',config_hash)
        checked(root/m['checkpoint_rel'],m['checkpoint_sha256'])
        import pickle
        from export_canonical import mmrotate_records
        with (out/'predictions.pkl').open('rb') as f: samples=pickle.load(f) # trusted own-process dump only
        observed=[]
        for sample in samples:
            sample=sample if isinstance(sample,dict) else sample.to_dict()
            meta=sample.get('metainfo',{});observed.append(Path(sample.get('img_path',meta.get('img_path',''))).stem)
        require(len(observed)==len(set(observed)) and set(observed)==set(r['id'] for r in rows),'Missing, duplicate or extra predictions, including empty images')
        export_start=time.monotonic()
        write_rows(out/'detections.jsonl',mmrotate_records(samples,'dior',score_thr=0.05))
        write_rows(out/'image_inventory.jsonl',rows)
        shutil.copyfile(stage['ground_truth']['path'],out/'ground_truth.jsonl')
        for r in rows:
            checked(r['img_path'],r['input_sha256'])
            checked(Path(stage['data_root'])/r['annotation']['file'],r['annotation']['sha256'])
        load_bundle(bp);checked(Path(args.prepared),prepared_hash)
        checked(root/'data/dior/data_manifest.json',b['catalog']['data_manifest_sha256'])
        final_gpu=gpu_snapshot(gpuid,check_idle=False,allowed_pids=[os.getpid()])
        outputs={name:sha(out/name) for name in ['detections.jsonl','image_inventory.jsonl','ground_truth.jsonl','predictions.pkl','inputs/config.py']}
        receipt={'status':'PASS','scope':'INFERENCE_ONLY_'+stage['cohort'].upper(),'model':args.model,'condition':args.condition,
          'bundle_sha256':sha(bp/'bundle.json'),'prepared_sha256':sha(args.prepared),'checkpoint_sha256':m['checkpoint_sha256'],
          'archived_config_sha256':m['config_sha256'],'generated_config_sha256':config_hash,'lease_sha256':lease_hash,
          'python':sys.executable,'source_commit':m['source_commit'],'environment_freeze_sha256':m['freeze_sha256'],
          'images':len(rows),'prediction_image_ids':observed,'score_floor':0.05,'output_hashes':outputs,
          'wall_seconds':time.monotonic()-start,'inference_seconds':inference_seconds,'export_and_recheck_seconds':time.monotonic()-export_start,
          'batch_seconds':guard.seconds,'output_bytes':{n:(out/n).stat().st_size for n in outputs},
          'gpu_observations':[first,second,final_gpu],'completed_utc':utc(),'training_executed':False,
          'canonical_geometry':'le90 radians long-edge','command':sys.argv,'metric_claim':'No accuracy or coverage claim before offline analysis'}
        write_json(out/'run.json',receipt)
        write_json(out/'detections.jsonl.provenance.json',receipt)
        return receipt
    except Exception as e:
        if created_output and not (out/'failure.json').exists():
            write_json(out/'failure.json',{'status':'FAILED','error':str(e),'trace':traceback.format_exc(),'utc':utc()})
        raise
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','prepared','legacy-root','out','lease']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model',required=True);p.add_argument('--condition',required=True);p.add_argument('--pilot-budget',type=Path)
    a=p.parse_args();print(json.dumps(infer(a),indent=2))
if __name__=='__main__':main()

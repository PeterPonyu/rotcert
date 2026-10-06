#!/usr/bin/env python3
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
# Direct script execution adds jobs/ to sys.path, where our queue.py shadows
# Python's standard-library queue required by PyTorch. Keep kit imports explicit.
script_dir=Path(__file__).resolve().parent
sys.path[:]=[entry for entry in sys.path if Path(entry or os.getcwd()).resolve()!=script_dir]
sys.path.insert(0,str(Path(__file__).resolve().parents[1])); sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'export'))
from common import ROOT, PINS, sha256, atomic_json, utc, CLASSES
from canonical import write_export, mmrotate_records

sys.path.insert(0,str(ROOT/'data'))
from stage import EXPECTED, digest, snapshot, verify_staged_data

FREEZE_MODE='pip-freeze-all-v1'

def environment_snapshot(run_root,env):
    envdir=Path(run_root)/'env_manifests'
    env_content=(envdir/(env+'.json')).read_bytes()
    probe=json.loads(env_content)
    if probe['status']!='PASS' or probe['commit']!=PINS[env]: raise ValueError('Unverified environment')
    if probe.get('freeze_mode')!=FREEZE_MODE: raise ValueError('Unverified environment freeze mode; rebuild snapshots in a fresh run root')
    # Hash and compare one saved byte snapshot; never reopen it for provenance.
    freeze_content=(envdir/(env+'.freeze.txt')).read_bytes()
    freeze_hash=digest(freeze_content)
    marker=(envdir/('ENV_OK_'+env)).read_bytes().split()
    if not freeze_content.strip() or not marker or marker[0]!=freeze_hash.encode('ascii'):
        raise ValueError('Environment freeze marker mismatch')
    current=subprocess.check_output([sys.executable,'-m','pip','freeze','--all'])
    if current!=freeze_content: raise ValueError('Environment packages changed since probe')
    return probe,dict(env=digest(env_content),freeze_mode=FREEZE_MODE,freeze_sha256=freeze_hash)

def marker_snapshot(marker):
    record=json.loads(snapshot(marker)); manifest=Path(record['manifest'])
    if not manifest.is_file(): raise ValueError('Stale data marker: missing manifest')
    content=snapshot(manifest); content_hash=digest(content)
    if content_hash!=record['sha256']: raise ValueError('Stale data marker')
    metadata=json.loads(content)
    if metadata.get('dataset') in EXPECTED or metadata.get('dataset')=='hrsid':
        verify_staged_data(manifest,metadata)
    else:
        raise ValueError('Unknown staged dataset')
    return manifest,metadata,content_hash

def verify_marker(marker):
    manifest,metadata,_=marker_snapshot(marker)
    return manifest,metadata

def data_guard(marker,expected_path,dataset):
    manifest,metadata,content_hash=marker_snapshot(marker)
    if manifest.resolve()!=Path(expected_path).resolve() or metadata['dataset']!=dataset:
        raise ValueError('Data root/marker mismatch')
    def check():
        current,_,current_hash=marker_snapshot(marker)
        if current.resolve()!=manifest.resolve() or current_hash!=content_hash:
            raise ValueError('Data snapshot changed during job')
    return manifest,metadata,content_hash,check

def checked_phase(command,source,check_data):
    check_data()
    subprocess.run(command,cwd=source,check=True)
    check_data()

def prediction_ids(samples):
    ids=[]
    for sample in samples:
        if not isinstance(sample,dict): sample=sample.to_dict()
        meta=sample.get('metainfo',{})
        value=sample.get('img_path',meta.get('img_path',sample.get('img_id',meta.get('img_id'))))
        if value is None: raise ValueError('Prediction has no stable image id')
        ids.append(Path(str(value)).stem)
    return ids

def check_prediction_inventory(samples,expected):
    ids=prediction_ids(samples)
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected): raise ValueError('Inference missing/extra/duplicate images')
    return ids

def dataset_leaves(obj):
    if 'datasets' in obj:
        return [leaf for d in obj['datasets'] for leaf in dataset_leaves(d)]
    if 'dataset' in obj: return dataset_leaves(obj['dataset'])
    return [obj]

def smoke_config(cfg,work):
    cfg=copy.deepcopy(cfg); cfg['work_dir']=str(work)
    # Smoke trains on evaluation-image subsets; never used for scientific results.
    training_pipeline=copy.deepcopy(dataset_leaves(cfg['train_dataloader']['dataset'])[0].get('pipeline',[]))
    cfg['train_dataloader']['dataset']=copy.deepcopy(cfg['test_dataloader']['dataset'])
    for leaf in dataset_leaves(cfg['train_dataloader']['dataset']):
        leaf.update(test_mode=False,pipeline=training_pipeline,filter_cfg=dict(filter_empty_gt=False))
    cfg['train_dataloader']['sampler']=dict(type='InfiniteSampler',shuffle=True)
    cfg['train_cfg']=dict(type='IterBasedTrainLoop',max_iters=20,val_interval=21)
    cfg.update(val_cfg=None,val_dataloader=None,val_evaluator=None,param_scheduler=[])
    cfg['default_hooks']['checkpoint']=dict(type='CheckpointHook',interval=20,by_epoch=False,save_last=True,max_keep_ckpts=1)
    cfg['default_hooks']['logger']['interval']=1
    # Do not keep epoch-dependent pipeline-switch hooks in an iteration smoke.
    cfg['custom_hooks']=[]
    for key,limit in [('train_dataloader',16),('test_dataloader',10)]:
        loader=cfg[key]; loader.update(num_workers=0,persistent_workers=False,batch_size=1)
        leaves=dataset_leaves(loader['dataset']); cap=max(1,limit//len(leaves))
        for d in leaves: d['indices']=cap
    cfg['test_evaluator']=dict(type='DumpResults',out_file_path=str(work/'predictions.pkl'))
    return cfg

def execute_phase(config,phase,checkpoint):
    from mmengine import Config
    from mmdet.utils import register_all_modules as mmdet_register
    from mmrotate.utils import register_all_modules
    from mmengine.runner import Runner
    mmdet_register(init_default_scope=False); register_all_modules(init_default_scope=False)
    cfg=Config.fromfile(str(config))
    if phase=='infer': cfg.load_from=str(checkpoint); cfg.resume=False
    runner=Runner.from_cfg(cfg)
    if phase=='train': runner.train()
    else: runner.test()

def run(job,smoke_root=None):
    from mmengine import Config
    import pickle
    job=copy.deepcopy(job)
    if smoke_root: job.update(train=True,checkpoint=None)
    env=job['env']; run_root=Path(os.environ['PLANB_RUN']).resolve(); data_root=Path(os.environ['PLANB_DATA']).resolve()
    source=run_root/'src'/env
    package=__import__('mmrotate')
    if source.resolve() not in Path(package.__file__).resolve().parents: raise ValueError('Wrong mmrotate namespace for job')
    if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=PINS[env]: raise ValueError('Upstream commit mismatch')
    probe,env_identity=environment_snapshot(run_root,env)
    manifest,metadata,data_hash,check_data=data_guard(Path(os.environ['PLANB_STATE'])/'data_markers'/('DATA_OK_'+job['dataset']),data_root/job['dataset']/'data_manifest.json',job['dataset'])
    work=(Path(smoke_root).resolve()/job['id']) if smoke_root else run_root/'jobs'/job['id']
    work.mkdir(parents=True,exist_ok=True)
    cfg=Config.fromfile(str(ROOT/job['config'])).to_dict(); cfg['work_dir']=str(work)
    cfg['test_evaluator']=dict(type='DumpResults',out_file_path=str(work/'predictions.pkl'))
    if smoke_root: cfg=smoke_config(cfg,work)
    # MMEngine dumps its mutable runtime config to work/config.py in each phase.
    # Keep the hashed input outside that destination, including during inference.
    config=work/'inputs'/'config.py'; config.parent.mkdir(parents=True,exist_ok=True)
    Config(cfg).dump(str(config)); config_hash=sha256(config)
    fingerprint={'config':config_hash,'data':data_hash,**env_identity,'recipe':job['recipe']}
    def check_inputs():
        check_data()
        if sha256(config)!=config_hash: raise ValueError('Job input config changed during job')
    checkpoint=Path(os.path.expandvars(job['checkpoint'])) if job.get('checkpoint') else work/'final.pth'
    start=utc(); commands=[]
    def call(phase,ckpt=None):
        command=[sys.executable,str(Path(__file__).resolve()),'--phase',phase,'--config',str(config)]
        if ckpt: command+=['--checkpoint',str(ckpt)]
        commands.append(command)
        checked_phase(command,source,check_inputs)
    if job['train']:
        receipt=work/'train_complete.json'
        if receipt.exists():
            old=json.loads(receipt.read_text())
            if old['fingerprint']!=fingerprint or not checkpoint.is_file() or sha256(checkpoint)!=old['checkpoint_sha256']: raise ValueError('Training reuse mismatch; choose a fresh run root')
        else:
            call('train'); last=Path((work/'last_checkpoint').read_text().strip())
            if not last.is_absolute(): last=source/last
            if not last.is_file(): raise ValueError('Final checkpoint missing')
            import torch
            state=torch.load(last,map_location='cpu',weights_only=False)
            target=20 if smoke_root else cfg['train_cfg']['max_epochs']
            key='iter' if smoke_root else 'epoch'
            if state.get('meta',{}).get(key)!=target: raise ValueError('Checkpoint is not final '+key)
            if checkpoint.exists(): raise ValueError('Unreceipted final checkpoint; inspect before retry')
            checkpoint.symlink_to(last.resolve())
            atomic_json(receipt,{'fingerprint':fingerprint,'checkpoint_sha256':sha256(checkpoint)})
    if not checkpoint.is_file(): raise ValueError('Missing checkpoint '+str(checkpoint))
    call('infer',checkpoint)
    with (work/'predictions.pkl').open('rb') as f: samples=pickle.load(f)
    split=job['split']+('-'+job['modality'] if job['dataset']=='dv' else '')
    split_meta=metadata['splits'][split]; gt_source=Path(split_meta['gt_path'])
    gt_content=snapshot(gt_source)
    if digest(gt_content)!=split_meta['gt_sha256']: raise ValueError('Ground truth changed after staging')
    if len(samples)!=(min(10,split_meta['images']) if smoke_root else split_meta['images']): raise ValueError('Incomplete inference output')
    if 'inventory_sha256' in split_meta:
        inventory=manifest.parent/('inventory_'+split+'.json')
        inventory_content=snapshot(inventory)
        if digest(inventory_content)!=split_meta['inventory_sha256']: raise ValueError('Inventory changed')
        expected=[r['id'] for r in json.loads(inventory_content)['images']]
    else:
        split_path=manifest.parent/'split.json'
        split_content=snapshot(split_path)
        if digest(split_content)!=metadata['split_sha256']: raise ValueError('Split changed after staging')
        expected=json.loads(split_content)['parts'][split]
    observed=prediction_ids(samples)
    if smoke_root:
        if len(observed)!=len(set(observed)) or not set(observed)<=set(expected): raise ValueError('Invalid smoke image inventory')
    else: check_prediction_inventory(samples,expected)
    mapping_path=manifest.parent/'scene_map.json'
    mapping_content=snapshot(mapping_path) if mapping_path.is_file() else None
    if job['dataset']=='hrsid' and (mapping_content is None or digest(mapping_content)!=metadata['prepared_scene_map_sha256']): raise ValueError('Scene map changed after staging')
    mapping=json.loads(mapping_content) if mapping_content is not None else None
    offset=(100,100) if job['dataset']=='dv' else (0,0)
    provenance=dict(seed=job['seed'],config=str(config),checkpoint=str(checkpoint),data_manifest=str(manifest),command=commands,start_utc=start,gpu=probe['gpu'],upstream_commit=PINS[env],environment_manifest_sha256=env_identity['env'],freeze_mode=env_identity['freeze_mode'],freeze_sha256=env_identity['freeze_sha256'],coordinate_offset=list(offset),smoke_only=bool(smoke_root),train_reuse=bool(job['train'] and not any('train' in cmd for cmd in commands)))
    check_inputs()
    write_export(mmrotate_records(samples,job['dataset'],mapping,offset=offset),work/'detections.jsonl',provenance)
    if smoke_root:
        ids=set(observed)
        from common import write_jsonl
        write_jsonl(work/'ground_truth.jsonl',(json.loads(line) for line in gt_content.decode('utf-8').splitlines() if json.loads(line)['image_id'] in ids))
    else: (work/'ground_truth.jsonl').write_bytes(gt_content)
    check_inputs()
    atomic_json(work/'run.json',dict(data_verification='native_content_inventory_v2',job_id=job['id'],status='PASS',scope='smoke' if smoke_root else 'training_and_frozen_export',start_utc=start,end_utc=utc(),fingerprint=fingerprint,freeze_mode=env_identity['freeze_mode'],freeze_sha256=env_identity['freeze_sha256'],checkpoint_sha256=sha256(checkpoint),prediction_images=len(samples),mAP_computed=False))

def main():
    p=argparse.ArgumentParser(description='Execute clean MMRotate job in its own environment; GPU required')
    p.add_argument('--job'); p.add_argument('--jobs-file',type=Path,default=ROOT/'jobs/jobs.json'); p.add_argument('--smoke-root',type=Path)
    p.add_argument('--phase',choices=['train','infer']); p.add_argument('--config',type=Path); p.add_argument('--checkpoint',type=Path)
    a=p.parse_args()
    if a.phase:
        if not a.config or (a.phase=='infer' and not a.checkpoint): p.error('Phase needs config/checkpoint')
        execute_phase(a.config,a.phase,a.checkpoint)
    else:
        jobs={j['id']:j for j in json.loads(a.jobs_file.read_text())['jobs']}
        if a.job not in jobs: p.error('Unknown job')
        run(jobs[a.job],a.smoke_root)
if __name__=='__main__': main()

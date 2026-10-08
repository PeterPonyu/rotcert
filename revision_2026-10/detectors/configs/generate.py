#!/usr/bin/env python3
import argparse
import ast
import copy
from pathlib import Path
import pprint
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ROOT, CLASSES, PINS, atomic_json, sha256
RECIPES = {'orcnn':'oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py','roit':'roi_trans/roi-trans-le90_r50_fpn_1x_dota.py','s2anet':'s2anet/s2anet-le135_r50_fpn_1x_dota.py','rtmdet':'rotated_rtmdet/rotated_rtmdet_l-3x-dota.py','reppoints':'oriented_reppoints/oriented-reppoints-qbox_r50_fpn_1x_dota.py'}
DIOR = {'orcnn':'oriented-rcnn-le90_r50_fpn_1x_dior.py','roit':'roi-trans-le90_r50_fpn_1x_dior.py','s2anet':'s2anet-le135_r50_fpn_1x_dior.py','rtmdet':'rotated_rtmdet_l-3x-dior.py'}

def walk(obj,fun):
    if isinstance(obj,dict):
        fun(obj)
        for value in list(obj.values()): walk(value,fun)
    elif isinstance(obj,(list,tuple)):
        for value in obj: walk(value,fun)

def clean(cfg,seed,job_id,epochs):
    c=copy.deepcopy(cfg)
    c.update(randomness=dict(seed=seed,deterministic=False),work_dir='$PLANB_RUN/jobs/'+job_id,load_from=None,resume=False)
    c['train_cfg']=dict(type='EpochBasedTrainLoop',max_epochs=epochs,val_interval=epochs+1)
    c.update(val_cfg=None,val_dataloader=None,val_evaluator=None)
    c.setdefault('default_hooks',{})['checkpoint']=dict(type='CheckpointHook',interval=epochs,by_epoch=True,save_last=True,max_keep_ckpts=1,save_best=None)
    c['test_evaluator']=dict(type='DumpResults',out_file_path='$PLANB_RUN/jobs/'+job_id+'/predictions.pkl')
    c['test_cfg']=dict(type='TestLoop')
    walk(c.get('model',{}),lambda d:d.update(type='BN') if d.get('type')=='SyncBN' else None)
    return c

def set_root(node,root):
    walk(node,lambda d:d.update(data_root=root) if 'data_root' in d else None)

def standard_dataset(c,dataset,split):
    data='$PLANB_DATA/'+dataset+'/'
    walk(c['model'],lambda d:d.update(num_classes=len(CLASSES[dataset])) if 'num_classes' in d else None)
    if dataset in ('dior','rsar'):
        set_root(c['train_dataloader'],data); set_root(c['test_dataloader'],data)
        if dataset=='rsar':
            d=c['test_dataloader']['dataset']; d['ann_file']=split+'/annfiles/'; d['data_prefix']=dict(img_path=split+'/images/')
        return c
    for key,part in [('train_dataloader','train'),('test_dataloader',split)]:
        loader=c[key]; ds=loader['dataset']
        while 'dataset' in ds: ds=ds['dataset']
        loader['dataset']=dict(type='DOTADataset',data_root=data,ann_file=part+'/annfiles/',data_prefix=dict(img_path=part+'/images/'),metainfo=dict(classes=tuple(CLASSES[dataset])),pipeline=ds['pipeline'],img_suffix='png',test_mode=(key=='test_dataloader'))
        if key=='train_dataloader': loader['dataset']['filter_cfg']=dict(filter_empty_gt=True)
    return c

def resolve_recipe(ref,family,dataset,env):
    from mmengine import Config
    upstream=ref/('ai4rs' if env=='ai4rs' else 'mmrotate'); path=upstream/'configs'/RECIPES[family]
    if dataset=='rsar': path=upstream/'configs'/RECIPES[family].replace('_dota.py','_rsar.py').replace('le135','le90')
    if dataset=='rsar' and path.read_text().startswith('v_base_ ='):
        text=path.read_text().replace('v_base_ =','_base_ =',1)
        node=ast.parse(text).body[0]
        bases=[str((path.parent/p).resolve()) for p in ast.literal_eval(node.value)]
        lines=text.splitlines(); lines[node.lineno-1:node.end_lineno]=['_base_ = '+repr(bases)]
        staging=ROOT/'configs/generated/sources'/path.name; staging.parent.mkdir(parents=True,exist_ok=True)
        staging.write_text(chr(10).join(lines)+chr(10)); path=staging
    if dataset=='dior':
        source=ROOT.parents[1]/'pipeline/configs_dior'/DIOR[family]; text=source.read_text(); tree=ast.parse(text)
        base=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_base_' for t in n.targets))
        absolute=[str((path.parent/p).resolve()) for p in ast.literal_eval(base.value)]
        lines=text.splitlines(); lines[base.lineno-1:base.end_lineno]=['_base_ = '+repr(absolute)]
        staging=ROOT/'configs/generated/sources'/source.name; staging.parent.mkdir(parents=True,exist_ok=True)
        staging.write_text(chr(10).join(lines)+chr(10)); path=staging
    return Config.fromfile(str(path)).to_dict(),{'file':str(path.relative_to(ROOT)) if ROOT in path.parents else str(path.relative_to(ref)),'sha256':sha256(path),'upstream_commit':PINS[env]}

def dump_config(c,path):
    lines=['# Generated from pinned recipe with explicit clean protocol.','_base_ = []','import os']; q=chr(39)
    for key,value in c.items():
        if key.startswith('__'): continue
        text=pprint.pformat(value,width=100,sort_dicts=False)
        text=re.sub(q+'([$]PLANB_[A-Z_]+[^'+q+']*)'+q,lambda m:'os.path.expandvars('+repr(m.group(1))+')',text)
        lines.append(key+' = '+text)
    path.write_text(chr(10).join(lines)+chr(10))

def build(ref):
    jobs=[]; recipes={}; out=ROOT/'configs/generated'; out.mkdir(parents=True,exist_ok=True)
    def add(dataset,family,seed=0,split='test',env='mmrot',hours=1,train=True,depends=None):
        jid=f'{dataset}-{family}-s{seed}'+('' if train else '-'+split); cache=(dataset,family,env)
        if cache not in recipes: recipes[cache]=resolve_recipe(ref,family,dataset,env)
        c,origin=recipes[cache]; c=standard_dataset(copy.deepcopy(c),dataset,split); c=clean(c,seed,jid,36 if family=='rtmdet' else 12)
        dump_config(c,out/(jid+'.py'))
        checkpoint=None if train else (('$PLANB_RUN/jobs/'+depends[0]+'/final.pth') if depends else '$PLANB_CHECKPOINTS/'+dataset+'-'+family+'.pth')
        job=dict(id=jid,env=env,gpus=1,est_hours=hours,estimate_basis='Plan B ordering estimate; not benchmarked',data_deps=[dataset],depends_on_jobs=depends or [],env_marker='$PLANB_RUN/env_manifests/ENV_OK_'+env,work_dir='$PLANB_RUN/jobs/'+jid,cmd={'train':'${PLANB_RUN}/envs/'+env+'/bin/python ${PLANB_ROOT}/jobs/run_mm.py --job '+jid},outputs=['$PLANB_RUN/jobs/'+jid+'/'+f for f in ['detections.jsonl','detections.jsonl.provenance.json','ground_truth.jsonl','run.json','config.py','inputs/config.py']],dataset=dataset,family=family,seed=seed,split=split,train=train,checkpoint=checkpoint,config='configs/generated/'+jid+'.py',recipe=origin)
        jobs.append(job); return job
    for fam,count,h in [('orcnn',5,7.4),('roit',3,8),('s2anet',3,6),('rtmdet',3,30)]:
        for seed in range(count): add('dior',fam,seed,hours=h)
    for fam,count,h in [('orcnn',3,12),('roit',1,13),('rtmdet',1,30),('reppoints',1,12)]:
        for seed in range(count): add('dota',fam,seed,split='val200',hours=h)
    for split in ['val0','val500']: add('dota','orcnn',split=split,train=False,depends=['dota-orcnn-s0'])
    for fam in ['orcnn','roit','s2anet']:
        for split in ['val','test']: add('rsar',fam,split=split,env='ai4rs',train=False)
    for fam in ['orcnn','roit']: add('hrsid',fam,split='eval',env='ai4rs',hours=2)
    add('codrone','orcnn',env='ai4rs',hours=8)
    from mmengine import Config
    for modality in ['rgb','ir']:
        c,origin=resolve_recipe(ref,'orcnn','dota','ai4rs'); base=Config.fromfile(str(ref/'ai4rs/configs/_base_/datasets'/('dronevehicle_'+modality+'.py'))).to_dict()
        c.update(base); c['model']['roi_head']['bbox_head']['num_classes']=5; set_root(c,'$PLANB_DATA/dv/')
        jid='dv-orcnn-'+modality+'-s0'; c=clean(c,0,jid,12); dump_config(c,out/(jid+'.py'))
        jobs.append(dict(id=jid,env='ai4rs',gpus=1,est_hours=4,data_deps=['dv'],depends_on_jobs=[],env_marker='$PLANB_RUN/env_manifests/ENV_OK_ai4rs',work_dir='$PLANB_RUN/jobs/'+jid,cmd={'train':'${PLANB_RUN}/envs/ai4rs/bin/python ${PLANB_ROOT}/jobs/run_mm.py --job '+jid},outputs=['$PLANB_RUN/jobs/'+jid+'/'+f for f in ['detections.jsonl','detections.jsonl.provenance.json','ground_truth.jsonl','run.json','config.py','inputs/config.py']],dataset='dv',family='orcnn',modality=modality,seed=0,split='test',train=True,checkpoint=None,config='configs/generated/'+jid+'.py',recipe=origin))
    blocked=[('rsar-arsdetr','Spec B5 requires ARS-DETR; pinned implementation/config/checkpoint missing. O2 is not a substitute.'),('dv-o2-rgb','Pinned lazy O2 config requires installed ai4rs project modules and verified cropped-coordinate export; remote validation outstanding.'),('dv-o2-ir','Same O2 runtime/coordinate gate as RGB.'),('codrone-eavdetr','Pinned EAV upstream fit evaluates validation every epoch and dataset targets omit filename used by CP tools; torch2.0 AMP compatibility and clean runner require audited fixes. CODrone tiling/source split also unresolved. Raw tensor exporter is implemented; training/calibration are not claimed.'),('unit-audit','Local audit/unit_audit.py implements threshold and connected-component gates. Requires provenance-verified DINOv2-S fp16 embeddings and DOTA/HRSID scene truth; extraction and queue integration remain blocked. Null scene IDs must not be certified.'),('hrsc-conditional','Main must resolve conditional HRSC audit need and migrate approved data/checkpoints.'),('dior-haze','Optional lowest-priority arm; baseline freeze and haze data gate required before scheduling.')]
    jobs.extend(dict(id=jid,gpus=1,blocked_reason=reason,cmd={},outputs=[]) for jid,reason in blocked)
    atomic_json(ROOT/'jobs/jobs.json',{'schema_version':1,'scope':'Local recipes with explicit capability blocks; remote smoke mandatory','jobs':jobs})
    return {'generated_configs':sum('config' in j for j in jobs),'jobs':len(jobs),'blocked':len(blocked)}
def main():
    p=argparse.ArgumentParser(description='Resolve pinned recipes into portable configs and job graph'); p.add_argument('--refs',type=Path,default=Path('/tmp/planB_ref')); a=p.parse_args(); print(build(a.refs))
if __name__=='__main__': main()

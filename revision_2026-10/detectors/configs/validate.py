#!/usr/bin/env python3
'''Static recipe validation with mmengine; does not instantiate GPU models.'''
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ROOT,CLASSES,atomic_json,sha256,utc

def mappings(value):
    if isinstance(value,dict):
        yield value
        for child in value.values(): yield from mappings(child)
    elif isinstance(value,(tuple,list)):
        for child in value: yield from mappings(child)

def validate(jobs_file):
    from mmengine import Config
    os.environ.setdefault('PLANB_DATA','/validation/data'); os.environ.setdefault('PLANB_RUN','/validation/run')
    jobs=json.loads(Path(jobs_file).read_text())['jobs']; results=[]
    for j in jobs:
        if 'config' not in j: continue
        path=ROOT/j['config']; c=Config.fromfile(str(path)).to_dict()
        assert all(c[k] is None for k in ('val_cfg','val_dataloader','val_evaluator')),j['id']
        epochs=c['train_cfg']['max_epochs']; hook=c['default_hooks']['checkpoint']
        assert c['train_cfg']['val_interval']>epochs and hook['interval']==epochs and hook['save_last'] and hook['max_keep_ckpts']==1,j['id']
        assert c['test_evaluator']['type']=='DumpResults' and c['randomness']['seed']==j['seed'],j['id']
        assert c['load_from'] is None and c['resume'] is False,j['id']
        counts=[d['num_classes'] for d in mappings(c['model']) if 'num_classes' in d]
        assert counts and all(n==len(CLASSES[j['dataset']]) for n in counts),(j['id'],counts)
        assert not any(d.get('type')=='SyncBN' for d in mappings(c['model'])),j['id']
        text=str(c); assert '/tmp/planB_ref' not in text and '$PLANB_' not in text,j['id']
        for loader in ('train_dataloader','test_dataloader'):
            roots=[d['data_root'] for d in mappings(c[loader]) if 'data_root' in d]
            assert roots and all(root==os.environ['PLANB_DATA']+'/'+j['dataset']+'/' for root in roots),(j['id'],roots)
        if j['dataset']=='dota':
            train_ann=[d.get('ann_file','') for d in mappings(c['train_dataloader'])]
            assert not any('val' in name or 'test' in name for name in train_ann),j['id']
        results.append({'job':j['id'],'config':j['config'],'sha256':sha256(path),'classes':counts,'epochs':epochs,'status':'PASS_STATIC_ONLY'})
    return {'time':utc(),'configs':len(results),'results':results,'gpu_verified':False,'blocked_jobs':[{'id':j['id'],'reason':j['blocked_reason']} for j in jobs if j.get('blocked_reason')]}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--jobs',type=Path,default=ROOT/'jobs/jobs.json'); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    result=validate(a.jobs); atomic_json(a.output,result); print('Validated',result['configs'],'configs; GPU verification NOT_RUN')
if __name__=='__main__': main()

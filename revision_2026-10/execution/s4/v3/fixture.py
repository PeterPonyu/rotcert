"""Fully synthetic24-run archive in owned tree; no real prediction reads."""
from config import *
from io_s4 import *
from inputs import roles_from_ids,digest_obj
import math


def generate(name='synthetic-v1',per_role=200):
 root=owned(ROOT/'fixtures'/name);require(not root.exists(),'Fixture exists: preserve evidence')
 root.mkdir(parents=True);bundle=root/'bundle';bundle.mkdir()
 ids=[f'synthetic-{i:04d}' for i in range(2*per_role)];roles=roles_from_ids(ids)
 ordered={role:sorted([s for s in ids if roles[s]==role]) for role in ['calibration','evaluation']}
 rank={s:i for rr in ordered.values() for i,s in enumerate(rr)}
 sources=[dict(id=s,source_id=s,role=roles[s],width=256,height=256,sha256=digest_obj([s,'image']),annotation=dict(sha256=digest_obj([s,'labels']),file=s+'.xml')) for s in ids]
 write_rows(bundle/'sources.jsonl',sources)
 gt=[]
 for sid in ids:
  i=rank[sid]
  # Each role has200 sources, roughly10/class; heterogeneity and sparse cells survive.
  if i==per_role-1:continue # true zero-GT source, still inventory/FP accounting
  k=i%20
  for j in range(1+(i%3==0)):
   gt.append(dict(image_id=sid,scene_id=sid,class_=CLASSES[k],obb=[40.+60*j,40.,20.,10.,0.],gt_id=sid+':'+str(j)))
 gt=[{('class' if k=='class_' else k):v for k,v in r.items()} for r in gt]
 write_rows(bundle/'ground_truth_all.jsonl',gt)
 catalog={m:dict(checkpoint_sha256=digest_obj([m,'checkpoint']),config_sha256=digest_obj([m,'config']),source_commit='synthetic-source-only',freeze_sha256=digest_obj('synthetic-env')) for m in MODELS}
 b=dict(schema='SYNTHETIC-NO-SCIENCE',models=MODELS,conditions=CONDITIONS,protocol_id=PROTOCOL_ID,tier='full',catalog=dict(models=catalog),files={f:sha(bundle/f) for f in ['sources.jsonl','ground_truth_all.jsonl']})
 write_json(bundle/'bundle.json',b);entries=[]
 for mi,model in enumerate(MODELS):
  for ci,condition in enumerate(CONDITIONS):
   work=root/'runs'/(model+'__'+condition+'__science');(work/'inputs').mkdir(parents=True)
   detections=[]
   for row in gt:
    sid=row['image_id'];i=rank[sid]
    if i%11==0 or (ci>0 and i%7==0):continue # zero-TP GT-source/shift attrition
    if ci==5 and row['class']==CLASSES[19] and roles[sid]=='evaluation':continue # required missing eval class
    if ci==4 and row['class']==CLASSES[18] and roles[sid]=='calibration':continue # empty cal q=inf
    d={k:v for k,v in row.items() if k!='gt_id'};d['obb']=row['obb'].copy()
    shift=(i%5)*.6+ci*.3+mi*.05;d['obb'][0]+=shift;d['score']=.9
    detections.append(d)
   for sid in ids[::5]:detections.append({'image_id':sid,'scene_id':sid,'class':CLASSES[0],'obb':[220.,220.,15.,5.,0.],'score':.06})
   write_rows(work/'detections.jsonl',detections)
   write_rows(work/'ground_truth.jsonl',gt)
   inv=[dict(s,condition=condition,input_sha256=s['sha256'] if ci==0 else digest_obj([s['id'],condition,'pixels']),transformed_rgb_sha256=digest_obj([s['id'],condition,'rgb'])) for s in sources]
   write_rows(work/'image_inventory.jsonl',inv)
   (work/'predictions.pkl').write_bytes(b'SYNTHETIC FILE HASH ONLY, NEVER UNPICKLE')
   (work/'inputs/config.py').write_text('# synthetic inference metadata only\n')
   outputs={f:sha(work/f) for f in ['detections.jsonl','ground_truth.jsonl','image_inventory.jsonl','predictions.pkl','inputs/config.py']}
   m=catalog[model];receipt=dict(status='PASS',scope='INFERENCE_ONLY_SCIENCE',model=model,condition=condition,bundle_sha256=sha(bundle/'bundle.json'),checkpoint_sha256=m['checkpoint_sha256'],archived_config_sha256=m['config_sha256'],source_commit=m['source_commit'],environment_freeze_sha256=m['freeze_sha256'],generated_config_sha256=outputs['inputs/config.py'],training_executed=False,score_floor=.05,images=len(ids),prediction_image_ids=ids,output_hashes=outputs,synthetic=True)
   write_json(work/'run.json',receipt);entries.append(dict(model=model,condition=condition,path=str(work.relative_to(root)),receipt_sha256=sha(work/'run.json')))
 index=dict(schema='S4-input-index-v1',synthetic=True,bundle='bundle',bundle_sha256=sha(bundle/'bundle.json'),runs=entries)
 write_json(root/'index.json',index)
 return root/'index.json'

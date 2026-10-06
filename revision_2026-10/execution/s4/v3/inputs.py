"""Strict archive adapter. No pickle loading, remote access or inference."""
from config import *
from io_s4 import *
import hashlib
from collections import defaultdict
from rotcert_contracts.matching import validate_matching_inputs


def digest_obj(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def child(root,name):
 p=(Path(root)/name).resolve();require(not Path(name).is_absolute() and p.is_relative_to(Path(root).resolve()),'Archive path escape');return p


def roles_from_ids(ids):
 require(len(ids)==len(set(ids)),'Duplicate source IDs')
 rank=sorted(ids,key=lambda s:(hashlib.sha256((PROTOCOL_ID+'\0'+s).encode()).hexdigest(),s))
 return {sid:('calibration' if i<len(rank)//2 else 'evaluation') for i,sid in enumerate(rank)}


def validate_record(r,sources,gt=False):
 require(r.get('image_id') in sources,'Extra image in canonical rows')
 require(r.get('scene_id',r['image_id'])==sources[r['image_id']]['source_id'],'Canonical source changed')
 require(r.get('class') in CLASSES,'Unknown class')
 validate_matching_inputs([] if gt else [r],[r] if gt else [],.05)
 if not gt:require(float(r['score'])>=.05,'Below frozen export floor')
 else:require(isinstance(r.get('gt_id'),str) and r['gt_id'],'GT id required')


class InputSet:
 """Input index pins24 receipts and canonical outputs to one frozen GPU bundle.

 synthetic=True accepts only paths under an explicitly marked S4 fixture root;
 it cannot silently relax production schema/identity checks for real data.
 """
 def __init__(self,index_path,synthetic=False):
  self.index_path=Path(index_path).resolve();self.index=json.loads(self.index_path.read_text());self.synthetic=synthetic
  self.files={str(self.index_path):sha(self.index_path)}
  require(self.index.get('schema')=='S4-input-index-v1','Input index schema')
  if synthetic:
   require(self.index_path.is_relative_to(ROOT/'fixtures') and self.index.get('synthetic') is True,'Synthetic mode cannot consume external/real outputs')
  else:require(self.index.get('synthetic') is False,'Production excludes synthetic inputs')
  self.bundle_path=(self.index_path.parent/Path(self.index['bundle'])).resolve()
  if synthetic:require(self.bundle_path.is_relative_to(self.index_path.parent),'Synthetic bundle must stay inside fixture')
  self.bundle=self.bound(self.bundle_path/'bundle.json',self.index['bundle_sha256'],json_object=True)
  b=self.bundle
  if not synthetic:
   expected=json.loads((ROOT/'snapshot/GPU-BUNDLE.json').read_text())
   require(self.index['bundle_sha256']==sha(ROOT/'snapshot/GPU-BUNDLE.json') and b==expected,'Production bundle not frozen original full bundle')
  require(b['models']==MODELS and b['conditions']==CONDITIONS and b['tier']=='full','Must contain four families and six conditions')
  require(b['protocol_id']==PROTOCOL_ID,'Role hash domain changed')
  for name,h in b['files'].items():self.bound(child(self.bundle_path,name),h)
  sources=list(json_rows(self.bundle_path/'sources.jsonl'))
  self.sources={r['id']:r for r in sources}
  require(len(sources)==len(self.sources),'Duplicate image inventory')
  require(len({r['source_id'] for r in sources})==len(sources),'One source must be one image')
  require(all(r['source_id']==r['id'] for r in sources),'DIOR source=image contract')
  if not synthetic:require(len(sources)==11738,'Expected11738 fixed sources')
  self.ids=sorted(self.sources);self.source_index={s:i for i,s in enumerate(self.ids)}
  expected_roles=roles_from_ids(self.ids)
  for sid,r in self.sources.items():
   require(r['role']==expected_roles[sid],'Source role mismatch to predeclared hash rank')
   require(type(r['width']) is int and type(r['height']) is int and min(r['width'],r['height'])>0,'Invalid dimensions')
  self.roles=np.array([0 if expected_roles[s]=='calibration' else 1 for s in self.ids],dtype=np.int8)
  if not synthetic:require(np.sum(self.roles==0)==5869 and np.sum(self.roles==1)==5869,'Role counts5869/5869')
  self.gt_rows=list(json_rows(self.bundle_path/'ground_truth_all.jsonl'));self.gt=defaultdict(list)
  seen=set()
  for r in self.gt_rows:
   validate_record(r,self.sources,True);key=(r['image_id'],r['gt_id'])
   require(key not in seen,'Duplicate GT key');seen.add(key);self.gt[r['image_id']].append(r)
  self.gt_signature={s:digest_obj(self.gt[s]) for s in self.ids}
  self.gt_counts=np.zeros((len(self.ids),20),dtype=np.int64)
  for s,rows in self.gt.items():
   for r in rows:self.gt_counts[self.source_index[s],CLASSES.index(r['class'])]+=1
  entries=self.index['runs'];require(len(entries)==24,'Exactly24 run receipts required')
  self.runs={};self.inventory_hashes={};self.pixel_by_condition={}
  for entry in entries:
   tick('input_acceptance')
   key=(entry['model'],entry['condition']);require(key not in self.runs,'Duplicate model-condition receipt')
   require(key[0] in MODELS and key[1] in CONDITIONS,'Unplanned run')
   work=(self.index_path.parent/Path(entry['path'])).resolve()
   if synthetic:require(work.is_relative_to(self.index_path.parent),'Synthetic runs must stay inside fixture')
   r=self.bound(work/'run.json',entry['receipt_sha256'],json_object=True)
   require(r['status']=='PASS' and r['scope']=='INFERENCE_ONLY_SCIENCE','Only completed science outputs')
   require((r['model'],r['condition'])==key and r['bundle_sha256']==self.index['bundle_sha256'],'Run identity')
   m=b['catalog']['models'][key[0]]
   for rk,mk in [('checkpoint_sha256','checkpoint_sha256'),('archived_config_sha256','config_sha256'),('source_commit','source_commit'),('environment_freeze_sha256','freeze_sha256')]:require(r[rk]==m[mk],'Run model lineage '+rk)
   require(r['training_executed'] is False and r['score_floor']==.05,'Training or floor changed')
   require(r['images']==len(self.ids),'Run missing sources')
   require(len(r['prediction_image_ids'])==len(self.ids) and set(r['prediction_image_ids'])==set(self.ids),'Prediction source IDs incomplete/duplicate')
   required={'detections.jsonl','ground_truth.jsonl','image_inventory.jsonl','predictions.pkl','inputs/config.py'}
   require(required<=set(r['output_hashes']),'Missing canonical/provenance hash')
   for name,h in r['output_hashes'].items():self.bound(child(work,name),h)
   require(r['generated_config_sha256']==r['output_hashes']['inputs/config.py'],'Generated config mismatch')
   require(r['output_hashes']['ground_truth.jsonl']==b['files']['ground_truth_all.jsonl'],'GT bytes changed between variants')
   inv=list(json_rows(work/'image_inventory.jsonl'))
   require(len(inv)==len(self.ids) and {x['id'] for x in inv}==set(self.ids),'Missing zero-detection inventory row')
   pixels={}
   for row in inv:
    src=self.sources[row['id']]
    for k,v in src.items():require(row.get(k)==v,'Source/GT/image provenance changed '+k)
    require(row['condition']==key[1],'Pixel condition drift')
    for h in ['input_sha256','transformed_rgb_sha256']:require(isinstance(row.get(h),str) and len(row[h])==64,'Pixel hash absent')
    if key[1]=='identity':require(row['input_sha256']==src['sha256'],'Identity decoder input changed')
    pixels[row['id']]=(row['input_sha256'],row['transformed_rgb_sha256'])
   if key[1] in self.pixel_by_condition:require(pixels==self.pixel_by_condition[key[1]],'Different transform bytes across detectors')
   else:self.pixel_by_condition[key[1]]=pixels
   for row in json_rows(work/'detections.jsonl'):validate_record(row,self.sources)
   # All-source GT invariant is checked semantically as well as identical bytes.
   gt_again=defaultdict(list)
   for row in json_rows(work/'ground_truth.jsonl'):gt_again[row['image_id']].append(row)
   require({s:digest_obj(gt_again[s]) for s in self.ids}==self.gt_signature,'Per-source GT invariant failed')
   self.runs[key]=dict(path=work,receipt=r,receipt_sha256=entry['receipt_sha256'])
  require(set(self.runs)=={(m,c) for m in MODELS for c in CONDITIONS},'Incomplete24cell product')
  self.guard=InputGuard(self.files)
  self.binding=dict(index_sha256=sha(self.index_path),bundle_sha256=self.index['bundle_sha256'],sources_sha256=b['files']['sources.jsonl'],gt_sha256=b['files']['ground_truth_all.jsonl'],synthetic=synthetic,
                    run_receipts={m+'__'+c:r['receipt_sha256'] for (m,c),r in self.runs.items()})
  self.binding_hash=digest_obj(self.binding)
 def bound(self,p,h,json_object=False):
  require(sha(p)==h,'Bound input bytes changed '+str(p));self.files[str(p)]=h
  return json.loads(Path(p).read_text()) if json_object else None
 def detections(self,model,condition):
  result=defaultdict(list)
  for r in json_rows(self.runs[(model,condition)]['path']/'detections.jsonl'):result[r['image_id']].append(r)
  return result

"""Offline rotated-IoU matching of full source inventory, never survivors only."""
from config import *
from io_s4 import *
from analysis_vendor.matching import greedy_match
from analysis_vendor.gwd import obb_gwd
from rotcert_contracts.matching import validate_matching_inputs
import time
import shutil
from resources import write_allowance,write_finished


def match_condition(inputs,model,condition,tau):
 detections=inputs.detections(model,condition);n=len(inputs.ids)
 counts=np.zeros((n,20,3),dtype=np.int64) # gt,det,tp
 counts[:,:,0]=inputs.gt_counts
 tp_source=[];tp_class=[];tp_pred=[];tp_gt=[];tp_iou=[];det_source=[];det_class=[];det_box=[]
 for si,sid in enumerate(inputs.ids):
  tick('preparation')
  ds=detections[sid];gs=inputs.gt[sid];validate_matching_inputs(ds,gs,.05,tau)
  for k,cls in enumerate(CLASSES):
   dc=[r for r in ds if r['class']==cls];gc=[r for r in gs if r['class']==cls]
   counts[si,k,1]=len(dc)
   match=greedy_match(dc,gc,iou_thr=tau)
   counts[si,k,2]=len(match['matches'])
   require(len(match['matches'])+len(match['unmatched_gt_indices'])==len(gc),'GT conservation')
   require(len(match['matches'])+len(match['unmatched_det_indices'])==len(dc),'Detection conservation')
   for p in match['matches']:
    tp_source.append(si);tp_class.append(k);tp_pred.append(dc[p['det_index']]['obb']);tp_gt.append(gc[p['gt_index']]['obb']);tp_iou.append(p['iou'])
   for d in dc:det_source.append(si);det_class.append(k);det_box.append(d['obb'])
 p=np.asarray(tp_pred,dtype=float).reshape(-1,5);g=np.asarray(tp_gt,dtype=float).reshape(-1,5)
 score=obb_gwd(p,g) if len(p) else np.empty(0)
 require(np.isfinite(score).all() and np.all(score>=0),'Invalid localization score')
 return dict(counts=counts,tp_source=np.asarray(tp_source,dtype=np.int64),tp_class=np.asarray(tp_class,dtype=np.int16),
             tp_pred=p,tp_gt=g,tp_score=score,tp_iou=np.asarray(tp_iou),
             det_source=np.asarray(det_source,dtype=np.int64),det_class=np.asarray(det_class,dtype=np.int16),det_box=np.asarray(det_box,dtype=float).reshape(-1,5))


def prepared_metadata(inputs,source_sha,model,condition,tau):
 return dict(source_sha256=source_sha,input_binding_sha256=inputs.binding_hash,model=model,condition=condition,iou=tau,score_floor=.05,schema='S4-matched-v1')


def prepare_all(inputs,out,source_sha):
 out=owned(out);out.mkdir(parents=True,exist_ok=True);meta=out/'INPUTS.json'
 if meta.exists():require(json.loads(meta.read_text())==dict(binding=inputs.binding,files=inputs.files),'Prepared input binding differs')
 else:write_json(meta,dict(binding=inputs.binding,files=inputs.files))
 for model in MODELS:
  for tau in TAUS:
   for condition in CONDITIONS:
    tick('preparation')
    p=out/f'{model}__{condition}__iou{tau:.1f}.npz';m=prepared_metadata(inputs,source_sha,model,condition,tau)
    if p.exists() or p.with_suffix('.json').exists():load_npz(p,m)
    else:
     inputs.guard.check()
     if not import_prepared(inputs,p,m,model,condition,tau):
      arrays=match_condition(inputs,model,condition,tau);inputs.guard.check();save_npz(p,arrays,m)
 inputs.guard.check(full=True)
 return out


def load_model(inputs,prepared,source_sha,model,tau):
 return [load_npz(Path(prepared)/f'{model}__{c}__iou{tau:.1f}.npz',prepared_metadata(inputs,source_sha,model,c,tau)) for c in CONDITIONS]


def import_prepared(inputs,destination,metadata,model,condition,tau):
 """Reuse immutable matching arrays only through an explicit source/input gate."""
 contract_path=ROOT/'PREPARED-REUSE.json'
 if inputs.synthetic or not contract_path.exists():return False
 contract=json.loads(contract_path.read_text())
 require(contract['schema']=='S4-reviewed-prepared-reuse-v1','Prepared reuse schema')
 require(inputs.binding_hash==contract['original_input_binding_sha256'],'Reuse input mismatch')
 prior_root=Path(contract['original_source_root'])
 require(sha(prior_root/'FROZEN.json')==contract['original_source_sha256'],'Original freeze drift')
 old_manifest=json.loads((prior_root/'FROZEN.json').read_text())
 for name in contract['unchanged_source_dependencies']:
  require(sha(ROOT/name)==old_manifest['files'][name] and sha(prior_root/name)==old_manifest['files'][name],'Matching dependency drift '+name)
 require(sha(prior_root/'prepare.py')==contract['match_function_source_sha256'],'Original matcher source drift')
 life_path=Path(contract['original_lifecycle_path']);require(sha(life_path)==contract['original_lifecycle_sha256'],'Original stopped ledger changed')
 require(json.loads(life_path.read_text())['status']==contract['original_lifecycle_terminal_status'],'Original lifecycle not terminal')
 item=contract['prepared_files'][destination.name];old=Path(item['path']);side=Path(item['sidecar_path'])
 require(sha(old)==item['sha256'] and sha(side)==item['sidecar_sha256'],'Prepared bytes/sidecar drift')
 old_meta=json.loads(side.read_text())
 expected=dict(metadata,source_sha256=contract['original_source_sha256'])
 arrays=load_npz(old,expected)
 # Validate dtype/shape while reading, without matching/inference/recalibration.
 require(arrays['counts'].shape==(len(inputs.ids),20,3),'Prepared count shape')
 write_allowance(destination,old.stat().st_size)
 tmp=destination.with_name(destination.name+'.tmp-'+uuid.uuid4().hex)
 try:
  with old.open('rb') as src,tmp.open('xb') as dst:
   shutil.copyfileobj(src,dst,2**20);dst.flush();os.fsync(dst.fileno())
  require(sha(tmp)==item['sha256'],'Prepared copy mismatch');publish(tmp,destination);write_finished(destination)
 finally:
  if tmp.exists():tmp.unlink()
 write_json(destination.with_suffix('.json'),dict(metadata,file=destination.name,sha256=item['sha256'],
   provenance_reuse=dict(contract_sha256=sha(contract_path),original_source_sha256=contract['original_source_sha256'],original_sidecar_sha256=item['sidecar_sha256'])))
 return True

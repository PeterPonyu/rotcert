"""Local exploratory matching, exact polygon IoU and frozen greedy rule.
Only writes audit/analysis. Uses existing polygon construction, prunes pairs with
nonintersecting envelopes (IoU exactly zero), and vectorizes GEOS intersections.
Compares the resulting matches against the unmodified matcher before use.
"""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import collections,hashlib,json,sys,time
import numpy as np
import shapely
HERE=Path(__file__).resolve().parent
# The repository root, which holds the rotcert package and revision_2026-10/wpR.
WORKTREE=Path(os.environ.get('ROTCERT_ROOT',HERE.parents[1]))
sys.path.insert(0,str(WORKTREE));sys.path.insert(0,str(WORKTREE/'revision_2026-10/wpR'))
from rotcert.matching import greedy_match,obb_to_polygon
from rotcert import io as rio
from build_cache import build

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def accelerated(dets,gts):
 dp=np.array([obb_to_polygon(d['obb']) for d in dets],dtype=object)
 gp=np.array([obb_to_polygon(g['obb']) for g in gts],dtype=object)
 if len(dp):dp=shapely.buffer(dp,0,where=~shapely.is_valid(dp),out=dp.copy())
 if len(gp):gp=shapely.buffer(gp,0,where=~shapely.is_valid(gp),out=gp.copy())
 db=shapely.bounds(dp);gb=shapely.bounds(gp);ga=shapely.area(gp);da=shapely.area(dp)
 gc=np.array([g['class'] for g in gts]);used=np.zeros(len(gts),bool)
 matches=[];unmatched=[]
 for di in sorted(range(len(dets)),key=lambda i:(-float(dets[i]['score']),i)):
  if not len(gts):unmatched.append(di);continue
  b=db[di]
  ix=np.flatnonzero((~used)&(gc==dets[di]['class'])&(gb[:,0]<b[2])&(gb[:,2]>b[0])&(gb[:,1]<b[3])&(gb[:,3]>b[1]))
  if not len(ix):unmatched.append(di);continue
  inter=shapely.area(shapely.intersection(dp[di],gp[ix]));union=da[di]+ga[ix]-inter
  ious=np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
  k=int(np.argmax(ious));gj=int(ix[k]);val=float(ious[k])
  if val>=0.5:used[gj]=True;matches.append({'det_index':di,'gt_index':gj,'iou':val})
  else:unmatched.append(di)
 return {'matches':sorted(matches,key=lambda m:m['det_index']),'unmatched_det_indices':sorted(unmatched),'unmatched_gt_indices':np.flatnonzero(~used).tolist()}

def assert_equal(d,g):
 a=accelerated(d,g);b=greedy_match(d,g,iou_thr=.5,iou_metric='rotated')
 for key in ['unmatched_det_indices','unmatched_gt_indices']:assert a[key]==b[key],key
 assert len(a['matches'])==len(b['matches'])
 for x,y in zip(a['matches'],b['matches']):
  assert (x['det_index'],x['gt_index'])==(y['det_index'],y['gt_index'])
  assert abs(x['iou']-y['iou'])<1e-12,(x,y)

def load_group(p,kind):
 result=collections.defaultdict(list)
 validate=rio.validate_detection if kind=='det' else rio.validate_gt
 for i,line in enumerate(p.open()):
  row=validate(json.loads(line),tag=f'{kind}[{i}]');assert row.get('scene_id'),(p,i)
  result[row['image_id']].append(row)
 return result

def process(unit):
 name=unit['id'];collection=unit.get('collection','collected');src=HERE/collection/unit['relative'];out=HERE/'analysis'/name
 out.mkdir(parents=True,exist_ok=True)
 receipt=out/'match_receipt.json'
 if receipt.is_file():
  rec=json.loads(receipt.read_text())
  assert sha(out/'matched.jsonl')==rec['matched_sha256'] and sha(out/(name+'.npz'))==rec['cache_sha256']
  assert sha(src/'detections.jsonl')==rec['detections_sha256'] and sha(src/'ground_truth.jsonl')==rec['ground_truth_sha256']
  return rec
 t=time.monotonic()
 source=json.loads(sorted(HERE.glob('collection-source-*.json'))[-1].read_text())
 for n in ['detections.jsonl','ground_truth.jsonl']:
  assert sha(src/n)==source['files'][str((src/n).relative_to(HERE/collection))]
 dets=load_group(src/'detections.jsonl','det');gts=load_group(src/'ground_truth.jsonl','gt')
 images=sorted(set(dets)|set(gts))
 # Choose first, uniformly spaced, and medium-dense images deterministically.
 candidates=[i for i in images if dets.get(i) and gts.get(i)]
 picks=sorted(set(candidates[::max(1,len(candidates)//12)][:12]+sorted(candidates,key=lambda i:abs(len(gts[i])-12))[:3]))
 for i in picks:assert_equal(dets[i],gts[i])
 counts=collections.Counter();by_class=collections.defaultdict(collections.Counter)
 tmp=out/'matched.jsonl.partial'
 with tmp.open('w') as f:
  for image in images:
   d=dets.get(image,[]);g=gts.get(image,[]);scene=(d or g)[0]['scene_id']
   assert all(x['scene_id']==scene for x in d+g)
   match=accelerated(d,g)
   for m in match['matches']:
    di,gi=m['det_index'],m['gt_index'];row={'image_id':image,'scene_id':scene,'class':d[di]['class'],'pred_obb':d[di]['obb'],'gt_obb':g[gi]['obb'],'pred_score':d[di]['score'],'iou':m['iou'],'match_type':'tp'}
    f.write(json.dumps(row)+'\n');counts['tp']+=1;by_class[row['class']]['tp']+=1
   for di in match['unmatched_det_indices']:
    row={'image_id':image,'scene_id':scene,'class':d[di]['class'],'pred_obb':d[di]['obb'],'gt_obb':None,'pred_score':d[di]['score'],'iou':None,'match_type':'fp'}
    f.write(json.dumps(row)+'\n');counts['fp']+=1;by_class[row['class']]['fp']+=1
   for gi in match['unmatched_gt_indices']:
    row={'image_id':image,'scene_id':scene,'class':g[gi]['class'],'pred_obb':None,'gt_obb':g[gi]['obb'],'pred_score':None,'iou':None,'match_type':'fn'}
    f.write(json.dumps(row)+'\n');counts['fn']+=1;by_class[row['class']]['fn']+=1
 assert counts['tp']+counts['fp']==sum(map(len,dets.values()))
 assert counts['tp']+counts['fn']==sum(map(len,gts.values()))
 matched=out/'matched.jsonl';tmp.replace(matched)
 cache=out/(name+'.npz');build(str(matched),str(cache),scene_rule='image') # preserves explicit source scene_id
 rec={**unit,'counts':dict(counts),'by_class':dict(by_class),'recall':counts['tp']/(counts['tp']+counts['fn']),
      'precision_at_005':counts['tp']/(counts['tp']+counts['fp']),
      'source_scenes':len({x['scene_id'] for v in gts.values() for x in v}),
      'reference_matcher_parity_images':len(picks),'matched_sha256':sha(matched),'cache_sha256':sha(cache),
      'detections_sha256':sha(src/'detections.jsonl'),'ground_truth_sha256':sha(src/'ground_truth.jsonl'),
      'script_sha256':sha(__file__),'reference_matcher_sha256':sha(WORKTREE/'rotcert/matching.py'),
      'runtime_seconds':time.monotonic()-t,'scope':'exploratory frozen matching; crop-level counts; not mAP'}
 receipt.write_text(json.dumps(rec,indent=2)+'\n');return rec

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('--relative',required=True);a=p.parse_args()
 print(json.dumps(process({'id':a.id,'relative':a.relative})),flush=True)

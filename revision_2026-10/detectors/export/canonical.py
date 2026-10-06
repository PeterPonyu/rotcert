#!/usr/bin/env python3
# Only read trusted local framework pickle files: pickle can execute code.
import argparse
import json
import math
from pathlib import Path
import pickle
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CLASSES, canonical_obb, scene_id, sha256, atomic_json, write_jsonl, utc

def array(value):
    if hasattr(value, 'tensor'): value = value.tensor
    if hasattr(value, 'detach'): value = value.detach().cpu()
    return value.tolist() if hasattr(value, 'tolist') else value

def polygon_obb(coords):
    import cv2
    import numpy as np
    pts = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3 or not np.isfinite(pts).all(): raise ValueError('Invalid polygon')
    (x,y),(w,h),a = cv2.minAreaRect(pts)
    return canonical_obb([x,y,w,h,math.radians(a)])

def detection(image, label, box, score, dataset, mapping=None, offset=(0,0)):
    score = float(score)
    if not math.isfinite(score) or not 0 <= score <= 1: raise ValueError('Invalid score')
    classes = CLASSES[dataset]
    if isinstance(label, bool) or int(label) != label or not 0 <= int(label) < len(classes): raise ValueError('Invalid class index')
    box = polygon_obb(box) if len(box) == 8 else canonical_obb(box)
    box[0] += offset[0]; box[1] += offset[1]
    return {'image_id': image, 'scene_id': scene_id(image,dataset,mapping), 'class': classes[int(label)], 'obb': box, 'score': score}

def mmrotate_records(samples, dataset, mapping=None, score_thr=0.05, offset=(0,0)):
    seen=set()
    for sample in samples:
        if not isinstance(sample, dict):
            sample = sample.to_dict()
        meta = sample.get('metainfo', {})
        image = Path(str(sample.get('img_path', meta.get('img_path', sample.get('img_id', meta.get('img_id', '')))))).stem
        if not image or image in seen: raise ValueError('Missing or duplicate image id: '+image)
        seen.add(image)
        pred = sample['pred_instances']
        if not isinstance(pred,dict): pred=pred.to_dict()
        boxes, labels, scores = [array(pred[k]) for k in ('bboxes','labels','scores')]
        if not len(boxes)==len(labels)==len(scores): raise ValueError('Prediction length mismatch')
        for box,label,score in zip(boxes,labels,scores):
            rec = detection(image,label,box,score,dataset,mapping,offset)
            if rec['score'] >= score_thr: yield rec

def eav_records(samples, mapping=None, score_thr=0.05, topk=300):
    # Matches pinned RTDETRPostProcessor focal top-k, angle stays in radians
    import numpy as np
    seen=set()
    for sample in samples:
        image = str(sample['image_id'])
        if image in seen: raise ValueError('Duplicate EAV image')
        seen.add(image)
        boxes=np.asarray(array(sample['pred_boxes']),dtype=float)
        logits=np.asarray(array(sample['pred_logits']),dtype=float)
        if boxes.ndim!=2 or boxes.shape[1]!=5 or logits.shape!=(len(boxes),len(CLASSES['codrone'])): raise ValueError('Invalid EAV tensor shapes')
        if not np.isfinite(logits).all() or not np.isfinite(boxes).all(): raise ValueError('Nonfinite EAV predictions')
        h,w=sample['image_size_hw']
        if h<=0 or w<=0: raise ValueError('Invalid image size')
        probs=np.exp(-np.logaddexp(0,-logits)).reshape(-1)
        order=np.argsort(-probs,kind='stable')[:min(topk,len(probs))]
        for idx in order:
            label=int(idx%logits.shape[1]); query=int(idx//logits.shape[1])
            box=boxes[query].copy(); box[:4]*=[w,h,w,h]
            rec=detection(image,label,box.tolist(),probs[idx],'codrone',mapping)
            if rec['score']>=score_thr: yield rec

def write_export(records, output, provenance):
    required={'seed','config','checkpoint','data_manifest','command','start_utc','gpu','upstream_commit'}
    if not required<=provenance.keys(): raise ValueError('Incomplete provenance: '+repr(required-provenance.keys()))
    provenance=dict(provenance)
    for key in ('config','checkpoint','data_manifest'):
        p=Path(provenance[key]); provenance[key+'_sha256']=sha256(p)
    write_jsonl(output,records)
    provenance.update(end_utc=utc(),output_sha256=sha256(output),score_thr=0.05,geometry='le90 radians long-edge',scene_policy='audited map, known source rule, or explicit null')
    atomic_json(str(output)+'.provenance.json',provenance)

def main():
    p=argparse.ArgumentParser(description='Canonical export from trusted framework outputs')
    p.add_argument('--format',choices=['mmrotate','ai4rs','eavdetr'],required=True)
    p.add_argument('--dataset',choices=list(CLASSES),required=True)
    p.add_argument('--input',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--provenance',type=Path,required=True); p.add_argument('--scene-map',type=Path)
    p.add_argument('--coordinate-offset',nargs=2,type=float,default=[0,0],help='Explicit restoration offset to raw-image coordinates')
    a=p.parse_args(); mapping=json.loads(a.scene_map.read_text()) if a.scene_map else None
    if a.format=='eavdetr':
        if a.dataset!='codrone': p.error('Pinned EAV supports CODrone only')
        samples=[json.loads(line) for line in a.input.read_text().splitlines() if line]
        records=eav_records(samples,mapping)
    else:
        with a.input.open('rb') as f: samples=pickle.load(f)
        records=mmrotate_records(samples,a.dataset,mapping,offset=a.coordinate_offset)
    write_export(records,a.output,json.loads(a.provenance.read_text()))
if __name__=='__main__': main()

"""Shared I/O only; no detector frameworks imported at module load."""
import hashlib
import json
import math
import os
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
PINS = {'mmrot': '3ff004eb21ea040455b5585db229edba4037f1bf',
        'ai4rs': '6370b9bc27a2f7b4b76129c9182db87e74225d8c',
        'eavdetr': 'fb99da7b0f49edffcd1fa9e45a4424e9ae86e1eb'}
CLASSES = {
 'dota': ['plane','baseball-diamond','bridge','ground-track-field','small-vehicle','large-vehicle','ship','tennis-court','basketball-court','storage-tank','soccer-ball-field','roundabout','harbor','swimming-pool','helicopter'],
 'dior': ['airplane','airport','baseballfield','basketballcourt','bridge','chimney','expressway-service-area','expressway-toll-station','dam','golffield','groundtrackfield','harbor','overpass','ship','stadium','storagetank','tenniscourt','trainstation','vehicle','windmill'],
 'rsar': ['ship','aircraft','car','tank','bridge','harbor'], 'hrsid':['ship'],
 'dv': ['car','freight car','truck','bus','van'],
 'codrone':['car','truck','traffic-sign','people','motor','bicycle','traffic-light','tricycle','bridge','bus','boat','ship']}

def utc():
 return datetime.now(timezone.utc).isoformat()

def sha256(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()

def atomic_json(path, data):
 p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
 tmp.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n')
 tmp.replace(p)

def write_jsonl(path, records):
 p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
 with tmp.open('w') as f:
  for row in records: f.write(json.dumps(row,allow_nan=False)+'\n')
 tmp.replace(p)

def canonical_obb(box):
 if len(box)!=5: raise ValueError('Expected five OBB coordinates')
 x,y,w,h,a=map(float,box)
 if not all(math.isfinite(v) for v in (x,y,w,h,a)) or w<=0 or h<=0:
  raise ValueError('Nonfinite or nonpositive OBB')
 if w<h: w,h,a=h,w,a+math.pi/2
 return [x,y,w,h,(a+math.pi/2)%math.pi-math.pi/2]

def scene_id(image_id,dataset,scene_map=None):
 import re
 if scene_map is not None:
  if image_id not in scene_map: raise ValueError('Missing audited scene for '+image_id)
  value=scene_map[image_id]
  if not isinstance(value,str) or not value: raise ValueError('Empty audited scene for '+image_id)
  return value
 if dataset=='dota':
  m=re.fullmatch(r'(P[0-9]+)(?:__.*)?',image_id)
  if not m: raise ValueError('Unrecognized DOTA source scene: '+image_id)
  return m.group(1)
 if dataset=='codrone':
  m=re.fullmatch(r'(.+_(?:day|night|dusk|dawn)_[0-9]+m_[0-9]+c)_frame_[0-9]+(?:__.*)?',image_id)
  if not m: raise ValueError('Unrecognized CODrone sequence: '+image_id)
  return m.group(1)
 if dataset=='hrsid': raise ValueError('HRSID requires an audited image-to-panorama map')
 if dataset in ('rsar','dv'): return None
 return image_id

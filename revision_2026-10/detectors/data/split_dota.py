#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import atomic_json,PINS

def main():
    p=argparse.ArgumentParser(description='Run pinned DOTA splitter on train-only and val gaps 0/200/500')
    p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--mmrotate',type=Path,required=True); p.add_argument('--workers',type=int,default=8)
    a=p.parse_args()
    if subprocess.check_output(['git','-C',str(a.mmrotate),'rev-parse','HEAD'],text=True).strip()!=PINS['mmrot']: raise ValueError('Unexpected splitter revision')
    base=json.loads((a.mmrotate/'tools/data/dota/split/split_configs/ss_train.json').read_text())
    for split,gap,name in [('train',200,'train'),('val',0,'val0'),('val',200,'val200'),('val',500,'val500')]:
        dest=a.output/name
        if dest.exists(): raise ValueError('Refuse partial/existing split directory '+str(dest))
        cfg=dict(base,nproc=a.workers,img_dirs=[str((a.source/split/'images').resolve())],ann_dirs=[str((a.source/split/'labelTxt').resolve())],gaps=[gap],save_dir=str(dest.resolve()))
        config=a.output/('split_'+name+'.json'); atomic_json(config,cfg)
        subprocess.run([sys.executable,str(a.mmrotate/'tools/data/dota/split/img_split.py'),'--base-json',str(config)],check=True)
        if name=='val200' and len(list((dest/'images').glob('*.png')))!=5297: raise ValueError('DOTA val200 must contain 5297 tiles')
    print('Splits prepared; run stage_dota.sh for hashes, GT and DATA_OK.')
if __name__=='__main__': main()

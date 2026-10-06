#!/usr/bin/env python
"""Export EAV-DETR detections on the CODrone validation and test tiles for the RotCert comparison (2026-10-04).

One forward pass per tile with the official model, config and dataset class (CODroneDetection, no augmentation outside
training), loading the final-epoch checkpoint. Saved per tile: the tile name, its ground truth as the dataset class
builds it (normalized cx, cy, w, h, angle; labels; difficulty 0 only, as in the authors' loader), and every query whose
highest class probability is at least --min-score (normalized box, all twelve class probabilities). Shards by index
modulo --num-shards so that one process per GPU covers a split. No calibration happens here.

Usage: export_eav.py --config CFG --checkpoint CKPT --split val|test --shard i --num-shards n --out DIR [--limit N]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
from export_contract import CLASSES, sha256, validate_arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--split', choices=('val', 'test'), required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--min-score', type=float, default=0.05)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--expected-epoch', type=int, help='require checkpoint last_epoch to equal this value')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.num_shards < 1 or not 0 <= a.shard < a.num_shards or a.limit < 0:
        ap.error('require num_shards > 0, 0 <= shard < num_shards, limit >= 0')
    if not math.isfinite(a.min_score) or not 0 <= a.min_score <= 1:
        ap.error('min-score must be finite and in [0, 1]')
    import torch
    sys.path.insert(0, os.getcwd())
    from src.core import YAMLConfig
    os.makedirs(a.out, exist_ok=True)
    out_path = os.path.join(a.out, f'{a.split}-shard{a.shard:02d}-of{a.num_shards:02d}.npz')
    receipt_path = out_path.replace('.npz', '.json')
    if os.path.exists(out_path) or os.path.exists(receipt_path):
        raise SystemExit(f'{out_path} exists; refusing to overwrite')
    checkpoint_hash, config_hash = sha256(a.checkpoint), sha256(a.config)
    cfg = YAMLConfig(a.config)
    loader = cfg.val_dataloader if a.split == 'val' else cfg.test_dataloader
    ds = loader.dataset
    ds.max_samples = None
    ds.samples = ds._load_samples()                # full split, independent of the per-epoch monitoring limit
    if tuple(ds.CLASSES) != CLASSES or ds.patch_size != 1024 or ds.split != a.split:
        raise ValueError('dataset class order, patch size or split disagrees with the export contract')
    n_all = len(ds.samples)
    image_count = sum(os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg', '.png', '.bmp')
                      for f in os.listdir(ds.img_folder) if os.path.isfile(os.path.join(ds.img_folder, f)))
    if n_all == 0 or n_all != image_count:
        raise ValueError(f'full dataset inventory mismatch: loaded {n_all}, image files {image_count}')
    idx = [i for i in range(n_all) if i % a.num_shards == a.shard]
    if a.limit:
        idx = idx[:a.limit]
    model = cfg.model.cuda()
    ck = torch.load(a.checkpoint, map_location='cpu')
    checkpoint_epoch = ck.get('last_epoch')
    if a.expected_epoch is not None and checkpoint_epoch != a.expected_epoch:
        raise ValueError(f'expected checkpoint epoch {a.expected_epoch}, got {checkpoint_epoch}')
    state = ck['ema']['module'] if 'ema' in ck else ck.get('model', ck)
    model.load_state_dict(state)
    model.eval()
    names, gt_tile, gt_box, gt_lab, p_tile, p_box, p_prob = [], [], [], [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for k, i in enumerate(idx):
            img, tgt = ds[i]
            if not torch.isfinite(img).all() or not torch.isfinite(tgt['boxes']).all():
                raise ValueError(f'non-finite input or GT at tile {ds.samples[i]["img_name"]}')
            out = model(img[None].cuda())
            if not all(torch.isfinite(out[key]).all() for key in ('pred_logits', 'pred_boxes')):
                raise ValueError(f'non-finite model output at tile {ds.samples[i]["img_name"]}')
            prob = torch.sigmoid(out['pred_logits'][0]).float().cpu().numpy()
            box = out['pred_boxes'][0].float().cpu().numpy()
            keep = prob.max(1) >= a.min_score
            names.append(ds.samples[i]['img_name'])
            gt_tile.append(np.full(len(tgt['boxes']), k, np.int32)); gt_box.append(tgt['boxes'].numpy()); gt_lab.append(tgt['labels'].numpy())
            p_tile.append(np.full(int(keep.sum()), k, np.int32)); p_box.append(box[keep]); p_prob.append(prob[keep])
            if k % 2000 == 0:
                print(a.split, a.shard, k, len(idx), round(time.time() - t0, 1), flush=True)
    cat = lambda xs, shape, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(shape, dt)
    arrays = dict(tile_names=np.array(names, dtype=np.str_), tile_index=np.array(idx, np.int64),
                        gt_tile=cat(gt_tile, (0,), np.int32), gt_box=cat(gt_box, (0, 5), np.float32),
                        gt_label=cat(gt_lab, (0,), np.int64), pred_tile=cat(p_tile, (0,), np.int32),
                  pred_box=cat(p_box, (0, 5), np.float32), pred_prob=cat(p_prob, (0, 12), np.float32))
    validate_arrays(arrays, a.min_score)
    if sha256(a.checkpoint) != checkpoint_hash or sha256(a.config) != config_hash:
        raise ValueError('checkpoint or config changed during inference; use an immutable checkpoint copy')
    with open(out_path, 'xb') as f:
        np.savez_compressed(f, **arrays)
    rec = dict(schema='rotcert.eav-export.v2', split=a.split, shard=a.shard, num_shards=a.num_shards, tiles=len(idx),
               split_tiles=n_all, limit=a.limit, min_score=a.min_score, checkpoint=a.checkpoint,
               checkpoint_sha256=checkpoint_hash, checkpoint_epoch=checkpoint_epoch,
               weights='ema' if 'ema' in ck else 'model', config=a.config, config_sha256=config_hash,
               class_names=list(CLASSES), box_format='normalized-cxcywh-angle-radians',
               script_sha256=sha256(__file__), contract_sha256=sha256(os.path.join(os.path.dirname(__file__), 'export_contract.py')),
               out=out_path, out_sha256=sha256(out_path), seconds=time.time() - t0, torch=torch.__version__)
    with open(receipt_path, 'x') as f:
        json.dump(rec, f, indent=1, allow_nan=False)
    print(json.dumps(rec))


if __name__ == '__main__':
    main()

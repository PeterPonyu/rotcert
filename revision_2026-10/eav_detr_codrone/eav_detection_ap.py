#!/usr/bin/env python
"""Detection accuracy of the reproduced EAV-DETR from the accepted final exports, scored with the authors' released
evaluator (src/data/codrone/codrone_eval.py at commit fb99da7b; 2026-10-05). Runs on the box with the EAV venv.

The authors' evaluation keeps, per tile, the 300 highest sigmoid scores over all (query, class) pairs (their
RTDETRPostProcessor), maps the normalized boxes to 1024-pixel tiles and computes all-point AP per class at IoU
0.50:0.05:0.95 with at most 100 detections per tile and class. The exports keep every query whose highest class score is
at least 0.05, with all twelve scores, so every pair scoring at least 0.05 is recovered; pairs below 0.05 that the
authors' top-300 could still include are absent. --check-monitoring scores the first 2,800 validation tiles, the subset
of the per-epoch monitoring evaluation, so that the logged final-epoch values test this reconstruction.

Usage: eav_detection_ap.py --exports DIR --eav-repo DIR --split val|test [--first N] --out FILE [--workers 12]
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

TILE = 1024.0
TOP = 300


class FastIds(list):
    """List with set-backed membership; same contents and order as the evaluator's own list."""

    def __init__(self):
        super().__init__()
        self._seen = set()

    def append(self, x):
        self._seen.add(x)
        super().append(x)

    def __contains__(self, x):
        return x in self._seen


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exports', required=True)
    ap.add_argument('--eav-repo', required=True)
    ap.add_argument('--split', choices=('val', 'test'), required=True)
    ap.add_argument('--first', type=int, default=0, help='score only tiles with index < N (0: all)')
    ap.add_argument('--stride', type=int, default=1, help='keep tiles with index %% STRIDE == OFFSET (rank share check)')
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit('output exists')
    sys.path.insert(0, a.eav_repo)
    from src.data.codrone.codrone_eval import CODroneEvaluator
    t0 = time.time()
    files = sorted(f for f in os.listdir(a.exports) if f.startswith(a.split + '-shard') and f.endswith('.npz'))
    per_tile = {}
    for f in files:
        with np.load(os.path.join(a.exports, f), allow_pickle=False) as z:
            idx, gt_tile, gt_box, gt_label = z['tile_index'], z['gt_tile'], z['gt_box'], z['gt_label']
            pred_tile, pred_box, pred_prob = z['pred_tile'], z['pred_box'], z['pred_prob']
        for local, tile in enumerate(idx.tolist()):
            if (a.first and tile >= a.first) or tile % a.stride != a.offset:
                continue
            per_tile[tile] = dict(gt=(gt_box[gt_tile == local], gt_label[gt_tile == local]),
                                  pred=(pred_box[pred_tile == local], pred_prob[pred_tile == local]))
    tiles = sorted(per_tile)                               # the evaluator's image id is the position in this list
    names = ['car', 'truck', 'traffic-sign', 'people', 'motor', 'bicycle', 'traffic-light', 'tricycle', 'bridge', 'bus',
             'boat', 'ship']
    ev = CODroneEvaluator(class_names=names, iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                          maxDets=100, num_workers=a.workers)
    ev.img_ids = FastIds()                                  # speed only: the evaluator tests membership in a list
    annotations, kept_pairs = [], 0
    for position, tile in enumerate(tiles):
        gb, gl = per_tile[tile]['gt']
        g = gb.astype(np.float32).copy()
        g[:, :4] *= TILE
        annotations.append(dict(boxes=g, labels=gl.astype(np.int64)))
        pb, pp = per_tile[tile]['pred']
        q, c = np.nonzero(pp >= 0.05)                       # every recoverable (query, class) pair
        s = pp[q, c]
        order = np.argsort(-s, kind='stable')[:TOP]
        q, c, s = q[order], c[order], s[order]
        b = pb[q].astype(np.float32).copy()
        b[:, :4] *= TILE
        kept_pairs += len(s)
        ev.update({position: dict(boxes=b.reshape(-1, 5), labels=c.astype(np.int64), scores=s.astype(np.float32))})
    ev.add_annotations(annotations)
    ev.accumulate()
    res = ev.summarize()
    out = dict(schema='rotcert.eav-detection-ap.v1', split=a.split, tiles=len(tiles), first=a.first or None,
               stride=a.stride, offset=a.offset,
               pairs_scored=int(kept_pairs), score_floor_of_exports=0.05, top_pairs_per_tile=TOP, max_dets_per_class=100,
               iou_thresholds='0.50:0.05:0.95', ap='all-point interpolation per class, mean over classes',
               mAP=float(res['mAP']), AP50=float(res['mAP@0.50']), AP75=float(res['mAP@0.75']),
               per_class_AP50={n: float(res[f'{n}_AP@0.50']) for n in names},
               per_class_AP75={n: float(res[f'{n}_AP@0.75']) for n in names},
               evaluator=os.path.join(a.eav_repo, 'src/data/codrone/codrone_eval.py'),
               evaluator_sha256=sha256(os.path.join(a.eav_repo, 'src/data/codrone/codrone_eval.py')),
               inputs={f: sha256(os.path.join(a.exports, f)) for f in files}, script_sha256=sha256(__file__),
               seconds=time.time() - t0)
    with open(a.out, 'x') as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ('split', 'tiles', 'pairs_scored', 'mAP', 'AP50', 'AP75', 'seconds')}))


if __name__ == '__main__':
    main()

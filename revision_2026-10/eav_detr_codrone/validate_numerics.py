#!/usr/bin/env python3
"""Nonempty, deterministic numerical checks against released EAV and frozen S1."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from analyze_eav import greedy_match, hcp_by_class, margins, native_match, q_conformal, to_px
from export_contract import sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--eav-repo', required=True)
    ap.add_argument('--s1', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    if Path(args.out).exists():
        ap.error('output already exists')
    sys.path.insert(0, args.eav_repo)
    sys.path.insert(0, str(Path(args.eav_repo) / 'tools'))
    sys.path.insert(0, args.s1)
    import torch
    import calibrate
    import core
    from vendor.gwd import obb_gwd
    from src.zoo.eavdetr.oriented_ops import oriented_box_iou

    gt = np.array([[.2, .2, .16, .06, .2], [.5, .5, .18, .08, -.3],
                   [.8, .8, .14, .06, .4]], dtype=np.float32)
    pred = np.vstack([gt, gt[0], [0.05, 0.8, .1, .05, 0]]).astype(np.float32)
    pred[:3, 0] += .01
    prob = np.full((5, 12), .0001, dtype=np.float32)
    prob[np.arange(5), [0, 1, 0, 0, 0]] = [.99, .9, .95, .5, .4]
    data = dict(names=['synthetic_day_30m_30c_frame_0__1__0___0'], gt_tile=np.zeros(3, np.int32),
                gt_box=gt, gt_label=np.array([0, 1, 2]), pred_tile=np.zeros(5, np.int32),
                pred_box=pred, pred_prob=prob)
    actual_pairs = native_match(data, oriented_box_iou)
    official = calibrate.ConformalCalibrator.__new__(calibrate.ConformalCalibrator)
    boxes = torch.tensor(pred).cuda()
    targets = torch.tensor(gt).cuda()
    probabilities = torch.tensor(prob).cuda()
    cost, high = official.build_cost_matrix(boxes, probabilities.max(-1).values, targets)
    official_pairs, _ = official.hungarian_matching(cost)
    indices = high.nonzero().flatten().cpu().numpy()
    expected_pairs = np.array([(indices[p], g) for p, g in official_pairs])
    np.testing.assert_array_equal(actual_pairs, expected_pairs)
    np.testing.assert_array_equal(actual_pairs, [[0, 0], [1, 1], [2, 2]])
    actual_margin = margins(data, actual_pairs, args.eav_repo, 2)
    logits = torch.logit(probabilities)
    official_margin = official.process_single_image(
        {'pred_logits': logits[None], 'pred_boxes': boxes[None]},
        {'boxes': targets, 'labels': torch.tensor([0, 1, 2]).cuda()}, (1024, 1024))
    np.testing.assert_array_equal(actual_margin, official_margin)
    if not np.all(actual_margin > 0):
        raise AssertionError('nonempty verification must exercise positive expansion margins')
    protocol_pairs, false_positives = greedy_match(data)
    np.testing.assert_array_equal(protocol_pairs, [[0, 0], [1, 1]])
    np.testing.assert_array_equal(false_positives, [3] + [0] * 11)
    pixels = to_px(gt)
    translated = pixels.copy()
    translated[:, 0] += 6
    np.testing.assert_allclose(obb_gwd(pixels, translated), 6, atol=1e-8, rtol=0)
    np.testing.assert_allclose(obb_gwd(pixels, pixels), 0, atol=1e-5, rtol=0)
    swapped = pixels.copy()
    swapped[:, [2, 3]] = swapped[:, [3, 2]]
    swapped[:, 4] += np.pi / 2
    np.testing.assert_allclose(obb_gwd(pixels, swapped), 0, atol=1e-5, rtol=0)
    scores = np.arange(1., 11.)
    q = hcp_by_class(core, np.zeros(10, int), np.arange(10), scores, 10)
    if q[0] != 10 or not all(np.isinf(q[k]) for k in range(1, 12)):
        raise AssertionError('frozen source-uniform HCP threshold/support mismatch')
    if q_conformal(np.arange(9)) != 8 or not np.isinf(q_conformal(np.arange(8))):
        raise AssertionError('finite-sample rank boundary mismatch')
    np.testing.assert_allclose(core.angle_halfwidth(np.array([10., 10.]), np.array([10., 5.]), 0), [np.pi/2, 0])
    files = [Path(args.eav_repo) / 'tools/calibrate.py', Path(args.eav_repo) / 'tools/geometry_utils.py',
             Path(args.eav_repo) / 'src/zoo/eavdetr/oriented_ops.py', Path(args.s1) / 'core.py',
             Path(args.s1) / 'vendor/gwd.py', Path(__file__), Path(__file__).with_name('analyze_eav.py'),
             Path(__file__).with_name('export_contract.py')]
    result = dict(schema='rotcert.eav-numerical-validation.v1', passed=True,
                  native_pairs=actual_pairs.tolist(), native_margin_px=actual_margin.tolist(),
                  native_pair_equality='exact', native_margin_equality='exact float32',
                  protocol_pairs=protocol_pairs.tolist(), protocol_false_positives=false_positives.tolist(),
                  checks=['native matching equals released calibrator', 'positive margin equals released geometry',
                          'class-exact greedy TP/FP accounting', 'GWD center-translation distance',
                          'GWD identity and equivalent side/angle representation',
                          'frozen HCP finite/infinite support boundaries', 'corrected rank and square angle convention'],
                  inputs={str(p): sha256(p) for p in files}, torch=torch.__version__)
    with Path(args.out).open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()

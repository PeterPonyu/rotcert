"""Shared synthetic-data fixtures for rotcert tests. No network, no GPU, no mmrotate."""

from __future__ import annotations

import numpy as np
import pytest


def _gen_matched_pairs(rng, n, seam_frac=0.0, square_frac=0.0, noise_scale=(1.0, 1.0, 1.0, 0.5, 0.05)):
    """Synthetic (pred_obb, gt_obb) pairs. ``seam_frac``/``square_frac`` control how
    many GT angles are drawn near the +-90deg seam / how many boxes are near-square."""
    preds, gts = [], []
    for _ in range(n):
        cx, cy = rng.uniform(0, 200, 2)
        r = rng.random()
        if r < square_frac:
            w = rng.uniform(15, 25)
            h = w * rng.uniform(0.92, 1.0)
        else:
            w = rng.uniform(15, 30)
            h = rng.uniform(4, 12)
        if rng.random() < seam_frac:
            theta_gt = rng.choice([1.0, -1.0]) * (np.pi / 2 - rng.uniform(0.0, 0.05))
        else:
            theta_gt = rng.uniform(-np.pi / 2, np.pi / 2)
        gt = np.array([cx, cy, w, h, theta_gt])
        noise = rng.normal(scale=noise_scale)
        pred = gt + noise
        pred[2] = max(pred[2], 0.5)
        pred[3] = max(pred[3], 0.5)
        preds.append(pred)
        gts.append(gt)
    return np.array(preds), np.array(gts)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def gen_matched_pairs():
    return _gen_matched_pairs


@pytest.fixture
def synthetic_matched_records(rng):
    """List-of-dict matched TP records (pred_obb, gt_obb, class, scene_id), the
    ``rotcert.certify``/``rotcert.audit`` input shape."""
    preds, gts = _gen_matched_pairs(rng, 400)
    classes = rng.choice(["ship", "harbor", "plane"], size=400)
    scenes = rng.integers(0, 40, size=400)
    out = []
    for p, g, c, s in zip(preds, gts, classes, scenes):
        out.append({"pred_obb": p.tolist(), "gt_obb": g.tolist(), "class": str(c), "scene_id": f"P{s:04d}"})
    return out

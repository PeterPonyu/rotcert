"""Extended nonconformity scores for oriented boxes (revision spec §5.4).

All functions are vectorized over rows of ``(N, 5)`` arrays ``(cx, cy, w, h, theta)``
(any angle convention; canonicalized to le90 here). Gaussian-based scores floor
the sides at ``MIN_SIDE`` pixels so covariances stay non-singular.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

from rotcert.gwd import canonicalize_le90, obb_to_gaussian

__all__ = [
    "MIN_SIDE", "kld", "hellinger", "bhattacharyya", "corner_max", "expansion_margin",
    "coord_max_additive", "coord_max_multiplicative", "wrapped_coord_residuals",
    "MaxRank", "EXT_SCORES",
]

MIN_SIDE = 1.0


def _canon(obb):
    obb = np.atleast_2d(np.asarray(obb, dtype=float))
    w, h, t = canonicalize_le90(obb[:, 2], obb[:, 3], obb[:, 4])
    return obb[:, 0], obb[:, 1], w, h, t


def _gauss(obb):
    cx, cy, w, h, t = _canon(obb)
    return obb_to_gaussian(cx, cy, np.maximum(w, MIN_SIDE), np.maximum(h, MIN_SIDE), t, canonicalize=False)


def kld(pred, gt):
    """KL(N_pred || N_gt) (Yang et al., NeurIPS 2021; GT as reference)."""
    mp, sp = _gauss(pred)
    mg, sg = _gauss(gt)
    sg_inv = np.linalg.inv(sg)
    d = (mp - mg)[..., None]
    maha = (np.swapaxes(d, -1, -2) @ sg_inv @ d)[..., 0, 0]
    tr = np.trace(sg_inv @ sp, axis1=-2, axis2=-1)
    logdet = np.log(np.linalg.det(sg)) - np.log(np.linalg.det(sp))
    return np.maximum(0.5 * (maha + tr + logdet) - 1.0, 0.0)


def bhattacharyya(pred, gt):
    mp, sp = _gauss(pred)
    mg, sg = _gauss(gt)
    s = 0.5 * (sp + sg)
    d = (mp - mg)[..., None]
    maha = (np.swapaxes(d, -1, -2) @ np.linalg.inv(s) @ d)[..., 0, 0]
    ld = np.log(np.linalg.det(s)) - 0.5 * (np.log(np.linalg.det(sp)) + np.log(np.linalg.det(sg)))
    return np.maximum(0.125 * maha + 0.5 * ld, 0.0)


def hellinger(pred, gt):
    """Hellinger distance between box Gaussians (ProbIoU = 1 - hellinger; Llerena et al.)."""
    return np.sqrt(np.clip(1.0 - np.exp(-bhattacharyya(pred, gt)), 0.0, 1.0))


def _corners(obb):
    cx, cy, w, h, t = _canon(obb)
    c, s = np.cos(t), np.sin(t)
    lx = np.array([-0.5, 0.5, 0.5, -0.5])[None, :] * w[:, None]
    ly = np.array([-0.5, -0.5, 0.5, 0.5])[None, :] * h[:, None]
    x = cx[:, None] + c[:, None] * lx - s[:, None] * ly
    y = cy[:, None] + s[:, None] * lx + c[:, None] * ly
    return np.stack([x, y], axis=-1)  # (N, 4, 2)


def corner_max(pred, gt):
    """Order-invariant max corner distance: min over cyclic correspondences of max_i |c_i - c'_{i+k}|."""
    cp, cg = _corners(pred), _corners(gt)
    best = None
    for k in range(4):
        d = np.linalg.norm(cp - np.roll(cg, k, axis=1), axis=-1).max(axis=1)
        best = d if best is None else np.minimum(best, d)
    return best


def expansion_margin(pred, gt):
    """EAV-DETR containment score: minimal outward offset of the predicted box's edges so that
    all ground-truth corners are inside (faithful to their tools/geometry_utils.py)."""
    cx, cy, w, h, t = _canon(pred)
    cg = _corners(gt)
    rx = cg[..., 0] - cx[:, None]
    ry = cg[..., 1] - cy[:, None]
    c, s = np.cos(t)[:, None], np.sin(t)[:, None]
    lx = c * rx + s * ry
    ly = -s * rx + c * ry
    need = np.maximum(np.abs(lx) - w[:, None] / 2.0, np.abs(ly) - h[:, None] / 2.0)
    return np.maximum(need.max(axis=1), 0.0)


def _wrapped(dt):
    d = np.mod(np.abs(dt), math.pi)
    return np.minimum(d, math.pi - d)


def _wrapped_residuals_and_pred_wh(pred, gt):
    """Shared computation behind :func:`wrapped_coord_residuals` and the two ``coord_max_*``
    scores: canonicalizes ``pred``/``gt`` exactly once each and returns ``(residuals, pw, ph)``
    -- ``pw``/``ph`` are the predicted box's own RAW (unfloored) canonical half-... sides, so
    callers that need them (the ``coord_max_*`` scores) don't have to re-canonicalize ``pred``."""
    px, py, pw, ph, pt = _canon(pred)
    gx, gy, gw, gh, gt_ = _canon(gt)
    c, s = np.cos(pt), np.sin(pt)
    dx, dy = gx - px, gy - py
    r = np.column_stack([np.abs(c * dx + s * dy), np.abs(-s * dx + c * dy),
                         np.abs(gw - pw), np.abs(gh - ph), _wrapped(gt_ - pt)])
    return r, pw, ph


def wrapped_coord_residuals(pred, gt):
    """(|dx'|, |dy'|, |dw|, |dh|, wrapped |dtheta|) with center offsets in the prediction frame."""
    r, _, _ = _wrapped_residuals_and_pred_wh(pred, gt)
    return r


def coord_max_additive(pred, gt):
    """Max-additive coordinate margin (after Andeol et al. 2023), angle as arc length at the box end."""
    r, pw, _ = _wrapped_residuals_and_pred_wh(pred, gt)
    return np.max(np.column_stack([r[:, 0], r[:, 1], r[:, 2], r[:, 3], 0.5 * pw * r[:, 4]]), axis=1)


def coord_max_multiplicative(pred, gt):
    """Max-multiplicative coordinate margin (after Andeol et al. 2023): size-relative residuals."""
    r, pw, ph = _wrapped_residuals_and_pred_wh(pred, gt)
    pw, ph = np.maximum(pw, MIN_SIDE), np.maximum(ph, MIN_SIDE)
    return np.max(np.column_stack([r[:, 0] / pw, r[:, 1] / ph, r[:, 2] / pw, r[:, 3] / ph, 0.5 * r[:, 4]]), axis=1)


@dataclass
class MaxRank:
    """Max of per-coordinate empirical CDFs fitted on the DESIGN split (Timans et al.), so the
    resulting scalar score is a fixed function when it is later calibrated on other scenes."""
    sorted_design: np.ndarray  # (n, d), each column sorted

    @classmethod
    def fit(cls, design_residuals: np.ndarray) -> "MaxRank":
        d = np.sort(np.asarray(design_residuals, dtype=float), axis=0)
        return cls(sorted_design=d)

    def score(self, residuals: np.ndarray) -> np.ndarray:
        r = np.atleast_2d(np.asarray(residuals, dtype=float))
        n = self.sorted_design.shape[0]
        cdf = np.column_stack([np.searchsorted(self.sorted_design[:, j], r[:, j], side="left") / n
                               for j in range(r.shape[1])])
        return cdf.max(axis=1)


EXT_SCORES: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "kld": kld,
    "hellinger": hellinger,
    "bhattacharyya": bhattacharyya,
    "corner_max": corner_max,
    "expansion_margin": expansion_margin,
    "coord_max_additive": coord_max_additive,
    "coord_max_multiplicative": coord_max_multiplicative,
}

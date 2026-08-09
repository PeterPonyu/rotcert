"""Rotated-IoU and the preregistered greedy detection<->GT matching rule (design §4.2).

Rotated IoU via shapely (pure python, no mmcv/mmrotate C-extensions)
----------------------------------------------------------------------
Each OBB ``(cx, cy, w, h, theta)`` is turned into its 4-corner polygon (le90-canonical
before corner construction, so the polygon is always well-formed even for degenerate
near-zero-area inputs) and intersected with :mod:`shapely`. ``rotated_iou`` is the
primary metric; ``hull_iou`` (axis-aligned bounding box of each polygon, THEN IoU) is
the anti-conservative G2 diagnostic the design calls for (§2.4, §4.2) -- never used to
drive the primary certificate.

The matching rule (preregistered, exact -- design §4.2, verbatim)
------------------------------------------------------------------------
Per image: sort predictions by confidence DESCENDING; match each to the
highest-rotated-IoU UNMATCHED ground-truth box of the SAME CLASS with IoU >= tau; each
GT matches at most one prediction; unmatched GTs are misses (feed G2); unmatched
predictions are false positives (excluded from G1, counted in the report). Deterministic
tie-breaks: confidence ties broken by ascending detection index (stable sort); IoU ties
among candidate GTs broken by ascending GT index. ``tau`` defaults to 0.5 with the
preregistered sensitivity sweep ``{0.5, 0.6, 0.7}`` (§4.2); the ``iou_metric`` switch
(``"rotated"`` default, ``"hull"`` diagnostic) makes the axis-aligned-hull baseline run
through the exact same matching code path as the primary -- apples-to-apples by
construction (design §3.1's ``--score naive-coord``/``hull`` rationale, extended here to
matching).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
from shapely.geometry import Polygon

from rotcert.gwd import canonicalize_le90

__all__ = ["obb_to_polygon", "rotated_iou", "hull_iou", "iou_matrix", "greedy_match"]


def obb_to_polygon(obb: Sequence[float], canonicalize: bool = True) -> Polygon:
    """4-corner :class:`shapely.geometry.Polygon` for one OBB ``(cx,cy,w,h,theta)``."""
    cx, cy, w, h, theta = (float(v) for v in obb)
    if canonicalize:
        w_arr, h_arr, theta_arr = canonicalize_le90(np.array(w), np.array(h), np.array(theta))
        w, h, theta = float(w_arr), float(h_arr), float(theta_arr)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx, dy = w / 2.0, h / 2.0
    # Corners in the box's local frame, then rotate + translate.
    local = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    world = local @ R.T + np.array([cx, cy])
    return Polygon(world)


def _safe_iou(poly_a: Polygon, poly_b: Polygon) -> float:
    if not poly_a.is_valid:
        poly_a = poly_a.buffer(0)
    if not poly_b.is_valid:
        poly_b = poly_b.buffer(0)
    area_a, area_b = poly_a.area, poly_b.area
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    inter = poly_a.intersection(poly_b).area
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def rotated_iou(obb_a: Sequence[float], obb_b: Sequence[float]) -> float:
    """Rotated (true polygon) IoU between two OBBs. The PRIMARY match metric."""
    return _safe_iou(obb_to_polygon(obb_a), obb_to_polygon(obb_b))


def hull_iou(obb_a: Sequence[float], obb_b: Sequence[float]) -> float:
    """Axis-aligned-BOUNDING-BOX IoU: each OBB's polygon is replaced by its axis-aligned
    envelope (``.envelope``) BEFORE intersecting. This is the G2 anti-conservatism
    diagnostic (design §2.4/§4.2).

    NOTE on the "hull_iou >= rotated_iou" anti-conservatism claim: this is NOT a
    per-pair inequality (a counterexample is trivial: an axis-aligned box against a
    45-degree-rotated box of the same size has a SMALLER hull IoU than rotated IoU,
    because only one side's footprint inflates). The design's anti-conservatism claim
    is about the REALISTIC matching regime -- a true-positive (prediction, GT) pair for
    the SAME physical object, hence similarly oriented -- where both polygons' hulls
    inflate by comparable amounts and the match is easier to sustain under hull-IoU
    than under rotated-IoU; :mod:`rotcert.certify`'s G2 anti-conservatism diagnostic
    verifies this EMPIRICALLY (aggregate recall, hull match vs rotated match) rather
    than assuming a universal per-pair bound. Never the primary matching metric."""
    poly_a = obb_to_polygon(obb_a).envelope
    poly_b = obb_to_polygon(obb_b).envelope
    return _safe_iou(poly_a, poly_b)


def iou_matrix(
    preds: np.ndarray, gts: np.ndarray, iou_metric: str = "rotated"
) -> np.ndarray:
    """Dense ``(n_pred, n_gt)`` IoU matrix for one image (one class already filtered
    upstream, or mixed -- caller decides; :func:`greedy_match` applies the class
    constraint separately from IoU)."""
    if iou_metric not in ("rotated", "hull"):
        raise ValueError(f"iou_matrix: iou_metric must be 'rotated' or 'hull', got {iou_metric!r}")
    fn = rotated_iou if iou_metric == "rotated" else hull_iou
    preds = np.asarray(preds, dtype=float)
    gts = np.asarray(gts, dtype=float)
    n_pred, n_gt = len(preds), len(gts)
    mat = np.zeros((n_pred, n_gt), dtype=float)
    for i in range(n_pred):
        for j in range(n_gt):
            mat[i, j] = fn(preds[i], gts[j])
    return mat


def greedy_match(
    dets: List[Dict[str, Any]],
    gts: List[Dict[str, Any]],
    iou_thr: float = 0.5,
    iou_metric: str = "rotated",
) -> Dict[str, Any]:
    """The preregistered greedy one-to-one matching rule for ONE IMAGE (design §4.2).

    Parameters
    ----------
    dets:
        List of detection dicts, each with keys ``obb`` (5-tuple), ``score`` (float,
        confidence), ``class`` (hashable), and any passthrough id (e.g. ``det_id``).
    gts:
        List of GT dicts, each with ``obb``, ``class``, passthrough id (e.g. ``gt_id``).
    iou_thr:
        Match threshold ``tau`` (default 0.5; sweep ``{0.5, 0.6, 0.7}`` per design §4.2).
    iou_metric:
        ``"rotated"`` (primary) or ``"hull"`` (G2 anti-conservatism diagnostic).

    Returns
    -------
    dict
        ``matches``: list of ``{"det_index", "gt_index", "iou"}`` (indices into ``dets``
        / ``gts`` as given). ``unmatched_det_indices``: false positives. ``unmatched_gt_
        indices``: misses (feed G2). Deterministic: detections processed in descending-
        score order (stable sort on ``-score`` then original index); each detection's
        candidate GT is the highest-IoU unmatched GT of the same class at ``IoU >=
        iou_thr`` (ties broken by ascending GT index, via the stable sort of GT index
        used as the tertiary key).
    """
    if not 0.0 <= iou_thr <= 1.0:
        raise ValueError("greedy_match: iou_thr must be in [0, 1]")
    n_det, n_gt = len(dets), len(gts)

    det_order = sorted(range(n_det), key=lambda i: (-float(dets[i]["score"]), i))
    gt_matched = [False] * n_gt

    matches: List[Dict[str, Any]] = []
    unmatched_det: List[int] = []

    for di in det_order:
        d = dets[di]
        best_iou = -1.0
        best_gj = -1
        for gj in range(n_gt):
            if gt_matched[gj]:
                continue
            if gts[gj]["class"] != d["class"]:
                continue
            iou = (rotated_iou if iou_metric == "rotated" else hull_iou)(d["obb"], gts[gj]["obb"])
            if iou >= iou_thr and (iou > best_iou or (iou == best_iou and (best_gj == -1 or gj < best_gj))):
                best_iou = iou
                best_gj = gj
        if best_gj >= 0:
            gt_matched[best_gj] = True
            matches.append({"det_index": di, "gt_index": best_gj, "iou": float(best_iou)})
        else:
            unmatched_det.append(di)

    # Restore matches/unmatched-det to original detection-index ascending order for
    # deterministic, order-independent downstream consumption.
    matches.sort(key=lambda m: m["det_index"])
    unmatched_det.sort()
    unmatched_gt = sorted(gj for gj in range(n_gt) if not gt_matched[gj])

    return {
        "matches": matches,
        "unmatched_det_indices": unmatched_det,
        "unmatched_gt_indices": unmatched_gt,
        "iou_thr": float(iou_thr),
        "iou_metric": iou_metric,
        "n_det": n_det,
        "n_gt": n_gt,
    }

"""Runtime guard for using a prefix-stable matcher at confidence thresholds.

The caller supplies the audited matcher; no import or modification of a shared
worktree occurs. This guard makes the finite-score/order/box prerequisites
explicit. SHA pin the matcher alongside all generated risk records.
"""
from __future__ import annotations
import math
import numpy as np


def validate_matching_inputs(dets, gts, lam, iou_thr=.5, iou_metric="rotated"):
    if not math.isfinite(float(lam)) or not 0 <= lam <= 1:
        raise ValueError("confidence lambda must be finite and in [0,1]")
    if not math.isfinite(float(iou_thr)) or not 0 <= iou_thr <= 1:
        raise ValueError("IoU threshold must be finite and in [0,1]")
    if iou_metric not in ("rotated", "hull"):
        raise ValueError("unknown matching IoU metric")
    for record in list(dets)+list(gts):
        obb = np.asarray(record["obb"], dtype=float)
        if obb.shape != (5,) or not np.all(np.isfinite(obb)) or min(obb[2:4]) <= 0:
            raise ValueError("matching requires a finite nondegenerate OBB")
        hash(record["class"])
    for record in dets:
        score = float(record["score"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("finite confidence in [0,1] required for prefix stability")


def rematch_at_lambda(dets, gts, lam, *, matcher, iou_thr=.5, iou_metric="rotated"):
    """Filter in original input order; restore original detection IDs in output."""
    validate_matching_inputs(dets, gts, lam, iou_thr, iou_metric)
    retained = [i for i,d in enumerate(dets) if float(d["score"]) >= lam]
    result = matcher([dets[i] for i in retained], gts, iou_thr=iou_thr, iou_metric=iou_metric)
    return {**result,
            "matches": [{**m,"det_index":retained[m["det_index"]]} for m in result["matches"]],
            "unmatched_det_indices": [retained[i] for i in result["unmatched_det_indices"]],
            "retained_detection_indices": retained, "confidence_operator": ">="}

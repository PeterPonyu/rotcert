"""End-to-end detection + localization certificate (revision spec §3 C3).

Per scene with n GT boxes, L(q) = 1 - #{GT detected at confidence >= lam AND localized with
score <= q} / n. L is non-increasing in q, so fixed-sequence testing from the largest q
downward (each at level delta) controls the error at delta without multiplicity correction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from rotcert.ltt import eb_pvalue, hb_pvalue

__all__ = ["FIXED_Q_GRID_PX", "e2e_loss_matrix", "certify_e2e", "composition_bound"]

FIXED_Q_GRID_PX = np.geomspace(0.25, 512.0, 100)
_PV = {"eb": eb_pvalue, "hb": hb_pvalue}


def e2e_loss_matrix(scene_records: List[Tuple[np.ndarray, np.ndarray]], q_grid: Sequence[float], lam: float) -> np.ndarray:
    q = np.asarray(list(q_grid), dtype=float)
    rows = []
    for conf, score in scene_records:
        conf = np.asarray(conf, dtype=float)
        score = np.asarray(score, dtype=float)
        if conf.size == 0:
            continue
        det = np.nan_to_num(conf, nan=-np.inf) >= lam
        score_filled = np.nan_to_num(score, nan=np.inf)
        # AND with det explicitly: misses use +inf as a score sentinel, and
        # inf <= inf is True, so comparing the sentinel alone would falsely
        # count an all-miss scene as "localized" at q = +inf.
        ok = det[None, :] & (score_filled[None, :] <= q[:, None])
        rows.append(1.0 - ok.sum(axis=1) / conf.size)
    return np.array(rows, dtype=float).reshape(-1, q.size)


def certify_e2e(loss_matrix: np.ndarray, q_grid: Sequence[float], eps: float, delta: float, p_value: str = "eb") -> Dict[str, Any]:
    lm = np.asarray(loss_matrix, dtype=float)
    q = np.asarray(list(q_grid), dtype=float)
    if lm.ndim != 2 or lm.shape[1] != q.size:
        raise ValueError("loss_matrix must be (n_scenes, len(q_grid))")
    order = np.argsort(q)
    q, lm = q[order], lm[:, order]
    pf = _PV[p_value]
    trace, star = [], None
    for k in range(q.size - 1, -1, -1):
        p = pf(lm[:, k], eps)
        rej = bool(p <= delta)
        trace.append({"q": float(q[k]), "p_value": float(p), "rejected": rej, "mean_loss": float(lm[:, k].mean())})
        if not rej:
            break
        star = k
    return {"certified": star is not None, "q_star": None if star is None else float(q[star]),
            "realized_loss": None if star is None else float(lm[:, star].mean()),
            "eps": float(eps), "delta": float(delta), "n_scenes": int(lm.shape[0]), "trace": trace}


def composition_bound(alpha: float, beta: float) -> float:
    """E[E2E] <= alpha + beta when G1 (scene-weighted, alpha) and G2 (beta) hold at the same lambda."""
    return float(alpha + beta)

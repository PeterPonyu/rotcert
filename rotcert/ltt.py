"""Risk-matrix threshold tests and the historical fitted support screen.

The p-value calculations and threshold selection are numerical primitives. Their
probability interpretation additionally requires a candidate family independent of
the tested data (or a suitable uniform argument), valid pointwise tests and
independent scene sampling. This module cannot verify those premises. It therefore
reports ``diagnostic_pass`` and never promotes a numerical rejection to
``certified=True``. Function names are retained for source compatibility.

The fitted support screen is an archived heuristic, not a proved Bentkus
correction or a sufficient data-collection budget. Historical result files retain
their original schema; reproducing those files requires their original code version.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import binom

__all__ = [
    "hb_pvalue",
    "eb_pvalue",
    "build_lambda_grid",
    "ltt_certify_matrix",
    "power_floor_n_img",
]


def _kl_bernoulli(a: float, b: float) -> float:
    a = float(np.clip(a, 0.0, 1.0))
    b = float(np.clip(b, 1e-12, 1.0 - 1e-12))
    term1 = 0.0 if a <= 0.0 else a * np.log(a / b)
    term2 = 0.0 if a >= 1.0 else (1.0 - a) * np.log((1.0 - a) / (1.0 - b))
    return float(term1 + term2)


def hb_pvalue(y: np.ndarray, alpha: float) -> float:
    """Hoeffding-Bentkus p-value testing ``H0: E[Y] >= alpha`` for ``Y`` bounded in
    ``[0, 1]`` (Bates et al. 2021, eq. 3). See ``asr_gate.ltt.hb_pvalue`` for the full
    derivation; identical construction, reproduced here to keep ``rotcert`` dependency-
    free of ``asr-gate``."""
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("hb_pvalue: y must be a non-empty 1-D array")
    if np.any((y < -1e-9) | (y > 1.0 + 1e-9)):
        raise ValueError("hb_pvalue: y must be bounded in [0, 1]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("hb_pvalue: alpha must be in (0, 1)")
    n = y.size
    ybar = float(np.mean(y))
    h1 = _kl_bernoulli(min(ybar, alpha), alpha)
    p_hoeffding = float(np.exp(-n * h1))
    k = int(np.ceil(n * ybar))
    p_bentkus = float(np.e * binom.cdf(k, n, alpha))
    return float(min(1.0, p_hoeffding, p_bentkus))


def eb_pvalue(y: np.ndarray, alpha: float) -> float:
    """Empirical-Bernstein p-value testing ``H0: E[Y] >= alpha``, adaptive to the
    sample variance of ``y`` (Maurer & Pontil 2009). See ``asr_gate.ltt.eb_pvalue`` for
    the full derivation; identical construction."""
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("eb_pvalue: y must be a non-empty 1-D array")
    if np.any((y < -1e-9) | (y > 1.0 + 1e-9)):
        raise ValueError("eb_pvalue: y must be bounded in [0, 1]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("eb_pvalue: alpha must be in (0, 1)")
    n = y.size
    ybar = float(np.mean(y))
    target = alpha - ybar
    if target <= 0.0 or n < 2:
        return 1.0
    v_hat = float(np.var(y, ddof=1))
    a = 7.0 / (3.0 * (n - 1))
    b = float(np.sqrt(2.0 * v_hat / n))
    disc = b * b + 4.0 * a * target
    t = (-b + np.sqrt(disc)) / (2.0 * a)
    x_star = t * t
    p = 2.0 * np.exp(-x_star)
    return float(min(1.0, max(0.0, p)))


def build_lambda_grid(
    scores: np.ndarray, n_grid: int = 50, min_accept_frac: float = 0.05
) -> np.ndarray:
    """Default candidate confidence-threshold grid: quantiles of ``scores`` (detection
    confidences), ascending, deduplicated, capped at the ``1 - min_accept_frac``
    quantile so the most-conservative candidate does not retain a near-empty
    detection set (mirrors ``asr_gate.ltt.build_lambda_grid``'s power rationale)."""
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        raise ValueError("build_lambda_grid: scores must be non-empty")
    if not 0.0 <= min_accept_frac < 1.0:
        raise ValueError("build_lambda_grid: min_accept_frac must be in [0, 1)")
    qs = np.linspace(0.0, 1.0 - min_accept_frac, n_grid)
    return np.unique(np.quantile(scores, qs))


_P_VALUE_FUNCS = {"hb": hb_pvalue, "eb": eb_pvalue}


def ltt_certify_matrix(
    risk_matrix: np.ndarray,
    lambda_grid: Sequence[float],
    beta: float,
    delta: float = 0.05,
    procedure: str = "bonferroni",
    p_value: str = "eb",
) -> Dict[str, Any]:
    """Diagnostic threshold tests for a precomputed per-scene risk matrix.

    Parameters
    ----------
    risk_matrix:
        Shape ``(n_images, K)``; ``risk_matrix[i, k]`` = image ``i``'s miss rate at
        ``lambda_grid[k]``, in ``[0, 1]``. ``K`` must equal ``len(lambda_grid)``.
    lambda_grid:
        Candidate confidence thresholds, any order (sorted internally, ascending).
    beta:
        Target FNR bound.
    delta:
        Failure probability.
    procedure:
        ``"bonferroni"`` (default, ordering-free -- every lambda tested at level
        ``delta/K``, select the passing lambda with the SMALLEST value, i.e. the
        most detections retained / least conservative among the valid ones) or
        ``"fixed-sequence"`` (walk from the most conservative -- highest -- lambda
        downward, stop at the first non-rejection; kept for parity with
        ``asr_gate.ltt``, mainly for the pilot-failure regression test).
    p_value:
        ``"eb"`` (default) or ``"hb"``; see :func:`eb_pvalue` / :func:`hb_pvalue`.

    Returns
    -------
    dict
        ``lambda_star`` (diagnostic threshold or ``None``), ``diagnostic_pass`` (bool),
        ``certified`` (always false), ``validity`` (``diagnostic_only``),
        ``realized_risk`` (mean risk at ``lambda_star``, or ``None``), ``beta``,
        ``delta``, ``n_images``, ``K``, ``procedure``, ``p_value``, ``trace`` (list of
        ``{"lambda", "p_value", "rejected", "mean_risk"}``, ascending-lambda order for
        ``"bonferroni"``, testing order for ``"fixed-sequence"``).
    """
    risk_matrix = np.asarray(risk_matrix, dtype=float)
    if risk_matrix.ndim != 2 or risk_matrix.size == 0:
        raise ValueError("ltt_certify_matrix: risk_matrix must be a non-empty 2-D array")
    if not np.all(np.isfinite(risk_matrix)) or np.any((risk_matrix < -1e-9) | (risk_matrix > 1.0 + 1e-9)):
        raise ValueError("ltt_certify_matrix: risk_matrix must be bounded in [0, 1]")
    grid = np.asarray(sorted(float(v) for v in lambda_grid), dtype=float)
    if grid.size == 0 or not np.all(np.isfinite(grid)):
        raise ValueError("lambda_grid must be non-empty and finite")
    if risk_matrix.shape[1] != grid.size:
        raise ValueError("ltt_certify_matrix: risk_matrix column count must equal len(lambda_grid)")
    if not 0.0 < beta < 1.0:
        raise ValueError("ltt_certify_matrix: beta must be in (0, 1)")
    if not 0.0 < delta < 1.0:
        raise ValueError("ltt_certify_matrix: delta must be in (0, 1)")
    if procedure not in ("bonferroni", "fixed-sequence"):
        raise ValueError(f"ltt_certify_matrix: procedure must be 'bonferroni' or 'fixed-sequence', got {procedure!r}")
    if p_value not in _P_VALUE_FUNCS:
        raise ValueError(f"ltt_certify_matrix: p_value must be one of {sorted(_P_VALUE_FUNCS)}")

    n_images = risk_matrix.shape[0]
    K = grid.size
    pfunc = _P_VALUE_FUNCS[p_value]
    # Re-order risk_matrix columns to match the sorted grid (grid may not have been sorted going in).
    col_order = np.argsort(np.asarray(list(lambda_grid), dtype=float))
    risk_sorted = risk_matrix[:, col_order]

    def _test_col(k: int) -> Dict[str, Any]:
        y = risk_sorted[:, k]
        p = pfunc(y, beta)
        return {"lambda": float(grid[k]), "p_value": p, "mean_risk": float(np.mean(y))}

    trace: List[Dict[str, Any]] = []
    lambda_star: Optional[float] = None
    lambda_star_idx: Optional[int] = None

    if procedure == "fixed-sequence":
        for k in range(K - 1, -1, -1):
            entry = _test_col(k)
            rejected = entry["p_value"] <= delta
            entry["rejected"] = bool(rejected)
            trace.append(entry)
            if rejected:
                lambda_star, lambda_star_idx = entry["lambda"], k
            else:
                break
    else:
        level = delta / K if K > 0 else delta
        best_k = -1
        for k in range(K):
            entry = _test_col(k)
            rejected = entry["p_value"] <= level
            entry["rejected"] = bool(rejected)
            trace.append(entry)
            if rejected and (best_k == -1 or k < best_k):
                best_k = k
        if best_k >= 0:
            lambda_star, lambda_star_idx = float(grid[best_k]), best_k

    if lambda_star is None:
        return {
            "lambda_star": None,
            "certified": False,
            "diagnostic_pass": False,
            "validity": "diagnostic_only",
            "realized_risk": None,
            "beta": float(beta),
            "delta": float(delta),
            "n_images": int(n_images),
            "K": int(K),
            "procedure": procedure,
            "p_value": p_value,
            "trace": trace,
        }

    return {
        "lambda_star": lambda_star,
        "certified": False,
        "diagnostic_pass": True,
        "validity": "diagnostic_only",
        "realized_risk": float(np.mean(risk_sorted[:, lambda_star_idx])),
        "beta": float(beta),
        "delta": float(delta),
        "n_images": int(n_images),
        "K": int(K),
        "procedure": procedure,
        "p_value": p_value,
        "trace": trace,
    }


def power_floor_n_img(
    beta: float, delta: float, grid_size: int, r_hat: float, bentkus_factor: float = 1.75
) -> Dict[str, Any]:
    """Historical fitted screen, conditional on the current empirical mean.

    The returned ``bentkus_floor`` name is retained for archived-schema
    compatibility; the fitted factor is not a proved Bentkus correction. An
    infinite value means the current empirical mean reaches beta, not that
    future samples cannot help. Neither finite value is a sufficient budget.
    """
    if not 0.0 < beta < 1.0:
        raise ValueError("power_floor_n_img: beta must be in (0, 1)")
    if not 0.0 < delta < 1.0:
        raise ValueError("power_floor_n_img: delta must be in (0, 1)")
    if grid_size < 1:
        raise ValueError("power_floor_n_img: grid_size must be >= 1")
    if bentkus_factor <= 0:
        raise ValueError("power_floor_n_img: bentkus_factor must be positive")
    headroom = beta - r_hat
    if headroom <= 0.0:
        hoeffding_floor = float("inf")
        bentkus_floor = float("inf")
    else:
        ln_term = float(np.log(grid_size / delta))
        hoeffding_floor = ln_term / (2.0 * headroom ** 2)
        bentkus_floor = hoeffding_floor / bentkus_factor
    return {
        "beta": float(beta),
        "delta": float(delta),
        "grid_size": int(grid_size),
        "r_hat": float(r_hat),
        "headroom": float(headroom),
        "hoeffding_floor": hoeffding_floor,
        "bentkus_floor": bentkus_floor,
        "bentkus_factor": float(bentkus_factor),
    }

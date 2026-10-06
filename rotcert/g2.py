"""G2 certification with an exogenous lambda grid (revision spec §3 C4).

Validity of Learn-then-Test requires the candidate family to be fixed independently of
the tested data. :func:`certify_risk` therefore refuses grids whose provenance is not declared as
``"fixed"`` (a priori) or ``"design_split"`` (built on scenes disjoint from the tested ones).
It also refuses any grid that is not finite and strictly increasing: ``risk_matrix``
column ``j`` is assumed to hold the risk at ``lambda_grid[j]`` (true of
:func:`rotcert.certify.image_risk_matrix`'s always-ascending output only when the caller's
grid is itself ascending), so silently re-sorting an out-of-order grid here would risk
pairing one lambda's label with another lambda's risk column -- a silent false certificate.

Bonferroni over ``len(grid) * family_size`` hypotheses (family = classes within a
detector x dataset cell) controls the FWER at ``delta``; every rejected lambda is valid.
``lambda_star`` (smallest, most permissive rejected lambda) and ``lambda_max`` (largest
rejected lambda -- the deployable operating threshold: highest confidence with certified
recall) bracket the valid range; ``n_rejected`` is its size.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence

import numpy as np

from rotcert.ltt import eb_pvalue, hb_pvalue

__all__ = ["FIXED_LAMBDA_GRID", "GridProvenanceError", "certify_risk", "eb_min_images", "eb_necessary_images"]

FIXED_LAMBDA_GRID = tuple(float(x) for x in np.round(np.arange(0.01, 1.0, 0.02), 2))
_ALLOWED = ("fixed", "design_split")
_PV = {"eb": eb_pvalue, "hb": hb_pvalue}


class GridProvenanceError(ValueError):
    """Raised when a lambda grid's provenance does not support a finite-sample guarantee."""


def certify_risk(
    risk_matrix: np.ndarray,
    lambda_grid: Sequence[float],
    beta: float,
    delta: float,
    family_size: int = 1,
    grid_source: str = "fixed",
    p_value: str = "eb",
) -> Dict[str, Any]:
    if grid_source not in _ALLOWED:
        raise GridProvenanceError(f"grid_source must be one of {_ALLOWED}, got {grid_source!r}")
    rm = np.asarray(risk_matrix, dtype=float)
    grid = np.asarray(list(lambda_grid), dtype=float)
    if rm.ndim != 2 or rm.shape[1] != grid.size or grid.size == 0:
        raise ValueError("risk_matrix must be (n_scenes, len(lambda_grid))")
    if not np.all(np.isfinite(rm)) or rm.min() < -1e-9 or rm.max() > 1 + 1e-9:
        raise ValueError("risk_matrix must be finite and in [0, 1]")
    if int(family_size) < 1:
        raise ValueError("family_size must be >= 1")
    if not np.all(np.isfinite(grid)) or not np.all(np.diff(grid) > 0):
        raise ValueError(
            "lambda_grid must be finite and strictly increasing (risk_matrix column j "
            "must correspond to lambda_grid[j]; certify_risk does not re-sort)"
        )
    level = delta / (grid.size * int(family_size))
    pf = _PV[p_value]
    trace = []
    star = None
    last = None
    n_rejected = 0
    for k in range(grid.size):
        p = pf(rm[:, k], beta)
        rej = bool(p <= level)
        trace.append({"lambda": float(grid[k]), "p_value": float(p), "rejected": rej, "mean_risk": float(rm[:, k].mean())})
        if rej:
            n_rejected += 1
            if star is None:
                star = k
            last = k
    return {
        "certified": star is not None,
        "lambda_star": None if star is None else float(grid[star]),
        "realized_risk": None if star is None else float(rm[:, star].mean()),
        "lambda_max": None if last is None else float(grid[last]),
        "realized_risk_max": None if last is None else float(rm[:, last].mean()),
        "n_rejected": int(n_rejected),
        "level": float(level),
        "beta": float(beta),
        "delta": float(delta),
        "family_size": int(family_size),
        "grid_source": grid_source,
        "p_value": p_value,
        "n_scenes": int(rm.shape[0]),
        "trace": trace,
    }


def _eb_rejects(n: int, r: float, v: float, beta: float, L: float) -> bool:
    return 7.0 * L / (3.0 * (n - 1)) + math.sqrt(2.0 * v * L / n) <= beta - r


def eb_min_images(r: float, v: float, beta: float, eta: float, n_max: int = 10**9) -> float:
    """Smallest n at which the empirical-Bernstein test rejects H0: E[R] >= beta at level eta
    for a sample with mean r and sample variance v (plug-in design quantity, no fitted constant)."""
    if beta <= r:
        return math.inf
    L = math.log(2.0 / eta)
    if not _eb_rejects(n_max, r, v, beta, L):
        return math.inf
    lo, hi = 2, n_max
    while lo < hi:
        mid = (lo + hi) // 2
        if _eb_rejects(mid, r, v, beta, L):
            hi = mid
        else:
            lo = mid + 1
    return float(lo)


def eb_necessary_images(r: float, beta: float, eta: float) -> float:
    """Zero-variance necessary condition: n >= 1 + 7 ln(2/eta) / (3 (beta - r))."""
    if beta <= r:
        return math.inf
    return 1.0 + 7.0 * math.log(2.0 / eta) / (3.0 * (beta - r))

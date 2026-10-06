"""Multiple-comparison helpers: Holm-Bonferroni (FWER) and Benjamini-Hochberg (FDR).

Both return adjusted p-values AND boolean reject decisions at the given
``alpha`` so results tables can report either. Adjusted p-values are
monotonicity-enforced and clipped to 1.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

__all__ = ["holm_bonferroni", "benjamini_hochberg"]


def _validate_pvals(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("pvals must be a non-empty 1-D sequence")
    if np.any((p < 0) | (p > 1)) or np.any(np.isnan(p)):
        raise ValueError("pvals must be in [0, 1] and non-NaN")
    return p


def holm_bonferroni(pvals: Sequence[float], alpha: float = 0.05) -> Dict[str, np.ndarray]:
    """Holm-Bonferroni step-down procedure (controls FWER at ``alpha``).

    Parameters
    ----------
    pvals:
        Raw p-values, any order.
    alpha:
        Family-wise error rate.

    Returns
    -------
    dict
        ``adjusted_p`` (Holm-adjusted p-values, original order) and
        ``reject`` (boolean, original order; equivalent to
        ``adjusted_p <= alpha``).
    """
    p = _validate_pvals(pvals)
    m = p.size
    order = np.argsort(p, kind="stable")
    adj_sorted = (m - np.arange(m)) * p[order]
    adj_sorted = np.minimum(np.maximum.accumulate(adj_sorted), 1.0)  # step-down monotone
    adjusted = np.empty(m)
    adjusted[order] = adj_sorted
    return {"adjusted_p": adjusted, "reject": adjusted <= alpha}


def benjamini_hochberg(pvals: Sequence[float], alpha: float = 0.05) -> Dict[str, np.ndarray]:
    """Benjamini-Hochberg step-up procedure (controls FDR at ``alpha``).

    Parameters
    ----------
    pvals:
        Raw p-values, any order.
    alpha:
        Target false-discovery rate.

    Returns
    -------
    dict
        ``adjusted_p`` (BH-adjusted p-values / q-values, original order) and
        ``reject`` (boolean, original order; equivalent to
        ``adjusted_p <= alpha``).
    """
    p = _validate_pvals(pvals)
    m = p.size
    order = np.argsort(p, kind="stable")
    ranks = np.arange(1, m + 1)
    adj_sorted = p[order] * m / ranks
    # Step-up: enforce monotone non-decreasing from the largest p downward.
    adj_sorted = np.minimum(np.minimum.accumulate(adj_sorted[::-1])[::-1], 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adj_sorted
    return {"adjusted_p": adjusted, "reject": adjusted <= alpha}

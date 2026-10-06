"""Blocked bootstrap with multi-seed support and percentile / BCa intervals.

Why blocked
-----------
Portfolio datasets are clustered: multiple forecasts per weather event,
multiple rollouts per robot initialization, multiple relaxations per
material. Resampling individual samples i.i.d. understates variance. Here
the resampling unit is the BLOCK (``block_ids`` = event / group / init id):
blocks are drawn with replacement and every member of a drawn block enters
the replicate.

Multi-seed
----------
``seeds`` may contain several integers; ``n_boot`` replicates are drawn per
seed and pooled. The seeds used are recorded in the returned dict so any
interval can be reproduced exactly.

Nested designs
--------------
:func:`nested_bootstrap` generalizes the single-level blocking above to an
arbitrary-depth block hierarchy (e.g. env -> instance -> seed): outer blocks
are drawn with replacement, then within each drawn block the next level is
drawn with replacement, down to the individual rows of the finest block.

BCa primitive
-------------
:func:`bca_interval` is the shared bias-corrected-and-accelerated quantile
adjustment (two-sided and one-sided modes). It operates on precomputed
bootstrap replicates plus either a jackknife of the statistic or a
precomputed acceleration constant, so both the block bootstrap here and the
seed-mean bootstrap in ``drlcommons.verdict`` route through one codepath.

Returned dict
-------------
``{"point": float, "ci": (lo, hi), "seeds": [...], "method": str,
"n_boot": int, "ci_level": float, "n_blocks": int, "replicates": ndarray}``
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from scipy.stats import norm

__all__ = ["blocked_bootstrap", "nested_bootstrap", "bca_interval", "as_arrays"]

ArrayLike = Union[np.ndarray, Sequence[float]]


def as_arrays(data: Union[ArrayLike, Sequence[ArrayLike]]) -> list:
    """Normalize ``data`` to a list of equal-length 1-D float-compatible arrays.

    Accepts a single 1-D array (wrapped to a one-element list) or a sequence of
    equal-length 1-D arrays (each resampled with shared indices). Pure shape
    plumbing shared across the bootstrap entry points; contains no statistics.
    """
    if isinstance(data, np.ndarray) and data.ndim == 1:
        arrays = [data]
    elif isinstance(data, (list, tuple)) and len(data) > 0 and np.ndim(data[0]) >= 1:
        arrays = [np.asarray(a) for a in data]
    else:
        arrays = [np.asarray(data)]
    n = len(arrays[0])
    for a in arrays:
        if len(a) != n:
            raise ValueError("all data arrays must have equal length")
    if n == 0:
        raise ValueError("data must be non-empty")
    return arrays


def bca_interval(
    replicates: ArrayLike,
    point: float,
    *,
    jackknife: Optional[ArrayLike] = None,
    acceleration: Optional[float] = None,
    ci_level: float = 0.95,
    side: str = "two-sided",
    alpha_lower: Optional[float] = None,
    alpha_upper: Optional[float] = None,
) -> tuple:
    """Bias-corrected and accelerated (BCa) interval from precomputed replicates.

    The bias correction ``z0`` is read off ``replicates`` relative to ``point``;
    the acceleration is either supplied directly via ``acceleration`` or derived
    from a leave-one-out ``jackknife`` of the statistic (any resampling level —
    delete-one-block for a blocked bootstrap, delete-one-element for a seed-mean
    bootstrap). Exactly one of ``jackknife`` / ``acceleration`` must be given.

    Parameters
    ----------
    replicates:
        1-D array of bootstrap statistic values.
    point:
        The statistic evaluated on the full data.
    jackknife:
        Leave-one-out statistic values used to compute the acceleration
        constant ``a = sum((jm - j)^3) / (6 * (sum((jm - j)^2))^1.5)``.
    acceleration:
        Precomputed acceleration constant (alternative to ``jackknife``).
    ci_level:
        Two-sided confidence level; only used when ``alpha_lower`` and
        ``alpha_upper`` are both ``None`` (the ``side``-derived path).
    side:
        ``"two-sided"`` (both tails ``(1 - ci_level) / 2``), ``"upper"`` (all
        of ``1 - ci_level`` in the upper tail; lower bound ``-inf``), or
        ``"lower"`` (all in the lower tail; upper bound ``+inf``).
    alpha_lower, alpha_upper:
        Explicit tail masses. When given they OVERRIDE ``ci_level``/``side``:
        the lower bound is the ``adj(alpha_lower)`` replicate quantile and the
        upper bound is the ``adj(1 - alpha_upper)`` replicate quantile. A tail
        mass of exactly ``0.0`` yields an infinite bound on that side.

    Returns
    -------
    tuple
        ``(lo, hi)`` interval bounds.
    """
    replicates = np.asarray(replicates, dtype=float)
    if replicates.ndim != 1 or replicates.size == 0:
        raise ValueError("replicates must be a non-empty 1-D array")
    if (jackknife is None) == (acceleration is None):
        raise ValueError("provide exactly one of jackknife or acceleration")

    if alpha_lower is None and alpha_upper is None:
        alpha = 1.0 - ci_level
        if side == "two-sided":
            alpha_lower = alpha_upper = alpha / 2.0
        elif side == "upper":
            alpha_lower, alpha_upper = 0.0, alpha
        elif side == "lower":
            alpha_lower, alpha_upper = alpha, 0.0
        else:
            raise ValueError("side must be 'two-sided', 'upper', or 'lower'")
    else:
        alpha_lower = 0.0 if alpha_lower is None else float(alpha_lower)
        alpha_upper = 0.0 if alpha_upper is None else float(alpha_upper)
    if not (0.0 <= alpha_lower < 1.0 and 0.0 <= alpha_upper < 1.0):
        raise ValueError("tail masses must lie in [0, 1)")

    # Bias correction z0 (midrank handling of ties at the point estimate).
    prop = (np.sum(replicates < point) + 0.5 * np.sum(replicates == point)) / len(
        replicates
    )
    prop = float(np.clip(prop, 1e-9, 1.0 - 1e-9))
    z0 = norm.ppf(prop)

    if acceleration is None:
        jack = np.asarray(jackknife, dtype=float)
        jm = jack.mean()
        num = np.sum((jm - jack) ** 3)
        den = 6.0 * np.sum((jm - jack) ** 2) ** 1.5
        a = float(num / den) if den > 0 else 0.0
    else:
        a = float(acceleration)

    def adj(q: float) -> float:
        z = norm.ppf(q)
        denom = 1.0 - a * (z0 + z)
        if denom == 0:
            denom = 1e-12
        return float(norm.cdf(z0 + (z0 + z) / denom))

    lo = -np.inf if alpha_lower == 0.0 else float(np.quantile(replicates, adj(alpha_lower)))
    hi = np.inf if alpha_upper == 0.0 else float(np.quantile(replicates, adj(1.0 - alpha_upper)))
    return (lo, hi)


def _block_jackknife(
    stat_fn: Callable[..., float], arrays: list, members: list
) -> np.ndarray:
    """Delete-one-block jackknife of ``stat_fn`` over the given block members."""
    n_blocks = len(members)
    jack = np.empty(n_blocks)
    all_idx = np.arange(len(arrays[0]))
    for b in range(n_blocks):
        keep = np.setdiff1d(all_idx, members[b], assume_unique=True)
        jack[b] = stat_fn(*(a[keep] for a in arrays))
    return jack


def blocked_bootstrap(
    stat_fn: Callable[..., float],
    data: Union[ArrayLike, Sequence[ArrayLike]],
    block_ids: Optional[ArrayLike] = None,
    n_boot: int = 1000,
    seeds: Sequence[int] = (0,),
    ci_level: float = 0.95,
    method: str = "percentile",
) -> Dict[str, object]:
    """Blocked bootstrap confidence interval for ``stat_fn(*data)``.

    Parameters
    ----------
    stat_fn:
        Callable taking the (resampled) data arrays as positional arguments
        and returning a scalar statistic.
    data:
        A single 1-D array or a sequence of equal-length 1-D arrays; each is
        resampled with the same indices and passed to ``stat_fn``.
    block_ids:
        Per-sample block key (event/group/init id). Samples sharing a block
        id are resampled together. ``None`` means i.i.d. (each sample its
        own block).
    n_boot:
        Number of bootstrap replicates PER SEED.
    seeds:
        Sequence of integer seeds; replicates are pooled across seeds and
        the seeds are recorded in the output for reproducibility.
    ci_level:
        Two-sided confidence level, e.g. ``0.95``.
    method:
        ``"percentile"`` or ``"bca"`` (bias-corrected and accelerated;
        acceleration from a delete-one-BLOCK jackknife).

    Returns
    -------
    dict
        ``point`` (statistic on the full data), ``ci`` (lo, hi tuple),
        ``seeds`` (list), ``method``, ``n_boot`` (total pooled), ``ci_level``,
        ``n_blocks``, and ``replicates`` (pooled bootstrap statistics).
    """
    if method not in ("percentile", "bca"):
        raise ValueError("method must be 'percentile' or 'bca'")
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be in (0, 1)")
    if len(seeds) == 0:
        raise ValueError("provide at least one seed")

    arrays = as_arrays(data)
    n = len(arrays[0])
    if block_ids is None:
        block_ids = np.arange(n)
    block_ids = np.asarray(block_ids)
    if len(block_ids) != n:
        raise ValueError("block_ids must have the same length as the data")

    unique_blocks, inverse = np.unique(block_ids, return_inverse=True)
    n_blocks = len(unique_blocks)
    # Member indices per block, precomputed once.
    order = np.argsort(inverse, kind="stable")
    boundaries = np.searchsorted(inverse[order], np.arange(n_blocks + 1))
    members = [order[boundaries[b]:boundaries[b + 1]] for b in range(n_blocks)]

    point = float(stat_fn(*arrays))

    replicates = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for _ in range(n_boot):
            drawn = rng.integers(0, n_blocks, size=n_blocks)
            idx = np.concatenate([members[b] for b in drawn])
            replicates.append(float(stat_fn(*(a[idx] for a in arrays))))
    replicates = np.asarray(replicates)

    if method == "percentile":
        alpha = 1.0 - ci_level
        lo, hi = np.quantile(replicates, [alpha / 2.0, 1.0 - alpha / 2.0])
        ci = (float(lo), float(hi))
    else:
        jack = _block_jackknife(stat_fn, arrays, members)
        ci = bca_interval(
            replicates, point, jackknife=jack, ci_level=ci_level, side="two-sided"
        )

    return {
        "point": point,
        "ci": ci,
        "seeds": list(seeds),
        "method": method,
        "n_boot": int(len(replicates)),
        "ci_level": float(ci_level),
        "n_blocks": int(n_blocks),
        "replicates": replicates,
    }


def _build_nested_tree(row_indices: np.ndarray, level_ids: List[np.ndarray]):
    """Recursively group ``row_indices`` by the block-id levels.

    Returns a nested list mirroring the block hierarchy: at each level a list
    of child subtrees, one per unique id value (in ``np.unique`` sorted order,
    resolved WITHIN the parent group); the leaf (after the last level) is the
    1-D array of row indices of the finest block. Row order is preserved
    ascending throughout so the resample index stream is reproducible.
    """
    if not level_ids:
        return row_indices
    ids = level_ids[0][row_indices]
    children = []
    for u in np.unique(ids):
        sub = row_indices[ids == u]
        children.append(_build_nested_tree(sub, level_ids[1:]))
    return children


def _draw_nested(node, rng: np.random.Generator) -> np.ndarray:
    """Draw one nested-bootstrap resample of row indices from a prebuilt tree.

    A leaf (ndarray of rows) is resampled with replacement; an internal node
    (list of children) draws its children with replacement and concatenates
    each drawn child's own recursive resample, so variance propagates from
    every level of the hierarchy.
    """
    if isinstance(node, np.ndarray):
        n = len(node)
        return node[rng.integers(0, n, size=n)]
    k = len(node)
    drawn = rng.integers(0, k, size=k)
    parts = [_draw_nested(node[j], rng) for j in drawn]
    return np.concatenate(parts) if parts else np.empty(0, dtype=int)


def nested_bootstrap(
    stat_fn: Callable[..., float],
    data: Union[ArrayLike, Sequence[ArrayLike]],
    block_levels: Sequence[ArrayLike],
    n_boot: int = 1000,
    seeds: Sequence[int] = (0,),
    ci_level: float = 0.95,
) -> Dict[str, object]:
    """Nested (multi-level) cluster bootstrap CI for ``stat_fn(*data)``.

    Generalizes :func:`blocked_bootstrap` to an arbitrary-depth block
    hierarchy. ``block_levels`` is a sequence of per-row id arrays ordered
    OUTERMOST first (e.g. ``[env_ids, instance_ids]``). Each replicate draws
    the outermost blocks with replacement, then within each drawn block draws
    the next level with replacement, and so on; within the finest block the
    individual rows are resampled with replacement (that innermost row level is
    the seed level under the portfolio's one-row-per-seed convention). This is
    the honest nested random-effects resample: a level's variance only enters
    when the level above it was also resampled.

    A percentile CI is used (not BCa): a BCa acceleration for a nested scheme
    would need a delete-one jackknife of the nested resample, which has no
    simple closed form.

    Returns
    -------
    dict
        Same shape as :func:`blocked_bootstrap` with ``method =
        "nested-percentile"`` and ``n_blocks`` = the number of outermost
        blocks.
    """
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be in (0, 1)")
    if len(seeds) == 0:
        raise ValueError("provide at least one seed")
    if len(block_levels) == 0:
        raise ValueError("provide at least one block level")

    arrays = as_arrays(data)
    n = len(arrays[0])
    levels = [np.asarray(b) for b in block_levels]
    for b in levels:
        if len(b) != n:
            raise ValueError("each block level must have the same length as the data")

    tree = _build_nested_tree(np.arange(n), levels)
    n_blocks = len(tree) if isinstance(tree, list) else 1

    point = float(stat_fn(*arrays))

    replicates = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for _ in range(n_boot):
            idx = _draw_nested(tree, rng)
            replicates.append(float(stat_fn(*(a[idx] for a in arrays))))
    replicates = np.asarray(replicates)

    alpha = 1.0 - ci_level
    lo, hi = np.quantile(replicates, [alpha / 2.0, 1.0 - alpha / 2.0])

    return {
        "point": point,
        "ci": (float(lo), float(hi)),
        "seeds": list(seeds),
        "method": "nested-percentile",
        "n_boot": int(len(replicates)),
        "ci_level": float(ci_level),
        "n_blocks": int(n_blocks),
        "replicates": replicates,
    }

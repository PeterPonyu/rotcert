"""Exact calibration boundaries for finite, nonnegative scores.

Fraction/string/Decimal parameters preserve their written rational value. Floating
parameters preserve their *binary represented* value, including np.float32; use
"0.1" for the exactly nominal one-tenth design. No tolerance changes a decision.
"""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from fractions import Fraction
import math
import numbers
from typing import Sequence

import numpy as np


def rational(value) -> Fraction:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("boolean is not a numerical parameter")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (str, Decimal)):
        try:
            return Fraction(value)
        except (ValueError, OverflowError) as exc:
            raise ValueError("finite rational parameter required") from exc
    if isinstance(value, numbers.Integral):
        return Fraction(int(value))
    if isinstance(value, numbers.Real):
        f = float(value)
        if not math.isfinite(f):
            raise ValueError("finite rational parameter required")
        return Fraction.from_float(f)
    raise ValueError("use Fraction, Decimal, numeric string, integer or finite float")


def probability(value, name="probability") -> Fraction:
    result = rational(value)
    if not 0 < result < 1:
        raise ValueError(f"{name} must be strictly between zero and one")
    return result


def integer(value, name="count", minimum=0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer (no float truncation)")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def score_groups(groups: Sequence[Sequence[float]]) -> list[np.ndarray]:
    result = []
    for group in groups:
        a = np.asarray(group, dtype=float)
        if a.ndim != 1 or not np.all(np.isfinite(a)) or np.any(a < 0):
            raise ValueError("each scene needs a 1-D finite nonnegative score array")
        result.append(a)
    return result


def conformal_rank(n, alpha) -> int:
    n, alpha = integer(n), probability(alpha, "alpha")
    # ceil((n+1)*(1-alpha)), performed entirely with integer arithmetic.
    return n + 1 - ((n + 1) * alpha.numerator // alpha.denominator)


def pooled_threshold(groups, alpha) -> float:
    groups = [g for g in score_groups(groups) if len(g)]
    alpha = probability(alpha, "alpha")
    if not groups:
        return math.inf
    scores = np.concatenate(groups)
    rank = conformal_rank(len(scores), alpha)
    return math.inf if rank > len(scores) else float(np.partition(scores, rank - 1)[rank - 1])


def scene_max_threshold(groups, alpha) -> float:
    groups = score_groups(groups)
    return pooled_threshold([[float(g.max())] for g in groups if len(g)], alpha)


def hcp_threshold(groups, alpha) -> float:
    """Exact HCP CDF comparison, including the +infinity scene atom.

    Aggregate equal scene cardinalities. A binary search over observed scores
    needs only O(log N * distinct_cardinalities) rational additions, instead of
    accumulating one floating weight per object. Ties enter the CDF together.
    """
    alpha = probability(alpha, "alpha")
    groups = [g for g in score_groups(groups) if len(g)]
    k = len(groups)
    target = (k + 1) * (1 - alpha)
    if k == 0 or k < target:
        return math.inf
    by_size = defaultdict(list)
    for group in groups:
        by_size[len(group)].append(group)
    arrays = {m: np.sort(np.concatenate(gs)) for m, gs in by_size.items()}
    candidates = np.unique(np.concatenate(list(arrays.values())))

    def reaches(q):
        mass = sum((Fraction(int(np.searchsorted(a, q, side="right")), m)
                    for m, a in arrays.items()), Fraction(0))
        return mass >= target

    lo, hi = 0, len(candidates) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if reaches(candidates[mid]):
            hi = mid
        else:
            lo = mid + 1
    return float(candidates[lo])


def crc_budget(n, alpha, deployment_bound) -> Fraction:
    n = integer(n)
    d = integer(deployment_bound, "deployment_bound")
    alpha = probability(alpha, "alpha")
    return alpha * n - (1 - alpha) * d


def _crc_from_sorted(scores, alpha, d) -> float:
    budget = crc_budget(len(scores), alpha, d)
    if budget < 0:
        return math.inf
    allowed = budget.numerator // budget.denominator
    rank = len(scores) - allowed
    return 0.0 if rank <= 0 else float(scores[rank - 1])


def crc_object_adaptive_thresholds(groups, alpha, d_tests) -> np.ndarray:
    """CRC-D; D>=m_test and integrability remain deployment assumptions."""
    alpha = probability(alpha, "alpha")
    groups = [g for g in score_groups(groups) if len(g)]
    scores = np.sort(np.concatenate(groups)) if groups else np.empty(0)
    # dtype=object preserves integers beyond 2**53 instead of converting to float.
    d = np.asarray(d_tests, dtype=object)
    cache = {}
    output = []
    for value in d.flat:
        count = integer(value, "d_test")
        if count not in cache:
            cache[count] = _crc_from_sorted(scores, alpha, count)
        output.append(cache[count])
    return np.asarray(output, dtype=float).reshape(d.shape)


def crc_object_adaptive_threshold(groups, alpha, d_test) -> float:
    return float(crc_object_adaptive_thresholds(groups, alpha, [d_test])[0])


def crc_object_threshold(groups, alpha, M) -> float:
    M = integer(M, "population_M", 1)
    groups = score_groups(groups)
    if any(len(g) > M for g in groups):
        raise ValueError("observed scene violates the declared population bound M")
    return crc_object_adaptive_threshold(groups, alpha, M)

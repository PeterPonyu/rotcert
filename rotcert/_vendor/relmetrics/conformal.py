"""Split conformal prediction with exact coverage and Mondrian variants.

Conventions
-----------
Nonconformity scores: HIGHER = more nonconforming (worse fit). A test point
is covered (in the prediction set) when its score is not too large relative
to the calibration scores.

Exact coverage via randomized tie-breaking
------------------------------------------
The classic split-conformal threshold ``ceil((n+1)(1-alpha))``-th calibration
score guarantees coverage >= 1 - alpha but OVER-covers, badly so with small
``n`` or heavily tied (discrete) scores. The smoothed conformal p-value

    p(s) = ( #{s_i > s} + U * (#{s_i == s} + 1) ) / (n + 1),   U ~ Unif(0,1)

is exactly Uniform(0,1) under exchangeability, so accepting when
``p(s) > alpha`` gives marginal coverage EXACTLY ``1 - alpha`` in
expectation, ties or not.

Mondrian / group-conditional conformal
--------------------------------------
Calibration is done separately within each group (a "Mondrian taxonomy"),
giving the coverage guarantee PER GROUP rather than only marginally --
essential when reliability must not be bought on one stratum and spent on
another (cf. selective-classification disparities, Jones et al. 2021).

Effective coverage reporting
----------------------------
:func:`effective_coverage` summarizes realized coverage overall and per
group, with counts, so papers can report the achieved (not just nominal)
coverage.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np

__all__ = [
    "conformal_pvalues",
    "SplitConformal",
    "MondrianConformal",
    "effective_coverage",
]


def conformal_pvalues(
    cal_scores: np.ndarray,
    test_scores: np.ndarray,
    rng: Optional[Union[int, np.random.Generator]] = None,
    randomize: bool = True,
) -> np.ndarray:
    """Smoothed (randomized) split-conformal p-values.

    Parameters
    ----------
    cal_scores:
        Calibration nonconformity scores, shape ``(n,)``.
    test_scores:
        Test nonconformity scores, shape ``(m,)``.
    rng:
        Seed or ``numpy.random.Generator`` for the tie-breaking uniforms.
    randomize:
        If ``False``, use the conservative (deterministic) p-value
        ``(#{s_i >= s} + 1) / (n + 1)`` instead of the smoothed one.

    Returns
    -------
    ndarray, shape ``(m,)``
        P-values; under exchangeability the randomized version is exactly
        Uniform(0,1), the deterministic version is super-uniform.
    """
    cal = np.sort(np.asarray(cal_scores, dtype=float))
    test = np.atleast_1d(np.asarray(test_scores, dtype=float))
    n = len(cal)
    if n == 0:
        raise ValueError("cal_scores must be non-empty")
    n_gt = n - np.searchsorted(cal, test, side="right")  # #{cal > s}
    n_eq = np.searchsorted(cal, test, side="right") - np.searchsorted(
        cal, test, side="left"
    )  # #{cal == s}
    if randomize:
        gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        u = gen.uniform(size=test.shape)
        p = (n_gt + u * (n_eq + 1)) / (n + 1)
    else:
        p = (n_gt + n_eq + 1) / (n + 1)
    return p


class SplitConformal:
    """Split conformal calibration with exact coverage via randomization.

    Parameters
    ----------
    alpha:
        Miscoverage level; target coverage is ``1 - alpha``.
    randomize:
        Use randomized tie-breaking (exact ``1 - alpha`` expected coverage).
        If ``False``, the conservative rule (coverage >= ``1 - alpha``).
    rng:
        Seed or Generator for tie-breaking.

    Examples
    --------
    >>> sc = SplitConformal(alpha=0.1, rng=0).fit(cal_scores)
    >>> covered = sc.covers(test_scores)          # bool array
    >>> thr = sc.threshold                        # deterministic threshold
    """

    def __init__(
        self,
        alpha: float = 0.1,
        randomize: bool = True,
        rng: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = float(alpha)
        self.randomize = bool(randomize)
        self._rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        self._cal: Optional[np.ndarray] = None

    def fit(self, cal_scores: np.ndarray) -> "SplitConformal":
        """Store calibration scores. Returns ``self``."""
        cal = np.asarray(cal_scores, dtype=float)
        if cal.ndim != 1 or cal.size == 0:
            raise ValueError("cal_scores must be a non-empty 1-D array")
        self._cal = np.sort(cal)
        return self

    @property
    def n_cal(self) -> int:
        """Number of calibration points."""
        self._check_fitted()
        return int(len(self._cal))

    @property
    def threshold(self) -> float:
        """Deterministic conservative threshold: the ``ceil((n+1)(1-alpha))``-th
        smallest calibration score (``+inf`` if the rank exceeds ``n``).

        Rank is computed as ``(n+1) - floor((n+1)*alpha)``, which equals
        ``ceil((n+1)(1-alpha))`` in exact arithmetic but avoids the float64
        overshoot where ``(n+1)*(1-alpha)`` lands just above an integer
        (e.g. ``alpha=1/3``) and naive ``ceil`` jumps by one.
        """
        self._check_fitted()
        n = len(self._cal)
        # Exact-arithmetic identity for integer N=n+1:
        #   ceil(N*(1-a)) = N - floor(N*a).
        rank = (n + 1) - int(np.floor((n + 1) * self.alpha))
        if rank > n:
            return float("inf")
        return float(self._cal[rank - 1])

    def pvalues(self, test_scores: np.ndarray) -> np.ndarray:
        """Conformal p-values for test scores (randomized iff configured)."""
        self._check_fitted()
        return conformal_pvalues(
            self._cal, test_scores, rng=self._rng, randomize=self.randomize
        )

    def covers(self, test_scores: np.ndarray) -> np.ndarray:
        """Boolean array: True where the test point is in the prediction set
        (p-value > alpha). Expected mean is exactly ``1 - alpha`` when
        randomized and scores are exchangeable."""
        return self.pvalues(test_scores) > self.alpha

    def _check_fitted(self) -> None:
        if self._cal is None:
            raise RuntimeError("call fit(cal_scores) first")


class MondrianConformal:
    """Group-conditional (Mondrian) split conformal.

    A separate :class:`SplitConformal` calibrator per group, giving the
    ``1 - alpha`` guarantee WITHIN each group.

    Parameters
    ----------
    alpha, randomize, rng:
        As in :class:`SplitConformal`; the rng is shared across groups.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        randomize: bool = True,
        rng: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        self.alpha = float(alpha)
        self.randomize = bool(randomize)
        self._rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        self._per_group: Dict[object, SplitConformal] = {}

    def fit(self, cal_scores: np.ndarray, cal_groups: np.ndarray) -> "MondrianConformal":
        """Fit one calibrator per unique value of ``cal_groups``."""
        cal_scores = np.asarray(cal_scores, dtype=float)
        cal_groups = np.asarray(cal_groups)
        if cal_scores.shape != cal_groups.shape:
            raise ValueError("cal_scores and cal_groups must have the same shape")
        self._per_group = {}
        for g in np.unique(cal_groups):
            sc = SplitConformal(self.alpha, self.randomize, self._rng)
            sc.fit(cal_scores[cal_groups == g])
            self._per_group[g] = sc
        return self

    @property
    def groups(self) -> list:
        """Groups seen during fit."""
        return list(self._per_group.keys())

    def thresholds(self) -> Dict[object, float]:
        """Per-group deterministic thresholds."""
        return {g: sc.threshold for g, sc in self._per_group.items()}

    def covers(self, test_scores: np.ndarray, test_groups: np.ndarray) -> np.ndarray:
        """Boolean coverage per test point using its group's calibrator.

        Raises ``KeyError`` for a test group unseen at calibration (no
        guarantee is available there; handle explicitly upstream).
        """
        test_scores = np.atleast_1d(np.asarray(test_scores, dtype=float))
        test_groups = np.atleast_1d(np.asarray(test_groups))
        if test_scores.shape != test_groups.shape:
            raise ValueError("test_scores and test_groups must have the same shape")
        out = np.zeros(test_scores.shape, dtype=bool)
        for g in np.unique(test_groups):
            if g not in self._per_group:
                raise KeyError(f"group {g!r} not seen during calibration")
            mask = test_groups == g
            out[mask] = self._per_group[g].covers(test_scores[mask])
        return out


def effective_coverage(
    covered: np.ndarray,
    groups: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Report realized (effective) coverage, overall and per group.

    Parameters
    ----------
    covered:
        Boolean array, e.g. from :meth:`SplitConformal.covers`.
    groups:
        Optional per-sample group labels for a per-group breakdown.

    Returns
    -------
    dict
        ``{"overall": float, "n": int, "per_group": {g: {"coverage": float,
        "n": int}}}`` (``per_group`` empty when ``groups`` is ``None``).
    """
    covered = np.atleast_1d(np.asarray(covered, dtype=bool))
    result: Dict[str, object] = {
        "overall": float(covered.mean()),
        "n": int(covered.size),
        "per_group": {},
    }
    if groups is not None:
        groups = np.atleast_1d(np.asarray(groups))
        if groups.shape != covered.shape:
            raise ValueError("groups must have the same shape as covered")
        for g in np.unique(groups):
            mask = groups == g
            result["per_group"][g] = {
                "coverage": float(covered[mask].mean()),
                "n": int(mask.sum()),
            }
    return result

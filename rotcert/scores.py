"""The six nonconformity constructions through one pipeline (design §2.2, §4.4).

======  =================================  ========  ==============================
name    construction                       role      coverage-set shape
======  =================================  ========  ==============================
gwd     Gaussian-Wasserstein distance (S0) PRIMARY   GWD-ball (rotcert.sets)
naive-coord  Euclidean coord-wise + Bonferroni (B1)  baseline  axis-aligned box, all 5 OBB coords
hull    axis-aligned-hull coord-wise + Bonferroni (B2) baseline  axis-aligned box, 4 hull coords
wrapped-coord  wrapped-geodesic coord-wise (A1)  ablation  axis-aligned box, theta wrapped
doubled doubled-angle (cos2t,sin2t) coord-wise (A2) ablation  axis-aligned box, 6 coords
iou     1 - rotated IoU (A3)               ablation  coordinate-free (IoU >= 1-q)
======  =================================  ========  ==============================

Every construction exposes the same three-method surface (:class:`ScalarScore` for
gwd/iou, :class:`BonferroniBoxScore` for the four coordinate-wise constructions), so
``certify.py``/``audit.py`` drive all six through identical calibrate/cover/set-size
calls -- design §3.1's "the head-to-head is apples-to-apples by construction" applied
to the score layer, not just the CLI.

Why B1/hull/wrapped-coord/doubled use per-coordinate Bonferroni (not a max-normalized
scalar)
--------------------------------------------------------------------------------------
A single scalar ``s = max_k(|delta_k| / sigma_k)`` with one split-conformal threshold
gives EXACT joint coverage with no Bonferroni correction needed (the threshold is
literally the quantile of a max) -- tempting, but it is not the baseline the design
names. The design's B1 is explicitly "naive coordinate-wise CP + Bonferroni on
(cx,cy,w,h,theta)" (the de Grancey et al. 2022 corner-wise/Bonferroni lineage): K
SEPARATE per-coordinate conformal intervals, each calibrated at level ``alpha/K``, so
each coordinate's OWN marginal coverage is ``1 - alpha/K`` and the union bound gives
joint coverage ``>= 1 - alpha`` -- CONSERVATIVE by construction, which is exactly the
mechanism the paper is testing for failure (a naive angle interval that does not wrap
either misses wrapped GT outright, breaking even the union-bound floor, or absorbs
wraparound outliers into an inflated ``q_theta`` that dominates the Bonferroni box).
Swapping in the max-normalized-scalar construction would remove the failure mode being
tested, so :class:`BonferroniBoxScore` implements literal per-coordinate Bonferroni.

Set size (the C2 head-to-head efficiency metric, design §4.4)
------------------------------------------------------------------
To keep the "calibrate to the same NOMINAL level, then compare set size" comparison
(the nominal alpha is matched across constructions; realized coverage is NOT -- the
union-bound baselines over-cover, so part of any size gap is baseline over-coverage)
apples-to-apples across constructions with fundamentally different set SHAPES (a disk
vs a box), :func:`set_size_cxcy_slice` measures every construction on the SAME
quantity: the AREA of the coverage region's (cx, cy) slice at the predicted shape held
fixed (w, h, theta = pred's own values). This is well-defined for every construction
here (closed form for gwd/naive-coord/hull/wrapped-coord/doubled; ``iou``'s slice has
no closed form and is intentionally left out of the exact-formula path -- it is
exploratory only, never in the confirmatory Holm-8, design §4.5).

B1 sharpened variant: aleatoric-scaled naive-coord (arXiv:2605.07549)
----------------------------------------------------------------------
Ries, Kassem Sbeyti, Bianco & Klein, "Probabilistic Object Detection with Conformal
Prediction," arXiv:2605.07549 (May 2026), sharpen the corner-wise/Bonferroni B1
lineage by scaling each coordinate residual by a per-DETECTION aleatoric-uncertainty
estimate BEFORE ranking: ``r_i = |coord_pred - coord_gt| / sigma_i``, with ``sigma_i``
coming from the detector's own loss-attenuation head (a learned per-coordinate
predictive-variance output), then running the same per-coordinate Bonferroni split
conformal on the SCALED residuals. :class:`ScaledBonferroniBoxScore` implements this
normalize-then-Bonferroni construction generically over any of B1's five coordinates.

**Faithfulness delta** (documented per the K6 survey's baseline-strength risk, design
§4.4): this repo's detectors are FROZEN zoo checkpoints (RTMDet-R / Oriented R-CNN,
design §4.1) with no loss-attenuation head, so there is no learned per-coordinate
sigma to read off. We substitute the one per-detection aleatoric-width PROXY available
at score-only access: ``sigma_i = 1 / max(score_i, score_floor)``
(:func:`aleatoric_sigma_from_score`) -- sigma grows as detector confidence falls,
floored at ``score_floor`` (:data:`DEFAULT_ALEATORIC_SCORE_FLOOR`) so sigma stays
finite (never a division by zero) when a score underflows to 0. This is a strictly
weaker signal than the published loss-attenuation sigma -- one scalar confidence
value stands in for a learned per-coordinate predictive covariance -- so what is
implemented here is the CLOSEST FAITHFUL version reachable without retraining or
reimplementing the detector's head, not a literal reproduction of 2605.07549. The
unscaled legacy construction remains the DEFAULT B1 (``SCORES["naive-coord"]`` /
:data:`naive_coord_score`, selected by :func:`b1_score` with ``scaled=False``) so no
preregistered comparison changes silently; the scaled flavor is opt-in via
``b1_score(scaled=True)`` / :data:`naive_coord_score_scaled`. Neither
``ScaledBonferroniBoxScore`` nor its instances are registered in :data:`SCORES` --
they are reached only through :func:`b1_score` or direct import, so the certify/audit/
CLI pipelines (which drive the six :data:`SCORES` entries by name) are unaffected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from relmetrics import conformal as _conformal

from rotcert.gwd import canonicalize_le90, obb_gwd
from rotcert.matching import obb_to_polygon, rotated_iou

__all__ = [
    "wrapped_angle_distance",
    "ScalarCalibrator",
    "ScalarScore",
    "BonferroniCalibrator",
    "BonferroniBoxScore",
    "gwd_score",
    "iou_score",
    "naive_coord_score",
    "hull_score",
    "wrapped_coord_score",
    "doubled_angle_score",
    "SCORES",
    "set_size_cxcy_slice",
    "DEFAULT_ALEATORIC_SCORE_FLOOR",
    "aleatoric_sigma_from_score",
    "ScaledBonferroniCalibrator",
    "ScaledBonferroniBoxScore",
    "naive_coord_score_scaled",
    "b1_score",
    "set_size_cxcy_slice_scaled",
    "EXPERIMENTAL_SCORES",
]


def wrapped_angle_distance(theta_a: np.ndarray, theta_b: np.ndarray) -> np.ndarray:
    """Geodesic distance on the ``R / pi*Z`` quotient: ``min(|d| mod pi, pi - |d| mod
    pi)``, always in ``[0, pi/2]``. This is the "minimal fix" for angle periodicity
    (design §2.1's "textbook PoA problem"; A1's whole reason to exist)."""
    theta_a = np.asarray(theta_a, dtype=float)
    theta_b = np.asarray(theta_b, dtype=float)
    d = np.mod(np.abs(theta_a - theta_b), np.pi)
    return np.minimum(d, np.pi - d)


def _hull_box(obb: np.ndarray) -> np.ndarray:
    """Axis-aligned hull of one OBB as ``(cx, cy, half_w, half_h)``."""
    poly = obb_to_polygon(obb)
    minx, miny, maxx, maxy = poly.bounds
    return np.array([(minx + maxx) / 2.0, (miny + maxy) / 2.0, (maxx - minx) / 2.0, (maxy - miny) / 2.0])


# ---------------------------------------------------------------------------
# Scalar constructions (gwd, iou): single nonconformity number per pair.
# ---------------------------------------------------------------------------


@dataclass
class ScalarCalibrator:
    q_hat: float
    alpha: float
    n_cal: int


class ScalarScore:
    """One scalar nonconformity per (pred, gt) pair, calibrated via
    ``relmetrics.conformal.SplitConformal`` (deterministic threshold -- design §2.3's
    ``q_hat = ceil((1-alpha)(n_cal+1))``-th order statistic)."""

    name: str

    def __init__(self, name: str, residual_fn: Callable[[np.ndarray, np.ndarray], float]):
        self.name = name
        self._residual_fn = residual_fn

    def residual(self, pred_obb: np.ndarray, gt_obb: np.ndarray) -> float:
        return float(self._residual_fn(np.asarray(pred_obb, dtype=float), np.asarray(gt_obb, dtype=float)))

    def calibrate(self, cal_residuals: np.ndarray, alpha: float) -> ScalarCalibrator:
        cal_residuals = np.asarray(cal_residuals, dtype=float)
        sc = _conformal.SplitConformal(alpha=alpha, randomize=False).fit(cal_residuals)
        return ScalarCalibrator(q_hat=sc.threshold, alpha=alpha, n_cal=sc.n_cal)

    def covers(self, calibrator: ScalarCalibrator, pred_obb: np.ndarray, gt_obb: np.ndarray) -> bool:
        return self.residual(pred_obb, gt_obb) <= calibrator.q_hat


def _gwd_residual(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(obb_gwd(pred, gt))


def _iou_residual(pred: np.ndarray, gt: np.ndarray) -> float:
    return 1.0 - rotated_iou(pred, gt)


gwd_score = ScalarScore("gwd", _gwd_residual)
iou_score = ScalarScore("iou", _iou_residual)


# ---------------------------------------------------------------------------
# Bonferroni box constructions (naive-coord, hull, wrapped-coord, doubled).
# ---------------------------------------------------------------------------


@dataclass
class BonferroniCalibrator:
    q: Dict[str, float]
    alpha: float
    alpha_per_coord: float
    n_cal: int
    coord_names: Tuple[str, ...]


class BonferroniBoxScore:
    """K separate per-coordinate split-conformal intervals, each at level
    ``alpha / K`` (literal Bonferroni -- see module docstring for why this, not a
    max-normalized scalar, is the correct baseline construction).
    """

    def __init__(
        self,
        name: str,
        coord_names: Tuple[str, ...],
        residual_fn: Callable[[np.ndarray, np.ndarray], Dict[str, float]],
        cxcy_box_coords: Optional[Tuple[str, str]] = ("cx", "cy"),
    ):
        self.name = name
        self.coord_names = coord_names
        self._residual_fn = residual_fn
        self._cxcy_box_coords = cxcy_box_coords

    def residuals(self, pred_obb: np.ndarray, gt_obb: np.ndarray) -> Dict[str, float]:
        return self._residual_fn(np.asarray(pred_obb, dtype=float), np.asarray(gt_obb, dtype=float))

    def calibrate(
        self, cal_pred: np.ndarray, cal_gt: np.ndarray, alpha: float
    ) -> BonferroniCalibrator:
        cal_pred = np.asarray(cal_pred, dtype=float)
        cal_gt = np.asarray(cal_gt, dtype=float)
        n = len(cal_pred)
        k = len(self.coord_names)
        alpha_k = alpha / k
        per_coord: Dict[str, List[float]] = {c: [] for c in self.coord_names}
        for i in range(n):
            res = self.residuals(cal_pred[i], cal_gt[i])
            for c in self.coord_names:
                per_coord[c].append(res[c])
        q: Dict[str, float] = {}
        for c in self.coord_names:
            sc = _conformal.SplitConformal(alpha=alpha_k, randomize=False).fit(np.array(per_coord[c]))
            q[c] = sc.threshold
        return BonferroniCalibrator(
            q=q, alpha=float(alpha), alpha_per_coord=float(alpha_k), n_cal=n, coord_names=self.coord_names
        )

    def covers(self, calibrator: BonferroniCalibrator, pred_obb: np.ndarray, gt_obb: np.ndarray) -> bool:
        res = self.residuals(pred_obb, gt_obb)
        return all(res[c] <= calibrator.q[c] for c in self.coord_names)

    def cxcy_box_halfwidths(self, calibrator: BonferroniCalibrator) -> Optional[Tuple[float, float]]:
        """``(q_cx, q_cy)`` for the (cx, cy) box half-widths, or ``None`` if this
        construction's coordinates don't include a translation pair (none currently
        lack one, but kept optional for future non-translation constructions)."""
        if self._cxcy_box_coords is None:
            return None
        cx_name, cy_name = self._cxcy_box_coords
        return calibrator.q[cx_name], calibrator.q[cy_name]


def _naive_coord_residuals(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    wp, hp, tp = canonicalize_le90(pred[2], pred[3], pred[4])
    wg, hg, tg = canonicalize_le90(gt[2], gt[3], gt[4])
    return {
        "cx": abs(float(pred[0] - gt[0])),
        "cy": abs(float(pred[1] - gt[1])),
        "w": abs(float(wp - wg)),
        "h": abs(float(hp - hg)),
        "theta": abs(float(tp - tg)),  # Euclidean, NOT wrapped -- this is the pathology B1 tests.
    }


def _hull_residuals(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    hp = _hull_box(pred)
    hg = _hull_box(gt)
    return {
        "cx": abs(float(hp[0] - hg[0])),
        "cy": abs(float(hp[1] - hg[1])),
        "half_w": abs(float(hp[2] - hg[2])),
        "half_h": abs(float(hp[3] - hg[3])),
    }


def _wrapped_coord_residuals(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    wp, hp, tp = canonicalize_le90(pred[2], pred[3], pred[4])
    wg, hg, tg = canonicalize_le90(gt[2], gt[3], gt[4])
    return {
        "cx": abs(float(pred[0] - gt[0])),
        "cy": abs(float(pred[1] - gt[1])),
        "w": abs(float(wp - wg)),
        "h": abs(float(hp - hg)),
        "theta": float(wrapped_angle_distance(tp, tg)),  # the periodicity fix.
    }


def _doubled_angle_residuals(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    wp, hp, tp = canonicalize_le90(pred[2], pred[3], pred[4])
    wg, hg, tg = canonicalize_le90(gt[2], gt[3], gt[4])
    return {
        "cx": abs(float(pred[0] - gt[0])),
        "cy": abs(float(pred[1] - gt[1])),
        "w": abs(float(wp - wg)),
        "h": abs(float(hp - hg)),
        "cos2t": abs(float(np.cos(2 * tp) - np.cos(2 * tg))),
        "sin2t": abs(float(np.sin(2 * tp) - np.sin(2 * tg))),
    }


naive_coord_score = BonferroniBoxScore(
    "naive-coord", ("cx", "cy", "w", "h", "theta"), _naive_coord_residuals
)
hull_score = BonferroniBoxScore("hull", ("cx", "cy", "half_w", "half_h"), _hull_residuals)
wrapped_coord_score = BonferroniBoxScore(
    "wrapped-coord", ("cx", "cy", "w", "h", "theta"), _wrapped_coord_residuals
)
doubled_angle_score = BonferroniBoxScore(
    "doubled", ("cx", "cy", "w", "h", "cos2t", "sin2t"), _doubled_angle_residuals
)


SCORES: Dict[str, Any] = {
    "gwd": gwd_score,
    "iou": iou_score,
    "naive-coord": naive_coord_score,
    "hull": hull_score,
    "wrapped-coord": wrapped_coord_score,
    "doubled": doubled_angle_score,
}


# ---------------------------------------------------------------------------
# B1 sharpened variant: aleatoric-scaled naive-coord (arXiv:2605.07549).
# Deliberately NOT registered in SCORES -- reached only via b1_score()/direct
# import, so the six-construction certify/audit/CLI pipeline is unaffected.
# See module docstring "B1 sharpened variant" section for the construction and
# the faithfulness delta from the published loss-attenuation version.
# ---------------------------------------------------------------------------

DEFAULT_ALEATORIC_SCORE_FLOOR: float = 1e-3
"""Floor on detector confidence score before inverting to the aleatoric-sigma proxy
(:func:`aleatoric_sigma_from_score`) -- keeps sigma finite (never a division by zero)
for a detection whose score underflows to, or is recorded as, exactly 0."""


def aleatoric_sigma_from_score(
    pred_score: float, score_floor: float = DEFAULT_ALEATORIC_SCORE_FLOOR
) -> float:
    """Score-based aleatoric-width proxy: ``sigma = 1 / max(score, score_floor)``.

    Substitutes for the learned per-coordinate loss-attenuation sigma of
    arXiv:2605.07549, which is unavailable for our frozen zoo checkpoints (see module
    docstring). Monotone DECREASING in score -- a confident detection gets a small
    sigma (tight scaled residual, tight resulting interval), a low-confidence
    detection gets a large one -- and bounded on ``(0, 1/score_floor]`` since
    ``score in (0, 1]`` for a real detection.
    """
    return 1.0 / max(float(pred_score), float(score_floor))


@dataclass
class ScaledBonferroniCalibrator:
    q: Dict[str, float]
    alpha: float
    alpha_per_coord: float
    n_cal: int
    coord_names: Tuple[str, ...]
    score_floor: float


class ScaledBonferroniBoxScore:
    """Aleatoric-scaled sibling of :class:`BonferroniBoxScore`: each coordinate
    residual is divided by a per-detection sigma (:func:`aleatoric_sigma_from_score`
    by default) before the per-coordinate Bonferroni split-conformal quantile is
    taken, and covers()/the live half-width are computed on the same scaled residual.
    The scaled construction reuses the SAME raw ``residual_fn`` as its unscaled
    counterpart (no duplicated residual geometry) -- only the normalization differs.
    """

    def __init__(
        self,
        name: str,
        coord_names: Tuple[str, ...],
        residual_fn: Callable[[np.ndarray, np.ndarray], Dict[str, float]],
        cxcy_box_coords: Optional[Tuple[str, str]] = ("cx", "cy"),
        score_floor: float = DEFAULT_ALEATORIC_SCORE_FLOOR,
        sigma_fn: Callable[[float, float], float] = aleatoric_sigma_from_score,
    ):
        self.name = name
        self.coord_names = coord_names
        self._residual_fn = residual_fn
        self._cxcy_box_coords = cxcy_box_coords
        self.score_floor = float(score_floor)
        self._sigma_fn = sigma_fn

    def residuals(self, pred_obb: np.ndarray, gt_obb: np.ndarray) -> Dict[str, float]:
        return self._residual_fn(np.asarray(pred_obb, dtype=float), np.asarray(gt_obb, dtype=float))

    def sigma(self, pred_score: float) -> float:
        return float(self._sigma_fn(pred_score, self.score_floor))

    def calibrate(
        self,
        cal_pred: np.ndarray,
        cal_gt: np.ndarray,
        cal_score: np.ndarray,
        alpha: float,
    ) -> ScaledBonferroniCalibrator:
        cal_pred = np.asarray(cal_pred, dtype=float)
        cal_gt = np.asarray(cal_gt, dtype=float)
        cal_score = np.asarray(cal_score, dtype=float)
        n = len(cal_pred)
        if len(cal_score) != n:
            raise ValueError(
                "ScaledBonferroniBoxScore.calibrate: cal_score length "
                f"{len(cal_score)} != cal_pred/cal_gt length {n}"
            )
        k = len(self.coord_names)
        alpha_k = alpha / k
        per_coord: Dict[str, List[float]] = {c: [] for c in self.coord_names}
        for i in range(n):
            res = self.residuals(cal_pred[i], cal_gt[i])
            sigma_i = self.sigma(cal_score[i])
            for c in self.coord_names:
                per_coord[c].append(res[c] / sigma_i)
        q: Dict[str, float] = {}
        for c in self.coord_names:
            sc = _conformal.SplitConformal(alpha=alpha_k, randomize=False).fit(np.array(per_coord[c]))
            q[c] = sc.threshold
        return ScaledBonferroniCalibrator(
            q=q,
            alpha=float(alpha),
            alpha_per_coord=float(alpha_k),
            n_cal=n,
            coord_names=self.coord_names,
            score_floor=self.score_floor,
        )

    def covers(
        self,
        calibrator: ScaledBonferroniCalibrator,
        pred_obb: np.ndarray,
        gt_obb: np.ndarray,
        pred_score: float,
    ) -> bool:
        res = self.residuals(pred_obb, gt_obb)
        sigma = self.sigma(pred_score)
        return all(res[c] / sigma <= calibrator.q[c] for c in self.coord_names)

    def cxcy_box_halfwidths(
        self, calibrator: ScaledBonferroniCalibrator, pred_score: float
    ) -> Optional[Tuple[float, float]]:
        """``(q_cx, q_cy)`` half-widths for THIS detection's own score -- unlike the
        unscaled construction, the box is not the same size for every detection."""
        if self._cxcy_box_coords is None:
            return None
        cx_name, cy_name = self._cxcy_box_coords
        sigma = self.sigma(pred_score)
        return calibrator.q[cx_name] * sigma, calibrator.q[cy_name] * sigma


naive_coord_score_scaled = ScaledBonferroniBoxScore(
    "naive-coord-scaled", ("cx", "cy", "w", "h", "theta"), _naive_coord_residuals
)


EXPERIMENTAL_SCORES: Dict[str, Any] = {
    "naive-coord-scaled": naive_coord_score_scaled,
}
"""Explicit OPT-IN registry for exploratory score constructions.

Deliberately separate from :data:`SCORES`: the preregistered six-construction
roster (the audit Holm family, every default) iterates ``SCORES`` and is
unchanged. ``g1_calibrate``/``g1_coverage``/the CLI ``calibrate`` command
resolve these names only when the caller asks for one BY NAME; whether
``naive-coord-scaled`` enters the confirmatory roster (vs staying a disclosed
exploratory arm) is a prereg-freeze decision recorded in the protocol
specification addendum (2026-07-10)."""


def b1_score(scaled: bool = False) -> Any:
    """Select the B1 baseline flavor by flag.

    ``scaled=False`` (default) returns the legacy unscaled construction --
    :data:`naive_coord_score`, the same object already registered as
    ``SCORES["naive-coord"]`` -- so nothing preregistered changes silently.
    ``scaled=True`` returns the aleatoric-scaled sharpened construction
    (arXiv:2605.07549, see module docstring), :data:`naive_coord_score_scaled`.
    """
    return naive_coord_score_scaled if scaled else naive_coord_score


def set_size_cxcy_slice_scaled(calibrator: ScaledBonferroniCalibrator, pred_score: float) -> float:
    """Area of the scaled-B1 coverage region's ``(cx, cy)`` slice for ONE detection at
    its own score -- the scaled-B1 analogue of :func:`set_size_cxcy_slice`, kept
    separate because the scaled region's half-widths are score-dependent (not fixed
    across detections the way the unscaled B1's are)."""
    sigma = aleatoric_sigma_from_score(pred_score, calibrator.score_floor)
    return float(4.0 * (calibrator.q["cx"] * sigma) * (calibrator.q["cy"] * sigma))


def set_size_cxcy_slice(score_name: str, calibrator: Any, pred_obb: Optional[np.ndarray] = None) -> Optional[float]:
    """Area of the coverage region's ``(cx, cy)`` slice at fixed predicted shape (the
    C2 head-to-head efficiency metric, module docstring). Returns ``None`` for ``iou``
    (no closed form; exploratory-only, never in the confirmatory Holm-8)."""
    if score_name == "gwd":
        q = calibrator.q_hat
        return float(math.pi * q * q)
    if score_name in ("naive-coord", "wrapped-coord", "doubled"):
        return float(4.0 * calibrator.q["cx"] * calibrator.q["cy"])
    if score_name == "hull":
        return float(4.0 * calibrator.q["cx"] * calibrator.q["cy"])
    if score_name == "iou":
        return None
    raise ValueError(f"set_size_cxcy_slice: unknown score {score_name!r}")

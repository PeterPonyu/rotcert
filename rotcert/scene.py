"""Scene-level conformal calibration for multi-object localization (revision spec §3 C1).

The exchangeable unit is the scene (source image); objects inside a scene may be
arbitrarily dependent. ``groups`` is a sequence of 1-D score arrays, one per scene;
empty scenes are allowed and carry no mass.

* :func:`pooled_threshold` -- legacy object-pooled split conformal (reviewed version);
  no finite-sample guarantee under within-scene dependence.
* :func:`hcp_threshold` -- hierarchical conformal prediction (Lee, Barber & Willett):
  scene-weighted per-object coverage >= 1 - alpha (Theorem 1).
* :func:`crc_object_threshold` -- conformal risk control with a known bound ``M`` on
  objects per scene: object-weighted miscoverage <= alpha (Theorem 2).
* :func:`crc_object_adaptive_threshold` -- scene-adaptive CRC (CRC-D): ``M`` is replaced
  by the deployment scene's own observable detection count ``d_test``; object-weighted
  miscoverage <= alpha with no bound on calibration scene sizes (Theorem 2b).
* :func:`scene_max_threshold` -- split conformal on per-scene maxima: every object of a
  new scene covered with probability >= 1 - alpha (Proposition 3).

Every threshold is ``math.inf`` when the requested level is unattainable.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "SceneCalibrationError",
    "conformal_rank",
    "pooled_threshold",
    "hcp_threshold",
    "crc_object_threshold",
    "crc_object_adaptive_threshold",
    "crc_object_adaptive_thresholds",
    "scene_max_threshold",
    "coverage_scene_weighted",
    "coverage_object_weighted",
    "coverage_scene_simultaneous",
    "grouped",
    "class_thresholds",
    "evaluate_q",
    "evaluate",
    "THRESHOLDS",
]

_TOL = 1e-9


class SceneCalibrationError(ValueError):
    """Raised on invalid scene-level calibration inputs."""


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise SceneCalibrationError(f"alpha must be in (0, 1), got {alpha!r}")


def _check_aligned(scores: np.ndarray, scene_codes: np.ndarray, class_codes: np.ndarray) -> None:
    if not (scores.shape == scene_codes.shape == class_codes.shape):
        raise SceneCalibrationError(
            "scores, scene_codes, and class_codes must have the same shape; got "
            f"{scores.shape}, {scene_codes.shape}, {class_codes.shape}"
        )


def _check_integer_codes(class_codes: np.ndarray) -> None:
    if class_codes.size and not np.all(class_codes == np.floor(class_codes)):
        raise SceneCalibrationError("class_codes must be integer-valued")


def _as_groups(groups: Sequence[Sequence[float]]) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for g in groups:
        a = np.asarray(g, dtype=float)
        if a.ndim == 0:
            raise SceneCalibrationError(
                "each group must be a 1-D array of scores for one scene; got a "
                "scalar — did you pass a flat pooled score array?"
            )
        a = a.ravel()
        if a.size and not np.all(np.isfinite(a)):
            raise SceneCalibrationError("scores must be finite")
        out.append(a)
    return out


def conformal_rank(n: int, alpha: float) -> int:
    """``ceil((n+1)(1-alpha))`` as ``(n+1) - floor((n+1)*alpha)`` (``alpha`` is cast to
    ``float`` first, so a lower-precision dtype such as ``np.float32`` can't shift the
    floored rank): exact whenever ``(n+1)*alpha`` is computed exactly; otherwise at most
    one rank conservative -- identical to ``relmetrics.conformal.SplitConformal.threshold``."""
    alpha = float(alpha)
    return (n + 1) - int(np.floor((n + 1) * alpha))


def pooled_threshold(groups: Sequence[Sequence[float]], alpha: float) -> float:
    alpha = float(alpha)
    _check_alpha(alpha)
    nonempty = [g for g in _as_groups(groups) if g.size]
    if not nonempty:
        return math.inf
    s = np.sort(np.concatenate(nonempty))
    k = conformal_rank(s.size, alpha)
    return math.inf if k > s.size else float(s[k - 1])


def hcp_threshold(groups: Sequence[Sequence[float]], alpha: float) -> float:
    """(1-alpha)-quantile of sum_i sum_j delta_{s_ij}/((K+1) m_i) + delta_{+inf}/(K+1)."""
    alpha = float(alpha)
    _check_alpha(alpha)
    nonempty = [g for g in _as_groups(groups) if g.size]
    k_scenes = len(nonempty)
    if k_scenes == 0:
        return math.inf
    scores = np.concatenate(nonempty)
    weights = np.concatenate([np.full(g.size, 1.0 / g.size) for g in nonempty])  # x (K+1)
    order = np.argsort(scores, kind="mergesort")
    cum = np.cumsum(weights[order])
    target = (1.0 - alpha) * (k_scenes + 1)
    idx = int(np.searchsorted(cum, target - _TOL, side="left"))
    return math.inf if idx >= scores.size else float(scores[order][idx])


def crc_object_threshold(groups: Sequence[Sequence[float]], alpha: float, M: int) -> float:
    """Smallest q with #{s > q} <= alpha*N - (1-alpha)*M over the N calibration objects.

    Valid when every scene (calibration and test) holds at most ``M`` objects."""
    alpha = float(alpha)
    _check_alpha(alpha)
    if int(M) < 1:
        raise SceneCalibrationError("M must be >= 1")
    gs = _as_groups(groups)
    largest = max((g.size for g in gs), default=0)
    if largest > M:
        raise SceneCalibrationError(f"a calibration scene holds {largest} > M={M} objects")
    nonempty = [g for g in gs if g.size]
    if not nonempty:
        return math.inf
    s = np.sort(np.concatenate(nonempty))
    n = s.size
    budget = alpha * n - (1.0 - alpha) * M
    if budget < -_TOL:
        return math.inf
    allowed = min(int(np.floor(budget + _TOL)), n)
    k = n - allowed
    return -math.inf if k <= 0 else float(s[k - 1])


def crc_object_adaptive_threshold(groups: Sequence[Sequence[float]], alpha: float, d_test: int) -> float:
    """Scene-adaptive CRC (Theorem 2b): object-weighted miscoverage <= alpha when the
    deployment scene's TP count is at most ``d_test`` (e.g. its detection count)."""
    alpha = float(alpha)
    _check_alpha(alpha)
    if not np.isfinite(d_test) or int(d_test) != d_test or d_test < 0:
        raise SceneCalibrationError("d_test must be a non-negative integer")
    nonempty = [g for g in _as_groups(groups) if g.size]
    if not nonempty:
        return math.inf
    s = np.sort(np.concatenate(nonempty))
    n = s.size
    budget = alpha * n - (1.0 - alpha) * int(d_test)
    if budget < -_TOL:
        return math.inf
    allowed = min(int(np.floor(budget + _TOL)), n)
    k = n - allowed
    return -math.inf if k <= 0 else float(s[k - 1])


def crc_object_adaptive_thresholds(groups: Sequence[Sequence[float]], alpha: float, d_tests) -> np.ndarray:
    """Vectorized :func:`crc_object_adaptive_threshold` over deployment-scene counts ``d_tests``."""
    alpha = float(alpha)
    _check_alpha(alpha)
    d = np.asarray(d_tests)
    if d.size and (np.any(~np.isfinite(d)) or np.any(d < 0) or np.any(np.floor(d) != d)):
        raise SceneCalibrationError("d_tests must be non-negative integers")
    nonempty = [g for g in _as_groups(groups) if g.size]
    if not nonempty:
        return np.full(d.shape, math.inf)
    s = np.sort(np.concatenate(nonempty))
    n = s.size
    budget = alpha * n - (1.0 - alpha) * d.astype(float)
    allowed = np.minimum(np.floor(budget + _TOL), n).astype(np.int64)
    k = n - allowed
    out = np.where(k >= 1, s[np.clip(k - 1, 0, n - 1)], -math.inf)
    return np.where(budget < -_TOL, math.inf, out).astype(float)


def scene_max_threshold(groups: Sequence[Sequence[float]], alpha: float) -> float:
    alpha = float(alpha)
    _check_alpha(alpha)
    maxima = np.array([g.max() for g in _as_groups(groups) if g.size], dtype=float)
    if maxima.size == 0:
        return math.inf
    maxima.sort()
    k = conformal_rank(maxima.size, alpha)
    return math.inf if k > maxima.size else float(maxima[k - 1])


def coverage_scene_weighted(groups: Sequence[Sequence[float]], q: float) -> float:
    fr = [float(np.mean(g <= q)) for g in _as_groups(groups) if g.size]
    return float(np.mean(fr)) if fr else math.nan


def coverage_object_weighted(groups: Sequence[Sequence[float]], q: float) -> float:
    nonempty = [g for g in _as_groups(groups) if g.size]
    if not nonempty:
        return math.nan
    return float(np.mean(np.concatenate(nonempty) <= q))


def coverage_scene_simultaneous(groups: Sequence[Sequence[float]], q: float) -> float:
    fr = [float(np.all(g <= q)) for g in _as_groups(groups) if g.size]
    return float(np.mean(fr)) if fr else math.nan


THRESHOLDS = {
    "pooled": pooled_threshold,
    "hcp": hcp_threshold,
    "crc_object": crc_object_threshold,
    "scene_max": scene_max_threshold,
}


def grouped(scores: np.ndarray, scene_codes: np.ndarray) -> List[np.ndarray]:
    """Split a flat score array into per-scene arrays, ordered by ascending scene code."""
    scores = np.asarray(scores, dtype=float)
    scene_codes = np.asarray(scene_codes)
    if scores.ndim != 1 or scene_codes.ndim != 1:
        raise SceneCalibrationError(
            "scores and scene_codes must be 1-D; got shapes "
            f"{scores.shape} and {scene_codes.shape}"
        )
    if scores.shape != scene_codes.shape:
        raise SceneCalibrationError("scores and scene_codes must have the same shape")
    if scores.size == 0:
        return []
    order = np.argsort(scene_codes, kind="mergesort")
    sc = scene_codes[order]
    cuts = np.flatnonzero(sc[1:] != sc[:-1]) + 1
    return np.split(scores[order], cuts)


def _threshold(method: str, groups: List[np.ndarray], alpha: float, M: Optional[int]) -> float:
    if method not in THRESHOLDS:
        raise SceneCalibrationError(f"unknown method {method!r}; choose from {sorted(THRESHOLDS)}")
    if method == "crc_object":
        # M-is-None is validated up front by class_thresholds(); not re-checked here.
        return crc_object_threshold(groups, alpha, M)
    return THRESHOLDS[method](groups, alpha)


def class_thresholds(
    scores: np.ndarray,
    scene_codes: np.ndarray,
    class_codes: np.ndarray,
    alpha: float,
    method: str,
    M: Optional[int] = None,
) -> Dict[int, Dict[str, Any]]:
    """Mondrian-by-class thresholds with the chosen scene-level method.

    ``method`` and ``alpha`` are validated up front, before the per-class loop,
    so an empty input can never mask a bad argument by returning ``{}``."""
    if method not in THRESHOLDS:
        raise SceneCalibrationError(f"unknown method {method!r}; choose from {sorted(THRESHOLDS)}")
    alpha = float(alpha)
    _check_alpha(alpha)
    if method == "crc_object" and M is None:
        raise SceneCalibrationError("crc_object needs the scene-size bound M")
    scores = np.asarray(scores, dtype=float)
    scene_codes = np.asarray(scene_codes)
    class_codes = np.asarray(class_codes)
    _check_aligned(scores, scene_codes, class_codes)
    _check_integer_codes(class_codes)
    out: Dict[int, Dict[str, Any]] = {}
    for c in np.unique(class_codes):
        m = class_codes == c
        groups = grouped(scores[m], scene_codes[m])
        q = _threshold(method, groups, alpha, M)
        out[int(c)] = {
            "q_hat": q,
            "n_scenes": len(groups),
            "n_objects": int(m.sum()),
            # q < inf (not math.isfinite(q)) so the CRC -inf branch -- a
            # trivially-certified, everything-fits budget, not a failure --
            # can never misread as "support_floor".
            "status": "certified" if q < math.inf else "support_floor",
        }
    return out


def evaluate_q(
    scores: np.ndarray,
    scene_codes: np.ndarray,
    class_codes: np.ndarray,
    q_obj: np.ndarray,
) -> Dict[str, Any]:
    """Like :func:`evaluate` but with a per-object threshold array (NaN = out of support).

    ``q_obj`` carries one threshold per object (e.g. from :func:`crc_object_adaptive_thresholds`,
    a per-scene-adaptive certificate) instead of a single per-class lookup."""
    scores = np.asarray(scores, dtype=float)
    scene_codes = np.asarray(scene_codes)
    class_codes = np.asarray(class_codes)
    _check_aligned(scores, scene_codes, class_codes)
    _check_integer_codes(class_codes)
    if scores.size and not np.all(np.isfinite(scores)):
        raise SceneCalibrationError("scores must be finite")
    qv = np.asarray(q_obj, dtype=float)
    if qv.shape != scores.shape:
        raise SceneCalibrationError(
            f"q_obj must have the same shape as scores; got {qv.shape}, expected {scores.shape}"
        )
    in_sup = ~np.isnan(qv)
    covered = np.zeros(scores.size, dtype=float)
    covered[in_sup] = (scores[in_sup] <= qv[in_sup]).astype(float)
    per_class: Dict[int, Dict[str, Any]] = {}
    for c in np.unique(class_codes[in_sup]):
        m = in_sup & (class_codes == c)
        groups = grouped(covered[m], scene_codes[m])
        per_class[int(c)] = {
            "cov_scene": float(np.mean([g.mean() for g in groups])),
            "cov_object": float(covered[m].mean()),
            "cov_sim": float(np.mean([g.min() for g in groups])),
            "n_scenes": len(groups),
            "n_objects": int(m.sum()),
        }
    if in_sup.any():
        groups = grouped(covered[in_sup], scene_codes[in_sup])
        overall = {
            "cov_scene": float(np.mean([g.mean() for g in groups])),
            "cov_object": float(covered[in_sup].mean()),
            "cov_sim": float(np.mean([g.min() for g in groups])),
        }
    else:
        overall = {"cov_scene": math.nan, "cov_object": math.nan, "cov_sim": math.nan}
    return {"per_class": per_class, "overall": overall, "n_out_of_support": int((~in_sup).sum())}


def evaluate(
    scores: np.ndarray,
    scene_codes: np.ndarray,
    class_codes: np.ndarray,
    q_by_class: Dict[int, float],
) -> Dict[str, Any]:
    """Coverage of per-class thresholds on evaluation objects.

    Objects whose class has no threshold are out of support (excluded, counted);
    an infinite threshold covers everything (reported, never hidden)."""
    scores = np.asarray(scores, dtype=float)
    scene_codes = np.asarray(scene_codes)
    class_codes = np.asarray(class_codes)
    # Validate shapes/codes *before* building q_obj: otherwise a NaN or non-scalar
    # class code raises a bare ValueError/TypeError out of the list comprehension
    # below instead of the intended SceneCalibrationError.
    _check_aligned(scores, scene_codes, class_codes)
    _check_integer_codes(class_codes)
    # A NaN value in q_by_class is a caller error, not "out of support" -- the
    # NaN sentinel for "out of support" is reserved for classes absent from the
    # dict (evaluate_q's own per-object API still treats a NaN q_obj entry as
    # out of support, since there the caller supplies one threshold per object).
    if any(math.isnan(v) for v in q_by_class.values()):
        raise SceneCalibrationError("q_by_class values must not be NaN")
    q_obj = np.array([q_by_class.get(int(c), math.nan) for c in class_codes], dtype=float)
    return evaluate_q(scores, scene_codes, class_codes, q_obj)

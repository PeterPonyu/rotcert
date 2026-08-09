"""G1 (per-detection coverage) + G2 (certified rotated-IoU FNR) certification, and the
honest-uncertainty refusal rules that gate both (design §2.3, §2.4, §3.3).

G1 -- per-Mondrian-cell split conformal (design §2.3)
--------------------------------------------------------
:func:`g1_calibrate` fits one of the six :mod:`rotcert.scores` constructions per
Mondrian stratum (typically class; any field works) on MATCHED true-positive pairs
only (design's "conditional-on-detection" caveat -- unmatched GT/detections are G2's
job, never G1's). A stratum whose calibration size violates the certifiability floor
``alpha_min = 1 / (n_cal + 1) > alpha`` is REFUSED (design §3.3 refusal table row 1):
no certificate for that stratum, recorded loudly in ``refused``, never silently pooled
into a neighboring stratum.

G2 -- certified image-level FNR via LTT-HB (design §2.4)
-------------------------------------------------------------
:func:`g2_certify_fnr` operates on already SCENE-aggregated match data (design's "the
exchangeable unit for G2 is the IMAGE," M3): for each scene, the list of matched
confidences for its ground-truth boxes (``None`` for a GT never matched at any
confidence). It builds the per-scene, per-candidate-lambda risk matrix and calls
:func:`rotcert.ltt.ltt_certify_matrix`, gated by the a-priori power floor
(:func:`rotcert.ltt.power_floor_n_img`) -- BELOW the floor, the function refuses the
per-class certificate and (when ``pooled_fallback=True``) recommends falling back to
the pooled-marginal FNR (design §2.4/§5 K5's preregistered remedy), never silently
returning an uncertifiable number.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from relmetrics import provenance as _provenance

from rotcert import ltt as _ltt
from rotcert.scores import (
    EXPERIMENTAL_SCORES,
    SCORES,
    BonferroniBoxScore,
    ScalarScore,
    ScaledBonferroniBoxScore,
    set_size_cxcy_slice,
    set_size_cxcy_slice_scaled,
)

__all__ = [
    "CertifyError",
    "g1_calibrate",
    "g1_coverage",
    "image_risk_matrix",
    "g2_certify_fnr",
    "g2_certify_fnr_mondrian",
]


class CertifyError(ValueError):
    """Raised on any certify.py precondition violation (bad score name, empty input)."""


def _resolve_score(score_name: str) -> Any:
    """Resolve a score by name: the preregistered :data:`SCORES` roster first,
    then the explicit opt-in :data:`EXPERIMENTAL_SCORES` (e.g.
    ``"naive-coord-scaled"`` -- exploratory, requires ``pred_score`` on every
    matched record, never enters a confirmatory family unless a prereg-freeze
    decision promotes it)."""
    if score_name in SCORES:
        return SCORES[score_name]
    if score_name in EXPERIMENTAL_SCORES:
        return EXPERIMENTAL_SCORES[score_name]
    raise CertifyError(
        f"unknown score {score_name!r}; choose from {sorted(SCORES)} "
        f"(preregistered) or {sorted(EXPERIMENTAL_SCORES)} (experimental, opt-in)"
    )


def _pred_scores_or_raise(rows: Sequence[Dict[str, Any]], where: str) -> np.ndarray:
    """Every scaled-score record must carry a real detector confidence."""
    vals = []
    for i, m in enumerate(rows):
        s = m.get("pred_score")
        if s is None:
            raise CertifyError(
                f"{where}: score 'naive-coord-scaled' needs a non-null 'pred_score' "
                f"on every matched record (row {i} has none) -- the aleatoric sigma "
                "proxy is 1/max(score, floor); see rotcert.scores module docstring"
            )
        vals.append(float(s))
    return np.asarray(vals, dtype=float)


# ---------------------------------------------------------------------------
# G1
# ---------------------------------------------------------------------------


def g1_calibrate(
    matched: Sequence[Dict[str, Any]],
    score_name: str,
    alpha: float = 0.10,
    mondrian_field: Optional[str] = None,
    scale_norm: Optional[str] = None,
) -> Dict[str, Any]:
    """Calibrate G1 for one score construction, per Mondrian stratum.

    Parameters
    ----------
    matched:
        Matched TRUE-POSITIVE pairs only (design's conditional-on-detection caveat),
        each ``{"pred_obb": [cx,cy,w,h,theta], "gt_obb": [...], <mondrian_field>: ...}``.
    score_name:
        One of ``rotcert.scores.SCORES`` (``"gwd"``, ``"naive-coord"``, ``"hull"``,
        ``"wrapped-coord"``, ``"doubled"``, ``"iou"``), or an explicit opt-in from
        ``rotcert.scores.EXPERIMENTAL_SCORES`` (``"naive-coord-scaled"`` -- the B1
        sharpened aleatoric-scaled variant; requires a non-null ``"pred_score"`` on
        every matched record and stays out of every confirmatory family unless a
        prereg-freeze decision promotes it).
    alpha:
        Target miscoverage.
    mondrian_field:
        Stratify by this key (e.g. ``"class"``); ``None`` = one marginal cell.
    scale_norm:
        Passed to ``gwd`` residual computation (``None`` or ``"sqrt-area"``); ignored
        for other scores.

    Returns
    -------
    dict
        ``score_name``, ``alpha``, ``mondrian_field``, ``strata`` (dict: stratum key ->
        ``{"calibrator", "n_cal", "set_size_cxcy"}``), ``refused`` (list of
        ``{"stratum", "n_cal", "alpha_min", "reason"}``), provenance-stamped.
    """
    score = _resolve_score(score_name)
    if not matched:
        raise CertifyError("g1_calibrate: matched must be non-empty")
    if not 0.0 < alpha < 1.0:
        raise CertifyError("g1_calibrate: alpha must be in (0, 1)")

    if mondrian_field is not None:
        strata_keys = sorted({m[mondrian_field] for m in matched})
    else:
        strata_keys = [None]

    strata: Dict[Any, Dict[str, Any]] = {}
    refused: List[Dict[str, Any]] = []
    for st in strata_keys:
        subset = matched if st is None else [m for m in matched if m[mondrian_field] == st]
        n_cal = len(subset)
        alpha_min = 1.0 / (n_cal + 1)
        if alpha_min > alpha:
            refused.append(
                {
                    "stratum": st,
                    "n_cal": n_cal,
                    "alpha_min": alpha_min,
                    "reason": (
                        f"certifiability floor alpha_min=1/(n_cal+1)={alpha_min:.4f} > "
                        f"requested alpha={alpha} (design §3.3 refusal table)"
                    ),
                }
            )
            continue

        preds = np.array([m["pred_obb"] for m in subset], dtype=float)
        gts = np.array([m["gt_obb"] for m in subset], dtype=float)

        if isinstance(score, ScaledBonferroniBoxScore):
            cal_scores = _pred_scores_or_raise(subset, "g1_calibrate")
            calibrator = score.calibrate(preds, gts, cal_scores, alpha)
            # The scaled region is per-detection (score-dependent), so there is
            # no single fixed cx-cy slice; report the median over the
            # calibration detections as the efficiency summary instead.
            strata[st] = {
                "calibrator": calibrator,
                "n_cal": n_cal,
                "set_size_cxcy": None,
                "set_size_cxcy_median_cal": float(
                    np.median([set_size_cxcy_slice_scaled(calibrator, s) for s in cal_scores])
                ),
            }
            continue
        if isinstance(score, BonferroniBoxScore):
            calibrator = score.calibrate(preds, gts, alpha)
        elif isinstance(score, ScalarScore):
            if score_name == "gwd" and scale_norm is not None:
                from rotcert.gwd import obb_gwd as _obb_gwd

                residuals = np.array([_obb_gwd(p, g, scale_norm=scale_norm) for p, g in zip(preds, gts)])
            else:
                residuals = np.array([score.residual(p, g) for p, g in zip(preds, gts)])
            calibrator = score.calibrate(residuals, alpha)
        else:  # pragma: no cover - the registries only contain the known types
            raise CertifyError(f"g1_calibrate: score {score_name!r} has an unrecognized type")

        strata[st] = {
            "calibrator": calibrator,
            "n_cal": n_cal,
            "set_size_cxcy": set_size_cxcy_slice(score_name, calibrator),
        }

    result: Dict[str, Any] = {
        "score_name": score_name,
        "alpha": float(alpha),
        "mondrian_field": mondrian_field,
        "strata": strata,
        "refused": refused,
    }
    return _provenance.stamp_result(result, script_path=__file__, seeds=None)


def g1_coverage(
    cert: Dict[str, Any], eval_matched: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Empirical G1 coverage on a (frozen) evaluation split, per stratum + overall.

    Uses the calibrators in ``cert["strata"]`` (from :func:`g1_calibrate`); evaluation
    rows whose stratum was REFUSED at calibration, or whose stratum was never seen at
    calibration, are excluded and counted (``n_out_of_support``) -- never silently
    scored against a neighboring stratum's threshold.
    """
    score_name = cert["score_name"]
    mondrian_field = cert["mondrian_field"]
    score = _resolve_score(score_name)

    per_stratum: Dict[Any, Dict[str, Any]] = {}
    n_out_of_support = 0
    all_covered: List[bool] = []

    by_stratum: Dict[Any, List[Dict[str, Any]]] = {}
    for m in eval_matched:
        st = m[mondrian_field] if mondrian_field is not None else None
        by_stratum.setdefault(st, []).append(m)

    for st, rows in by_stratum.items():
        if st not in cert["strata"]:
            n_out_of_support += len(rows)
            continue
        calibrator = cert["strata"][st]["calibrator"]
        covered = []
        if isinstance(score, ScaledBonferroniBoxScore):
            eval_scores = _pred_scores_or_raise(rows, "g1_coverage")
            for m, s in zip(rows, eval_scores):
                covered.append(bool(score.covers(calibrator, m["pred_obb"], m["gt_obb"], s)))
        else:
            for m in rows:
                covered.append(bool(score.covers(calibrator, m["pred_obb"], m["gt_obb"])))
        per_stratum[st] = {"coverage": float(np.mean(covered)), "n": len(covered)}
        all_covered.extend(covered)

    overall = float(np.mean(all_covered)) if all_covered else float("nan")
    return {
        "score_name": score_name,
        "overall_coverage": overall,
        "n": len(all_covered),
        "per_stratum": per_stratum,
        "n_out_of_support": n_out_of_support,
    }


# ---------------------------------------------------------------------------
# G2
# ---------------------------------------------------------------------------


def image_risk_matrix(
    scene_gt_confidences: Sequence[Sequence[Optional[float]]], lambda_grid: Sequence[float]
) -> np.ndarray:
    """Per-scene miss-rate matrix, shape ``(n_scenes, K)``.

    Parameters
    ----------
    scene_gt_confidences:
        One entry per scene: a list of matched-detection confidences, one per GT box
        in that scene (``None`` for a GT never matched by any detection at any
        confidence, i.e. a hard miss regardless of ``lambda``). A scene with zero GT
        for the class/stratum being certified should be EXCLUDED before calling this
        (it contributes no information and its risk is undefined); this function
        raises :class:`CertifyError` on any empty-GT scene rather than silently
        producing a NaN row.
    lambda_grid:
        Candidate confidence thresholds (any order).

    Returns
    -------
    ndarray
        ``risk[i, k] = (# GT in scene i with matched_confidence is None or < grid[k])
        / (# GT in scene i)``, in ``[0, 1]``.
    """
    grid = np.asarray(sorted(float(v) for v in lambda_grid), dtype=float)
    n_scenes = len(scene_gt_confidences)
    if n_scenes == 0:
        raise CertifyError("image_risk_matrix: scene_gt_confidences must be non-empty")
    risk = np.zeros((n_scenes, grid.size), dtype=float)
    for i, confs in enumerate(scene_gt_confidences):
        if len(confs) == 0:
            raise CertifyError(
                f"image_risk_matrix: scene index {i} has zero GT boxes -- exclude "
                "empty-GT scenes upstream (their risk is undefined, not zero)"
            )
        confs_arr = np.array([c if c is not None else -np.inf for c in confs], dtype=float)
        n_gt = confs_arr.size
        for k, lam in enumerate(grid):
            risk[i, k] = float(np.sum(confs_arr < lam)) / n_gt
    return risk


def g2_certify_fnr(
    scene_gt_confidences: Sequence[Sequence[Optional[float]]],
    beta: float = 0.20,
    delta: float = 0.05,
    lambda_grid: Optional[Sequence[float]] = None,
    n_grid: int = 50,
    min_accept_frac: float = 0.05,
    procedure: str = "bonferroni",
    p_value: str = "eb",
    bentkus_factor: float = 1.75,
) -> Dict[str, Any]:
    """G2: certified per-image FNR via LTT-HB, gated by the a-priori power floor
    (design §2.4). See module docstring; this is the single-stratum (e.g. one class,
    or the pooled-marginal) certificate -- :func:`g2_certify_fnr_mondrian` wraps this
    per class with the K5 pooled-fallback remedy.
    """
    n_img = len(scene_gt_confidences)
    if n_img == 0:
        raise CertifyError("g2_certify_fnr: scene_gt_confidences must be non-empty")

    all_confs = [c for confs in scene_gt_confidences for c in confs if c is not None]
    if lambda_grid is None:
        if not all_confs:
            raise CertifyError(
                "g2_certify_fnr: no matched detections anywhere -- cannot build a "
                "confidence lambda grid (every GT is an unconditional miss; G2 refuses)"
            )
        lambda_grid = _ltt.build_lambda_grid(np.array(all_confs), n_grid=n_grid, min_accept_frac=min_accept_frac)

    risk_matrix = image_risk_matrix(scene_gt_confidences, lambda_grid)
    grid_sorted = np.asarray(sorted(float(v) for v in lambda_grid))
    r_hat = float(np.mean(risk_matrix[:, 0]))  # most permissive lambda (retain everything)

    power_floor = _ltt.power_floor_n_img(beta, delta, grid_sorted.size, r_hat, bentkus_factor=bentkus_factor)
    powered = n_img >= power_floor["bentkus_floor"]

    if not powered:
        return {
            "certified": False,
            "refused": True,
            "reason": (
                f"n_img={n_img} below the LTT-HB Bentkus power floor "
                f"{power_floor['bentkus_floor']:.1f} at beta={beta}, delta={delta}, "
                f"r_hat={r_hat:.4f} (design §2.4/§5 K5 -- refuses rather than reports "
                "an uncertifiable number)"
            ),
            "n_img": n_img,
            "power_floor": power_floor,
            "beta": float(beta),
            "delta": float(delta),
            "lambda_star": None,
        }

    result = _ltt.ltt_certify_matrix(
        risk_matrix, grid_sorted, beta=beta, delta=delta, procedure=procedure, p_value=p_value
    )
    result["power_floor"] = power_floor
    result["refused"] = not result["certified"]
    return result


def g2_certify_fnr_mondrian(
    scene_gt_confidences_by_class: Dict[Any, Sequence[Sequence[Optional[float]]]],
    beta: float = 0.20,
    delta: float = 0.05,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Per-class G2, with the K5 pooled-marginal fallback for classes below the power
    floor (design §2.4/§5): the pooled FNR (all scenes across all classes, computed
    once) is always reported alongside so a refused class still has an honest fallback
    number, explicitly flagged as coverage-debt (design §2.4).
    """
    if not scene_gt_confidences_by_class:
        raise CertifyError("g2_certify_fnr_mondrian: scene_gt_confidences_by_class must be non-empty")

    pooled_scenes: List[Sequence[Optional[float]]] = []
    for scenes in scene_gt_confidences_by_class.values():
        pooled_scenes.extend(scenes)
    pooled = g2_certify_fnr(pooled_scenes, beta=beta, delta=delta, **kwargs)

    per_class: Dict[Any, Dict[str, Any]] = {}
    n_certified = 0
    for cls, scenes in scene_gt_confidences_by_class.items():
        try:
            res = g2_certify_fnr(scenes, beta=beta, delta=delta, **kwargs)
        except CertifyError as e:
            res = {"certified": False, "refused": True, "reason": str(e), "n_img": len(scenes)}
        per_class[cls] = res
        if res.get("certified"):
            n_certified += 1

    return {
        "beta": float(beta),
        "delta": float(delta),
        "per_class": per_class,
        "n_classes": len(scene_gt_confidences_by_class),
        "n_classes_certified": n_certified,
        "pooled_marginal": pooled,
    }

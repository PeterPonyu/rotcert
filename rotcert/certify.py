"""Matched-TP conformal calibration and diagnostic scene-average miss-risk tests.

G1 requires exchangeable calibration and future matched-TP scores within the
chosen stratum. Scene splitting does not establish object-level exchangeability.
False positives lie outside this conditional population.

G2 preserves numerical threshold tests but makes no validated risk-control claim.
The default grid is estimated from the tested records. A supplied grid alone does
not verify its independence or scene sampling. All G2 outputs are diagnostic;
``certified`` is always false. Mondrian tests allocate delta over the declared
class family and never pool repeated class-by-scene rows as independent scenes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from rotcert._vendor.relmetrics import provenance as _provenance

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
    """Return a diagnostic threshold pass, subject to the historical screen.

    ``lambda_star`` is a numerical readout, not a certified deployment threshold.
    ``refused`` refers to a probability certificate and is always true in this
    diagnostic interface. Frozen historical results are not regenerated here.
    """
    n_img = len(scene_gt_confidences)
    if n_img == 0:
        raise CertifyError("g2_certify_fnr: scene_gt_confidences must be non-empty")

    grid_source = "same_sample_quantiles" if lambda_grid is None else "caller_supplied_unverified"
    all_confs = [c for confs in scene_gt_confidences for c in confs if c is not None]
    if lambda_grid is None:
        if not all_confs:
            raise CertifyError(
                "g2_certify_fnr: no matched detections anywhere -- cannot build a "
                "confidence lambda grid (every GT is an unconditional miss; G2 refuses)"
            )
        lambda_grid = _ltt.build_lambda_grid(np.array(all_confs), n_grid=n_grid, min_accept_frac=min_accept_frac)

    grid_sorted = np.asarray(sorted(float(v) for v in lambda_grid))
    if grid_sorted.size == 0 or not np.all(np.isfinite(grid_sorted)):
        raise CertifyError("lambda_grid must be non-empty and finite")
    risk_matrix = image_risk_matrix(scene_gt_confidences, grid_sorted)
    r_hat = float(np.mean(risk_matrix[:, 0]))  # most permissive lambda (retain everything)

    power_floor = _ltt.power_floor_n_img(beta, delta, grid_sorted.size, r_hat, bentkus_factor=bentkus_factor)
    powered = n_img >= power_floor["bentkus_floor"]

    if not powered:
        return {
            "certified": False,
            "diagnostic_pass": False,
            "validity": "diagnostic_only",
            "grid_source": grid_source,
            "refused": True,
            "reason": (
                f"n_img={n_img} below the historical fitted power floor "
                f"{power_floor['bentkus_floor']:.1f} at beta={beta}, delta={delta}, "
                f"r_hat={r_hat:.4f}; this screen is not a sufficient sample budget"
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
    result["refused"] = True
    result["grid_source"] = grid_source
    return result


def g2_certify_fnr_mondrian(
    scene_gt_confidences_by_class: Dict[Any, Sequence[Sequence[Optional[float]]]],
    beta: float = 0.20,
    delta: float = 0.05,
    class_roster: Optional[Sequence[Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Diagnostic class-family tests with equal delta allocation.

    Supply the predetermined ``class_roster`` to retain absent classes in K.
    Without it, the family consists only of input keys and is explicitly marked
    as such. Neither mode verifies a prospective protocol. No pooled fallback
    is computed: concatenated class-by-scene rows can repeat the same scene.
    """
    if not scene_gt_confidences_by_class:
        raise CertifyError("g2_certify_fnr_mondrian: scene records must be non-empty")
    roster = list(scene_gt_confidences_by_class if class_roster is None else class_roster)
    if not roster or len(set(roster)) != len(roster):
        raise CertifyError("class_roster must be non-empty and unique")
    if set(scene_gt_confidences_by_class) - set(roster):
        raise CertifyError("class_roster must include every input class")
    if not 0.0 < delta < 1.0:
        raise CertifyError("delta must be in (0, 1)")
    delta_per_class = delta / len(roster)
    per_class: Dict[Any, Dict[str, Any]] = {}
    for cls in roster:
        scenes = scene_gt_confidences_by_class.get(cls, [])
        if not scenes:
            res = {"certified": False, "diagnostic_pass": False, "refused": True,
                   "validity": "diagnostic_only", "reason": "no_class_data",
                   "n_img": 0, "delta": delta_per_class, "lambda_star": None}
        else:
            try:
                res = g2_certify_fnr(scenes, beta=beta, delta=delta_per_class, **kwargs)
            except CertifyError as e:
                res = {"certified": False, "diagnostic_pass": False, "refused": True,
                       "validity": "diagnostic_only", "reason": str(e),
                       "n_img": len(scenes), "delta": delta_per_class, "lambda_star": None}
        per_class[cls] = res
    return {
        "beta": float(beta), "delta": float(delta), "delta_per_class": delta_per_class,
        "class_roster": roster,
        "family_source": "input_keys_only" if class_roster is None else "caller_declared",
        "per_class": per_class, "n_classes": len(roster),
        "n_classes_diagnostic_pass": sum(bool(r["diagnostic_pass"]) for r in per_class.values()),
        "n_classes_certified": 0, "certified": False, "validity": "diagnostic_only",
        "pooled_marginal": None,
        "pooled_marginal_status": "not_computed_class_scene_rows_are_not_independent",
    }

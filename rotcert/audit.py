"""The validity audit: scene-clustered bootstrap coverage CIs (V1/V2), the confirmatory
Holm-8 head-to-head (C2), and the K1 premise-death check (design §4.5, §5).

Scene-clustered, never box-level (design §4.2 M2 -- BINDING)
------------------------------------------------------------------
Every coverage CI in this module clusters its bootstrap at the SCENE level (block =
``scene_id``), per the design's explicit ban on box-level Clopper-Pearson ("anti-
conservatively narrow and is NOT used"): boxes within one scene are correlated
(co-located objects, shared imaging conditions), and treating them as independent
understates variance. :func:`coverage_ci` takes ``rotcert.bootstrap`` blocks by
construction -- there is no box-level code path in this module to accidentally reach
for.

Theta-stratum classification (boundary / square / interior, design §4.5)
------------------------------------------------------------------------------
:func:`classify_theta_stratum` implements the two named strata: "boundary" (within
``boundary_deg`` of the le90 seam, +-90 degrees) and "square" (short/long ratio >=
``square_ratio``); everything else is "interior". The square-stratum full-arc
angular-vacuity rule (design §4.5: a valid angle set there must cover ~the full arc,
so square-stratum results are a VALIDITY check, never an efficiency comparison) is a
CALLER contract, not enforced by any function here. NOTE (corrected 2026-07-12): there
is NO ``exclude_square`` parameter on :func:`holm8_confirmatory` or
:func:`set_size_contrast` -- the earlier claim that square exclusion was "checked, not
merely documented" via such a parameter was false. It is documentation only, and the
frozen holm8_run.py (2026-07-12) did NOT exclude square-stratum detections: it pooled
all matched true positives (mondrian_field=None). The impact on the confirmatory
(cx,cy)-slice-area contrast is limited because that metric is the center-localization
slice, not the angular set whose square-stratum vacuity the rule guards; a
square-excluded re-run is a documented follow-up.

The confirmatory Holm-8 (design §1 C2, §4.5) -- family size never hardcoded
--------------------------------------------------------------------------------
Mirrors ``asr-gate/asr_gate/audit.py``'s rule: the Holm family size is ALWAYS
``len(cells)``, the roster actually assembled by the caller -- never a literal ``8``
baked into this module. The design's own headline number (2 baselines x 2 detectors x
2 datasets = 8) is simply what that roster evaluates to once both detectors and both
datasets have real result cells; running with a partial roster (e.g. one detector
during a pilot) changes the family size automatically and honestly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from relmetrics import bootstrap as _bootstrap
from relmetrics import multiplicity as _multiplicity
from relmetrics import provenance as _provenance

from rotcert.gwd import canonicalize_le90

__all__ = [
    "classify_theta_stratum",
    "coverage_ci",
    "stratum_coverage_table",
    "set_size_contrast",
    "holm8_confirmatory",
    "k1_premise_death",
]


def classify_theta_stratum(
    w: float, h: float, theta: float, boundary_deg: float = 5.0, square_ratio: float = 0.9
) -> str:
    """``"boundary"`` (within ``boundary_deg`` of +-90deg), ``"square"`` (h/w >=
    ``square_ratio``), else ``"interior"``. Square is checked FIRST (a near-square box
    near the seam is reported as square -- its orientation is the more fundamental
    degeneracy, per the design's square-stratum angular-vacuity framing, §4.5)."""
    w_c, h_c, t_c = canonicalize_le90(np.array(w), np.array(h), np.array(theta))
    w_c, h_c, t_c = float(w_c), float(h_c), float(t_c)
    if w_c <= 0:
        raise ValueError("classify_theta_stratum: w must be positive")
    if h_c / w_c >= square_ratio:
        return "square"
    boundary_rad = np.deg2rad(boundary_deg)
    if (np.pi / 2.0 - abs(t_c)) <= boundary_rad:
        return "boundary"
    return "interior"


def coverage_ci(
    covered: Sequence[bool],
    scene_ids: Sequence[str],
    ci_level: float = 0.95,
    n_boot: int = 1000,
    seed: int = 0,
    method: str = "percentile",
) -> Dict[str, Any]:
    """Scene-clustered bootstrap CI on mean coverage (design §4.2/§4.5 M2)."""
    covered_arr = np.asarray(covered, dtype=float)
    if covered_arr.size == 0:
        raise ValueError("coverage_ci: covered must be non-empty")
    scene_arr = np.asarray(scene_ids)
    if scene_arr.shape != covered_arr.shape:
        raise ValueError("coverage_ci: scene_ids must match covered in length")
    boot = _bootstrap.blocked_bootstrap(
        lambda c: float(np.mean(c)),
        covered_arr,
        block_ids=scene_arr,
        n_boot=n_boot,
        seeds=[seed],
        ci_level=ci_level,
        method=method,
    )
    return {
        "point": boot["point"],
        "ci": boot["ci"],
        "n": int(covered_arr.size),
        "n_scenes": boot["n_blocks"],
    }


def stratum_coverage_table(
    rows: Sequence[Dict[str, Any]],
    covered_field: str = "covered",
    stratum_field: str = "theta_stratum",
    scene_field: str = "scene_id",
    **ci_kwargs: Any,
) -> Dict[str, Dict[str, Any]]:
    """Per-stratum scene-clustered coverage CI, e.g. for V2's
    {boundary, square, interior} x construction table (design §4.5)."""
    if not rows:
        raise ValueError("stratum_coverage_table: rows must be non-empty")
    by_stratum: Dict[Any, List[Dict[str, Any]]] = {}
    for r in rows:
        by_stratum.setdefault(r[stratum_field], []).append(r)
    out: Dict[str, Dict[str, Any]] = {}
    for st, subset in by_stratum.items():
        covered = [r[covered_field] for r in subset]
        scenes = [r[scene_field] for r in subset]
        out[st] = coverage_ci(covered, scenes, **ci_kwargs)
    return out


def set_size_contrast(
    set_size_primary: Sequence[float],
    set_size_baseline: Sequence[float],
    class_labels: Sequence[Any],
    n_perm: int = 2000,
    n_boot: int = 1000,
    seed: int = 0,
    ci_level: float = 0.95,
) -> Dict[str, Any]:
    """Paired per-class set-size contrast: primary (GWD) vs one baseline (B1 or B2),
    at the IDENTICAL NOMINAL calibration level (design §4.4 -- callers calibrate both
    constructions at the same nominal alpha on the dedicated matching split, then pass
    the frozen eval split's set sizes here). NOTE (corrected 2026-07-12): this equalizes
    the NOMINAL level, NOT realized marginal coverage -- in the frozen Holm-8 run the
    baselines over-cover (realized ~0.935-0.950) relative to GWD (~0.897-0.907), so the
    two constructions are NOT at matched realized coverage. Part of any measured size
    gap is therefore the price of baseline over-coverage; a coverage-matched ablation
    (retune each construction to a common realized coverage) is the clean follow-up.

    DEGENERACY WARNING: if the caller feeds this function set sizes that are constant
    within each construction (e.g. pooled split-conformal, where set_size_cxcy_slice
    returns one scalar per cell -- see holm8_run.py), every per-pair log-ratio is the
    SAME constant. The class-blocked bootstrap CI then collapses to zero width and the
    sign-flip permutation p is DETERMINISTIC at the 1/(n_perm+1) floor for any negative
    effect regardless of magnitude: effective n is 1, not len(class_labels). The p-value
    carries no evidence strength in that regime -- only the effect SIZE (ratio) does.

    Statistic: mean log-ratio ``log(size_primary / size_baseline)`` across classes
    (blocks). Point estimate + CI via a CLASS-BLOCKED bootstrap (design §4.5's
    "class-blocked bootstrap for the CI on the pooled set-size-ratio effect"); p-value
    via a class-blocked sign-flip permutation test (design §4.5's "paired ... permutation
    p-value" -- implemented here as a label-swap/sign-flip null on the log-ratio, the
    natural paired-permutation construction for a per-class ratio statistic; the
    design's literal "matched-abstention" terminology is a selective-risk-deferral
    concept from ``relmetrics.nulls`` that does not apply to a set-size ratio, so this
    is a documented, deliberate adaptation, not `relmetrics.nulls.matched_abstention_
    null` reused verbatim). One-sided: ``H1: primary < baseline`` (GWD strictly
    smaller, per C2).

    Returns
    -------
    dict
        ``point_log_ratio``, ``ci_log_ratio`` (class-blocked bootstrap), ``p_value``
        (one-sided, permutation), ``n_perm``, ``n_classes``, ``ratio`` (``exp(point_
        log_ratio)``, the human-readable "GWD is X times the baseline's size").
    """
    a = np.asarray(set_size_primary, dtype=float)
    b = np.asarray(set_size_baseline, dtype=float)
    labels = np.asarray(class_labels)
    if a.shape != b.shape or a.shape != labels.shape:
        raise ValueError("set_size_contrast: set_size_primary/baseline/class_labels must match in length")
    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("set_size_contrast: set sizes must be strictly positive")
    log_ratio = np.log(a) - np.log(b)

    def _stat(x: np.ndarray) -> float:
        return float(np.mean(x))

    boot = _bootstrap.blocked_bootstrap(
        _stat, log_ratio, block_ids=labels, n_boot=n_boot, seeds=[seed], ci_level=ci_level
    )
    observed = _stat(log_ratio)

    rng = np.random.default_rng(seed)
    n = log_ratio.size
    perm_stats = np.empty(n_perm)
    for i in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        perm_stats[i] = _stat(log_ratio * signs)
    p_value = float((np.sum(perm_stats <= observed) + 1) / (n_perm + 1))

    return {
        "point_log_ratio": boot["point"],
        "ci_log_ratio": boot["ci"],
        "ratio": float(np.exp(boot["point"])),
        "p_value": p_value,
        "n_perm": int(n_perm),
        "n_classes": int(n),
    }


def holm8_confirmatory(cells: Sequence[Dict[str, Any]], alpha: float = 0.05) -> Dict[str, Any]:
    """Holm-Bonferroni correction across the confirmatory-8 roster (design §1 C2,
    §4.5). ``cells``: one dict per (baseline, detector, dataset) cell, each carrying
    ``"p_value"`` (from :func:`set_size_contrast`) plus whatever identifying keys the
    caller wants echoed (``baseline``, ``detector``, ``dataset`` conventionally).
    Family size = ``len(cells)``, never hardcoded (module docstring)."""
    if not cells:
        raise ValueError("holm8_confirmatory: cells must be non-empty")
    pvals = [c["p_value"] for c in cells]
    holm = _multiplicity.holm_bonferroni(pvals, alpha=alpha)
    results = []
    for c, p_adj, rej in zip(cells, holm["adjusted_p"], holm["reject"]):
        r = dict(c)
        r["p_holm"] = float(p_adj)
        r["reject_holm"] = bool(rej)
        results.append(r)
    out = {"family_size": len(cells), "alpha": float(alpha), "results": results}
    return _provenance.stamp_result(out, script_path=__file__, seeds=None)


def k1_premise_death(
    boundary_square_gap_pp: float,
    holm_results: Sequence[Dict[str, Any]],
    boundary_gap_threshold_pp: float = 2.0,
    b1_gap_floor_pp: float = 5.0,
    b1_gap_pp: Optional[float] = None,
    advantage_threshold: float = 0.05,
    min_significant_cells: int = 3,
) -> Dict[str, Any]:
    """K1 premise-death check (design §5, verbatim): naive B1 does NOT under-cover the
    boundary/square strata (gap < ``boundary_gap_threshold_pp``) AND does NOT lose the
    set-size head-to-head (GWD advantage < ``advantage_threshold`` and Holm-significant
    on FEWER than ``min_significant_cells`` of the confirmatory-8) => the angle-aware
    premise is false.

    Parameters
    ----------
    boundary_square_gap_pp:
        Naive-coord (B1) minus GWD coverage gap, in percentage points, on the
        boundary/square strata (the LARGER of the two if reporting both; caller's
        choice of aggregation, documented at the call site).
    holm_results:
        The ``results`` list from :func:`holm8_confirmatory` (each with ``"ratio"``
        from the paired contrast and ``"reject_holm"``).
    b1_gap_pp:
        Optional explicit B1 boundary/square coverage gap if different from
        ``boundary_square_gap_pp`` (kept separate so V2's "GWD gap < 2pp AND B1 gap >=
        5pp" two-sided check, design §4.5, can be scored precisely); defaults to
        ``boundary_square_gap_pp`` if omitted.

    Returns
    -------
    dict
        ``premise_death`` (bool), plus every intermediate quantity used to decide it,
        for the report to show its work.
    """
    if not holm_results:
        raise ValueError("k1_premise_death: holm_results must be non-empty")
    b1_gap = boundary_square_gap_pp if b1_gap_pp is None else b1_gap_pp

    undercover_confirmed = b1_gap >= b1_gap_floor_pp
    gwd_holds = boundary_square_gap_pp < boundary_gap_threshold_pp

    n_significant = sum(1 for r in holm_results if r.get("reject_holm"))
    max_advantage = max(1.0 - r["ratio"] for r in holm_results if "ratio" in r) if any(
        "ratio" in r for r in holm_results
    ) else 0.0
    inflation_confirmed = (max_advantage >= advantage_threshold) and (n_significant >= min_significant_cells)

    premise_death = (not undercover_confirmed) and (not inflation_confirmed) and gwd_holds

    return {
        "premise_death": bool(premise_death),
        "b1_boundary_square_gap_pp": float(b1_gap),
        "gwd_boundary_square_gap_pp": float(boundary_square_gap_pp),
        "undercover_confirmed": bool(undercover_confirmed),
        "gwd_holds": bool(gwd_holds),
        "inflation_confirmed": bool(inflation_confirmed),
        "n_holm_significant": int(n_significant),
        "min_significant_cells": int(min_significant_cells),
        "max_gwd_advantage": float(max_advantage),
        "advantage_threshold": float(advantage_threshold),
    }

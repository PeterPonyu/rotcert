#!/usr/bin/env python3
"""Coverage-matched efficiency ablation (Part A -- repairs the M3 / efficiency support gap).

Motivation
----------
The Holm-8 head-to-head and its R=20 exploratory companion contrast GWD vs. the naive-coord
/ hull Bonferroni baselines at IDENTICAL NOMINAL alpha=0.10. The union-bound baselines
OVER-COVER at that nominal level (realized eval coverage ~0.915-0.948) while GWD sits near
nominal (~0.883 DOTA, ~0.899 DIOR), so part of GWD's apparent size advantage is simply the
price of baseline over-coverage, not a sharper region. This script removes that confound.

Protocol (identical to holm8_r20_exploratory.py, plus coverage-matching)
------------------------------------------------------------------------
Same 4 (detector,dataset) cells, same 2 baselines, same 40/20/40 scene split with
split-seed = repeat index (rotcert.splits.repeated_scene_splits). Per split: calibrate on
the MATCHING scenes, evaluate on the disjoint EVAL scenes (pooled split-conformal,
mondrian_field=None -- the frozen contrast's calibration, NOT the per-class Mondrian shipped
G1 certificate). For each cell/split:

  1. GWD is calibrated at nominal alpha=0.10 -> realized eval coverage c_gwd, region area
     A_gwd = pi*q_hat^2 (its shipped operating point; left untouched).
  2. Each baseline is RE-calibrated at an adjusted level alpha' chosen (by bisection on the
     calibration-set quantiles, monotone) so that the baseline's REALIZED eval joint coverage
     matches c_gwd. Region area A_base(alpha') = 4*q_cx*q_cy at that alpha'.
  3. Coverage-matched ratio = A_gwd / A_base(alpha')  (<1 => GWD smaller at matched coverage).

Both GWD-ball coverage and the baseline joint-box coverage are joint 5-/4-DOF statements
("true box lies in the region"), so matching realized coverage is apples-to-apples; the
(cx,cy)-slice area is the same efficiency metric used throughout (set_size_cxcy_slice).

This is a POST-HOC, coverage-FAIR ablation: alpha' is selected against realized eval coverage,
so it is an efficiency-at-a-fixed-operating-point comparison, NOT a coverage guarantee for the
baseline and NOT a preregistered/confirmatory test. The baseline region SHAPE is still
calibrated out-of-sample on the calibration scenes; only the scalar level is retuned to a
common realized-coverage target.

Validation gate: at alpha'=0.10 this script reproduces the holm8_r20_exploratory baseline
coverages and size ratios (printed as a cross-check).
"""
import json
import math
import sys
import time
from pathlib import Path

ROOT = _portal_repo_root()
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from rotcert import io as _io, certify as _certify, splits as _splits  # noqa: E402
from rotcert.scores import SCORES, set_size_cxcy_slice  # noqa: E402

CELLS = [
    ("rtmdet", "dota", ROOT / "pilot_results_2026-07-10/pulled/root/autodl-tmp/rotcert_results/pilot/matched.jsonl"),
    ("orcnn", "dota", ROOT / "orcnn_dota_cert_2026-07-11/matched.jsonl"),
    ("orcnn", "dior", ROOT / "dior_cert_results_2026-07-11/orcnn_fixedgt/matched.jsonl"),
    ("rtmdet", "dior", ROOT / "dior_cert_results_2026-07-11/rtmdet/matched.jsonl"),
]
BASELINES = ["naive-coord", "hull"]
ALPHA_GWD = 0.10
N_REPEATS = 20
OUT = ROOT / "coverage_matched_2026-07-13"


def load_tp(path):
    rows = _io.load_jsonl(path)
    return [{"pred_obb": r["pred_obb"], "gt_obb": r["gt_obb"], "class": r["class"],
             "scene_id": r["scene_id"]} for r in rows if r.get("match_type") == "tp"]


def precompute_residuals(matched):
    """Each matched pair's GWD residual and per-coordinate baseline residuals are
    INDEPENDENT of the scene split, so compute them ONCE per cell and index by split
    membership below (identical results to per-split recomputation, far cheaper).
    Returns (gwd_res[n], {baseline: {coord: res[n]}}, scene_code[n], scene_list)."""
    gwd = SCORES["gwd"]
    n = len(matched)
    gwd_res = np.empty(n, float)
    base_res = {b: {c: np.empty(n, float) for c in SCORES[b].coord_names} for b in BASELINES}
    scenes = sorted({m["scene_id"] for m in matched})
    scene_to_code = {s: i for i, s in enumerate(scenes)}
    scene_code = np.empty(n, dtype=np.int64)
    for i, m in enumerate(matched):
        p = np.asarray(m["pred_obb"], float)
        g = np.asarray(m["gt_obb"], float)
        gwd_res[i] = gwd.residual(p, g)
        for b in BASELINES:
            r = SCORES[b].residuals(p, g)
            for c in SCORES[b].coord_names:
                base_res[b][c][i] = r[c]
        scene_code[i] = scene_to_code[m["scene_id"]]
    return gwd_res, base_res, scene_code, scenes


def _threshold(sorted_arr, n_cal, alpha_k):
    """relmetrics.conformal.SplitConformal deterministic threshold, replicated exactly:
    the ceil((n+1)(1-alpha))-th smallest calibration residual (+inf if the rank exceeds n)."""
    rank = int(math.ceil((n_cal + 1) * (1.0 - alpha_k)))
    if rank > n_cal:
        return math.inf
    if rank < 1:
        rank = 1
    return float(sorted_arr[rank - 1])


def baseline_residual_arrays(score, rows):
    """Per-coordinate residual arrays for a Bonferroni-box score over `rows`.
    Returns (dict coord->np.ndarray, n)."""
    coords = score.coord_names
    acc = {c: [] for c in coords}
    for m in rows:
        res = score.residuals(np.asarray(m["pred_obb"], float), np.asarray(m["gt_obb"], float))
        for c in coords:
            acc[c].append(res[c])
    return {c: np.asarray(acc[c], float) for c in coords}, len(rows)


def baseline_coverage_and_area(score, cal_sorted, n_cal, eval_res, alpha_prime):
    """Realized eval JOINT coverage and (cx,cy)-slice area of the Bonferroni box at level
    alpha_prime (per-coord Bonferroni level alpha_prime/K), calibrated on cal, eval on eval."""
    coords = score.coord_names
    K = len(coords)
    alpha_k = alpha_prime / K
    q = {c: _threshold(cal_sorted[c], n_cal, alpha_k) for c in coords}
    covered = np.ones(len(next(iter(eval_res.values()))), dtype=bool)
    for c in coords:
        covered &= (eval_res[c] <= q[c])
    cov = float(covered.mean())
    cx_name, cy_name = coords[0], coords[1]  # (cx,cy) for naive-coord and hull alike
    area = float(4.0 * q[cx_name] * q[cy_name])
    return cov, area, {c: (None if math.isinf(q[c]) else q[c]) for c in coords}


def match_alpha(score, cal_sorted, n_cal, eval_res, target_cov, n_iter=60):
    """Bisection for alpha' whose realized eval joint coverage matches target_cov (coverage is
    monotone non-increasing in alpha'). Returns (alpha', cov, area, q)."""
    lo, hi = 1e-4, 0.999  # cov(lo) high, cov(hi) low
    lo_cov, _, _ = baseline_coverage_and_area(score, cal_sorted, n_cal, eval_res, lo)
    hi_cov, _, _ = baseline_coverage_and_area(score, cal_sorted, n_cal, eval_res, hi)
    # If target is outside the achievable range, clamp to the nearest endpoint.
    if target_cov >= lo_cov:
        a = lo
    elif target_cov <= hi_cov:
        a = hi
    else:
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            cov, _, _ = baseline_coverage_and_area(score, cal_sorted, n_cal, eval_res, mid)
            if cov > target_cov:      # coverage too high -> increase alpha'
                lo = mid
            else:
                hi = mid
        # Pick the endpoint whose achieved coverage is closest to target.
        lo_c, _, _ = baseline_coverage_and_area(score, cal_sorted, n_cal, eval_res, lo)
        hi_c, _, _ = baseline_coverage_and_area(score, cal_sorted, n_cal, eval_res, hi)
        a = lo if abs(lo_c - target_cov) <= abs(hi_c - target_cov) else hi
    cov, area, q = baseline_coverage_and_area(score, cal_sorted, n_cal, eval_res, a)
    return a, cov, area, q


def main():
    t0 = time.time()
    gwd = SCORES["gwd"]
    cells_out = []
    for det, ds, path in CELLS:
        matched = load_tp(path)
        # Residuals are split-independent -> compute once, index by split membership.
        gwd_res, base_res, scene_code, scenes = precompute_residuals(matched)
        reps = _splits.repeated_scene_splits(scenes, n_repeats=N_REPEATS, cal_frac=0.4, match_frac=0.2)
        code_of = {s: i for i, s in enumerate(scenes)}
        print(f"\n=== {det}/{ds}: {len(matched)} TP over {len(scenes)} scenes ({time.time()-t0:.0f}s) ===", flush=True)
        # accumulators per baseline
        acc = {b: {"ratio_matched": [], "ratio_nominal": [], "alpha_prime": [],
                   "cov_base_matched": [], "cov_base_nominal": [], "area_base_matched": [],
                   "area_base_nominal": []} for b in BASELINES}
        gwd_cov_list, gwd_area_list = [], []
        for r, rep in enumerate(reps):
            match_codes = np.array([code_of[s] for s in rep["matching"]], dtype=np.int64)
            eval_codes = np.array([code_of[s] for s in rep["eval"]], dtype=np.int64)
            match_mask = np.isin(scene_code, match_codes)
            eval_mask = np.isin(scene_code, eval_codes)
            # GWD at nominal alpha (threshold replicates relmetrics SplitConformal exactly)
            gwd_cal = np.sort(gwd_res[match_mask])
            n_cal_gwd = gwd_cal.size
            q_gwd = _threshold(gwd_cal, n_cal_gwd, ALPHA_GWD)
            if r == 0:  # one-time validation that _threshold == g1_calibrate's SplitConformal
                match_rows = [m for m, keep in zip(matched, match_mask) if keep]
                calib = _certify.g1_calibrate(match_rows, "gwd", alpha=ALPHA_GWD,
                                              mondrian_field=None)["strata"][None]["calibrator"]
                assert abs(float(calib.q_hat) - q_gwd) < 1e-9, (det, ds, calib.q_hat, q_gwd)
            A_gwd = float(math.pi * q_gwd * q_gwd)
            c_gwd = float(np.mean(gwd_res[eval_mask] <= q_gwd))
            gwd_cov_list.append(c_gwd)
            gwd_area_list.append(A_gwd)
            for base in BASELINES:
                score = SCORES[base]
                cal_sorted = {c: np.sort(base_res[base][c][match_mask]) for c in score.coord_names}
                n_cal = int(match_mask.sum())
                eval_arr = {c: base_res[base][c][eval_mask] for c in score.coord_names}
                # nominal alpha'=0.10 (validation / decomposition)
                cov_nom, area_nom, _ = baseline_coverage_and_area(score, cal_sorted, n_cal, eval_arr, ALPHA_GWD)
                # coverage-matched alpha'
                a_star, cov_m, area_m, _ = match_alpha(score, cal_sorted, n_cal, eval_arr, c_gwd)
                acc[base]["ratio_nominal"].append(A_gwd / area_nom)
                acc[base]["ratio_matched"].append(A_gwd / area_m)
                acc[base]["alpha_prime"].append(a_star)
                acc[base]["cov_base_matched"].append(cov_m)
                acc[base]["cov_base_nominal"].append(cov_nom)
                acc[base]["area_base_matched"].append(area_m)
                acc[base]["area_base_nominal"].append(area_nom)
            if r == 0:
                print(f"  [split0] c_gwd={c_gwd:.4f} A_gwd={A_gwd:.2f}", flush=True)
                for base in BASELINES:
                    print(f"    {base}: nominal(a=.10) cov={acc[base]['cov_base_nominal'][0]:.4f} "
                          f"ratio={acc[base]['ratio_nominal'][0]:.4f}  ||  "
                          f"matched a'={acc[base]['alpha_prime'][0]:.4f} cov={acc[base]['cov_base_matched'][0]:.4f} "
                          f"ratio={acc[base]['ratio_matched'][0]:.4f}", flush=True)
        gwd_cov = np.array(gwd_cov_list)
        for base in BASELINES:
            d = acc[base]
            rm = np.array(d["ratio_matched"])
            rn = np.array(d["ratio_nominal"])
            cell = {
                "detector": det, "dataset": ds, "baseline": base, "n_splits": N_REPEATS,
                "gwd_alpha": ALPHA_GWD,
                "gwd_cov_frozen_split0": round(gwd_cov_list[0], 6),
                "gwd_cov_r20_mean": round(float(gwd_cov.mean()), 6),
                "gwd_cov_r20_range": [round(float(gwd_cov.min()), 6), round(float(gwd_cov.max()), 6)],
                # nominal-alpha (matched NOMINAL) -- reproduces holm8_r20_exploratory
                "ratio_nominal_frozen_split0": round(rn[0], 6),
                "ratio_nominal_r20_mean": round(float(rn.mean()), 6),
                "ratio_nominal_r20_range": [round(float(rn.min()), 6), round(float(rn.max()), 6)],
                "cov_base_nominal_r20_mean": round(float(np.mean(d["cov_base_nominal"])), 6),
                # coverage-matched (matched REALIZED coverage) -- the Part A headline
                "ratio_matched_frozen_split0": round(rm[0], 6),
                "ratio_matched_r20_mean": round(float(rm.mean()), 6),
                "ratio_matched_r20_range": [round(float(rm.min()), 6), round(float(rm.max()), 6)],
                "alpha_prime_r20_mean": round(float(np.mean(d["alpha_prime"])), 6),
                "alpha_prime_r20_range": [round(float(np.min(d["alpha_prime"])), 6),
                                          round(float(np.max(d["alpha_prime"])), 6)],
                "cov_base_matched_r20_mean": round(float(np.mean(d["cov_base_matched"])), 6),
                "cov_match_abs_gap_r20_mean": round(float(np.mean(
                    np.abs(np.array(d["cov_base_matched"]) - gwd_cov))), 6),
                "matched_ratio_below_one_all_splits": bool(np.all(rm < 1.0)),
                "n_splits_matched_gwd_smaller": int(np.sum(rm < 1.0)),
                "per_split_ratio_matched": [round(float(x), 6) for x in rm],
                "per_split_ratio_nominal": [round(float(x), 6) for x in rn],
                "per_split_alpha_prime": [round(float(x), 6) for x in d["alpha_prime"]],
                "per_split_cov_base_matched": [round(float(x), 6) for x in d["cov_base_matched"]],
            }
            cells_out.append(cell)
            print(f"  SUMMARY {det}/{ds} vs {base}: "
                  f"nominal ratio {cell['ratio_nominal_r20_mean']:.3f} "
                  f"(base cov {cell['cov_base_nominal_r20_mean']:.3f}) -> "
                  f"MATCHED ratio {cell['ratio_matched_r20_mean']:.3f} "
                  f"[{cell['ratio_matched_r20_range'][0]:.3f},{cell['ratio_matched_r20_range'][1]:.3f}] "
                  f"(base cov {cell['cov_base_matched_r20_mean']:.3f}, gap {cell['cov_match_abs_gap_r20_mean']:.4f}, "
                  f"a'~{cell['alpha_prime_r20_mean']:.3f}) "
                  f"{'GWD-smaller-all' if cell['matched_ratio_below_one_all_splits'] else '<<CROSSES 1>>'}", flush=True)
    out = {
        "label": "POST-HOC coverage-matched efficiency ablation (Part A); coverage-FAIR, NOT preregistered/confirmatory",
        "protocol": "40/20/40 scene split, split-seed = repeat index; calibrate on MATCHING scenes, "
                    "eval on EVAL scenes; pooled split-conformal (mondrian_field=None). GWD held at "
                    "nominal alpha=0.10; each baseline re-calibrated at alpha' so its realized eval JOINT "
                    "coverage matches GWD's realized eval coverage on the same split.",
        "metric": "(cx,cy)-slice area ratio A_gwd/A_base at matched realized coverage (set_size_cxcy_slice)",
        "alpha_gwd": ALPHA_GWD, "n_repeats": N_REPEATS,
        "cells": cells_out,
        "elapsed_s": round(time.time() - t0, 1),
    }
    outpath = OUT / "results.json"
    json.dump(out, open(outpath, "w"), indent=1, default=str)
    print(f"\nCOVERAGE_MATCHED_DONE cells={len(cells_out)} elapsed={out['elapsed_s']}s -> {outpath}", flush=True)


if __name__ == "__main__":
    main()

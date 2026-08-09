#!/usr/bin/env python3
"""Coverage-matched efficiency ablation -- Config-B extension (2026-07-15).

Extends coverage_matched_2026-07-13/ (the frozen Part A ablation) to the two new
Config-B DIOR-R cells (RoI Transformer, S2A-Net; configB_cells_2026-07-15/) WITHOUT
touching the frozen parent script. Every per-split computation
(_threshold, precompute_residuals, baseline_coverage_and_area, match_alpha, the
r==0 g1_calibrate cross-check) is imported and reused verbatim from
coverage_matched_runner.py -- identical protocol to the four frozen cells, same
call sequence as coverage_matched_regime_runner.py's parent-reuse pattern.

Protocol (unchanged, see parent docstring): same 40/20/40 repeated scene splits
(split-seed = repeat index, N_REPEATS=20), same 2 baselines (naive-coord, hull),
GWD held at nominal alpha=0.10, each baseline re-tuned to alpha' so its realized
eval JOINT coverage matches GWD's realized eval coverage on the same split.
POST-HOC, coverage-FAIR, NOT preregistered/confirmatory (identical caveat to parent).
"""
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path("/home/zeyufu/Desktop/ml-reliability-research/reliability-commons/tools/rotcert")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "coverage_matched_2026-07-13"))
import numpy as np  # noqa: E402
from rotcert import certify as _certify  # noqa: E402
from rotcert.scores import SCORES as _SCORES  # noqa: E402
import coverage_matched_runner as parent  # noqa: E402

BASELINES = parent.BASELINES
ALPHA_GWD = parent.ALPHA_GWD
N_REPEATS = parent.N_REPEATS
OUT = ROOT / "coverage_matched_configB_ext_2026-07-15"

NEW_CELLS = [
    ("roi_trans", "dior", ROOT / "configB_cells_2026-07-15/roi_trans/matched.jsonl"),
    ("s2anet", "dior", ROOT / "configB_cells_2026-07-15/s2anet/matched.jsonl"),
]


def run_cell(det, ds, path, t0):
    from rotcert import splits as _splits
    matched = parent.load_tp(path)
    gwd_res, base_res, scene_code, scenes = parent.precompute_residuals(matched)
    reps = _splits.repeated_scene_splits(scenes, n_repeats=N_REPEATS, cal_frac=0.4, match_frac=0.2)
    code_of = {s: i for i, s in enumerate(scenes)}
    print(f"\n=== {det}/{ds}: {len(matched)} TP over {len(scenes)} scenes ({time.time()-t0:.0f}s) ===", flush=True)
    acc = {b: {"ratio_matched": [], "ratio_nominal": [], "alpha_prime": [],
               "cov_base_matched": [], "cov_base_nominal": [], "area_base_matched": [],
               "area_base_nominal": []} for b in BASELINES}
    gwd_cov_list, gwd_area_list = [], []
    for r, rep in enumerate(reps):
        match_codes = np.array([code_of[s] for s in rep["matching"]], dtype=np.int64)
        eval_codes = np.array([code_of[s] for s in rep["eval"]], dtype=np.int64)
        match_mask = np.isin(scene_code, match_codes)
        eval_mask = np.isin(scene_code, eval_codes)
        gwd_cal = np.sort(gwd_res[match_mask])
        n_cal_gwd = gwd_cal.size
        q_gwd = parent._threshold(gwd_cal, n_cal_gwd, ALPHA_GWD)
        if r == 0:  # one-time validation that _threshold == g1_calibrate's SplitConformal (parent's gate, reused)
            match_rows = [m for m, keep in zip(matched, match_mask) if keep]
            calib = _certify.g1_calibrate(match_rows, "gwd", alpha=ALPHA_GWD,
                                          mondrian_field=None)["strata"][None]["calibrator"]
            assert abs(float(calib.q_hat) - q_gwd) < 1e-9, (det, ds, calib.q_hat, q_gwd)
        A_gwd = float(math.pi * q_gwd * q_gwd)
        c_gwd = float(np.mean(gwd_res[eval_mask] <= q_gwd))
        gwd_cov_list.append(c_gwd)
        gwd_area_list.append(A_gwd)
        for base in BASELINES:
            score = _SCORES[base]
            cal_sorted = {c: np.sort(base_res[base][c][match_mask]) for c in score.coord_names}
            n_cal = int(match_mask.sum())
            eval_arr = {c: base_res[base][c][eval_mask] for c in score.coord_names}
            cov_nom, area_nom, _ = parent.baseline_coverage_and_area(score, cal_sorted, n_cal, eval_arr, ALPHA_GWD)
            a_star, cov_m, area_m, _ = parent.match_alpha(score, cal_sorted, n_cal, eval_arr, c_gwd)
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
    cells_out = []
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
            "ratio_nominal_frozen_split0": round(rn[0], 6),
            "ratio_nominal_r20_mean": round(float(rn.mean()), 6),
            "ratio_nominal_r20_range": [round(float(rn.min()), 6), round(float(rn.max()), 6)],
            "cov_base_nominal_r20_mean": round(float(np.mean(d["cov_base_nominal"])), 6),
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
    return cells_out


def main():
    t0 = time.time()
    cells_out = []
    for det, ds, path in NEW_CELLS:
        cells_out.extend(run_cell(det, ds, path, t0))
    out = {
        "label": "POST-HOC coverage-matched efficiency ablation (Part A) -- Config-B extension "
                 "(roi_trans, s2anet DIOR-R cells); coverage-FAIR, NOT preregistered/confirmatory. "
                 "Extends coverage_matched_2026-07-13/ (frozen parent, untouched); every per-split "
                 "helper imported and reused verbatim from coverage_matched_runner.py.",
        "protocol": "Identical to coverage_matched_2026-07-13/coverage_matched_runner.py: 40/20/40 scene "
                    "split, split-seed = repeat index; calibrate on MATCHING scenes, eval on EVAL scenes; "
                    "pooled split-conformal (mondrian_field=None). GWD held at nominal alpha=0.10; each "
                    "baseline re-calibrated at alpha' so its realized eval JOINT coverage matches GWD's "
                    "realized eval coverage on the same split.",
        "metric": "(cx,cy)-slice area ratio A_gwd/A_base at matched realized coverage (set_size_cxcy_slice)",
        "alpha_gwd": ALPHA_GWD, "n_repeats": N_REPEATS,
        "cells": cells_out,
        "elapsed_s": round(time.time() - t0, 1),
    }
    outpath = OUT / "results.json"
    json.dump(out, open(outpath, "w"), indent=1, default=str)
    print(f"\nCOVERAGE_MATCHED_CONFIGB_EXT_DONE cells={len(cells_out)} elapsed={out['elapsed_s']}s -> {outpath}", flush=True)


if __name__ == "__main__":
    main()

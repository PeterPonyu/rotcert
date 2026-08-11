#!/usr/bin/env python3
"""EXPLORATORY (NOT the frozen confirmatory test) robustness of the C2 GWD-vs-baseline
set-size contrast across R=20 scene splits.

Motivation (degeneracy of the frozen single-split Holm-8, verified below): under pooled
split-conformal (holm8_run.py uses mondrian_field=None) every construction yields ONE
scalar (cx,cy)-slice area per cell -- scores.set_size_cxcy_slice is a pure function of
the calibrator's scalar quantiles and ignores pred_obb -- so on a single split every
eval detection in a cell carries the IDENTICAL set size. The frozen per-pair permutation
p is then deterministic at the 1/(n_perm+1)=1/2001 floor for ANY negative effect and the
class-blocked bootstrap CI is zero-width; the effective n per cell is 1, not ~20k-40k.

This script therefore treats the PER-SPLIT set-size ratio as the atom and resamples the
SPLIT (the honest source of variability): for each cell we recompute the single GWD and
baseline (cx,cy)-slice areas on R=20 independent 40/20/40 scene splits (split-seed =
repeat index, reusing rotcert.splits.repeated_scene_splits, the same protocol the G1
headline uses), giving 20 per-split log-ratios per (cell, baseline). We report their
mean/min/max, an exact two-sided sign test across the 20 splits, and a percentile CI over
splits. Label EXPLORATORY everywhere.
"""
import json
import sys
import time
from pathlib import Path


def _portal_commons_root():
    import os
    from pathlib import Path
    for key in ("COMMONS_ROOT", "RELIABILITY_COMMONS"):
        v = os.environ.get(key)
        if v:
            p = Path(v).expanduser().resolve()
            if p.is_dir():
                return p
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        for cand in (parent / "reliability-commons", parent.parent / "reliability-commons"):
            if cand.is_dir():
                return cand
    raise RuntimeError(
        "Set COMMONS_ROOT to the reliability-commons checkout (or place it as a sibling of this repo)."
    )

def _portal_repo_root():
    from pathlib import Path
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / ".git").exists() or (p / "pyproject.toml").exists() or (p / "README.md").exists():
            return p
    return here

ROOT = _portal_repo_root()
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from scipy.stats import binomtest  # noqa: E402
from rotcert import io as _io, certify as _certify, splits as _splits  # noqa: E402
from rotcert.scores import SCORES, set_size_cxcy_slice  # noqa: E402

# Same 4 cells and 2 baselines as the frozen holm8_run.py.
CELLS = [
    ("rtmdet", "dota", ROOT / "pilot_results_2026-07-10/pulled/root/autodl-tmp/rotcert_results/pilot/matched.jsonl"),
    ("orcnn", "dota", ROOT / "orcnn_dota_cert_2026-07-11/matched.jsonl"),
    ("orcnn", "dior", ROOT / "dior_cert_results_2026-07-11/orcnn_fixedgt/matched.jsonl"),
    ("rtmdet", "dior", ROOT / "dior_cert_results_2026-07-11/rtmdet/matched.jsonl"),
]
BASELINES = ["naive-coord", "hull"]
ALPHA_COV = 0.10
N_REPEATS = 20
OUT = ROOT / "holm8_2026-07-12"


def load_tp(path):
    rows = _io.load_jsonl(path)
    return [{"pred_obb": r["pred_obb"], "gt_obb": r["gt_obb"], "class": r["class"],
             "scene_id": r["scene_id"]} for r in rows if r.get("match_type") == "tp"]


def cell_size_and_cov(matched, score_name, match_scenes, eval_scenes):
    """Return (single_set_size, n_unique_sizes, realized_coverage) for one split.

    Under pooled calibration the single_set_size is the ONE value shared by every eval
    detection; n_unique_sizes is reported so the degeneracy is measured, not assumed."""
    match_rows = [m for m in matched if m["scene_id"] in match_scenes]
    eval_rows = [m for m in matched if m["scene_id"] in eval_scenes]
    cert = _certify.g1_calibrate(match_rows, score_name, alpha=ALPHA_COV, mondrian_field=None)
    calibrator = cert["strata"][None]["calibrator"]
    score = SCORES[score_name]
    sizes, covered = [], []
    for m in eval_rows:
        s = set_size_cxcy_slice(score_name, calibrator, np.asarray(m["pred_obb"], dtype=float))
        if s is None or not np.isfinite(s) or s <= 0:
            continue
        sizes.append(float(s))
        covered.append(bool(score.covers(calibrator, m["pred_obb"], m["gt_obb"])))
    sizes = np.asarray(sizes)
    n_unique = int(np.unique(np.round(sizes, 6)).size)
    return float(sizes[0]), n_unique, (float(np.mean(covered)) if covered else float("nan"))


def verify_degeneracy():
    """Independent re-verification on orcnn/dota (reviewer measured GWD 1 unique value
    126.58, naive 1 value 212.51 over 20,872 pairs on the frozen split-seed 0)."""
    det, ds, path = CELLS[1]
    matched = load_tp(path)
    scenes = sorted({m["scene_id"] for m in matched})
    reps = _splits.repeated_scene_splits(scenes, n_repeats=1, cal_frac=0.4, match_frac=0.2)
    match_s, eval_s = set(reps[0]["matching"]), set(reps[0]["eval"])
    print("=== DEGENERACY VERIFICATION (orcnn/dota, frozen split-seed 0) ===", flush=True)
    for sc in ("gwd", "naive-coord", "hull"):
        size, nuniq, cov = cell_size_and_cov(matched, sc, match_s, eval_s)
        n_eval = len([m for m in matched if m["scene_id"] in eval_s])
        print(f"  {sc:12s}: unique_set_sizes={nuniq} over {n_eval} eval dets, "
              f"value={size:.2f}, realized_cov={cov:.4f}", flush=True)


def main():
    t0 = time.time()
    verify_degeneracy()
    print(f"\n=== R={N_REPEATS} EXPLORATORY across-split contrast ===", flush=True)
    cells_out = []
    for det, ds, path in CELLS:
        matched = load_tp(path)
        scenes = sorted({m["scene_id"] for m in matched})
        reps = _splits.repeated_scene_splits(scenes, n_repeats=N_REPEATS, cal_frac=0.4, match_frac=0.2)
        # Per-split single set size for gwd and each baseline.
        gwd_sizes = np.empty(N_REPEATS)
        gwd_cov = np.empty(N_REPEATS)
        for r, rep in enumerate(reps):
            gwd_sizes[r], _, gwd_cov[r] = cell_size_and_cov(
                matched, "gwd", set(rep["matching"]), set(rep["eval"]))
        for base in BASELINES:
            b_sizes = np.empty(N_REPEATS)
            b_cov = np.empty(N_REPEATS)
            for r, rep in enumerate(reps):
                b_sizes[r], _, b_cov[r] = cell_size_and_cov(
                    matched, base, set(rep["matching"]), set(rep["eval"]))
            log_ratios = np.log(gwd_sizes) - np.log(b_sizes)
            n_neg = int(np.sum(log_ratios < 0))   # GWD strictly smaller on that split
            n_pos = int(np.sum(log_ratios > 0))
            n_tie = int(np.sum(log_ratios == 0))
            # Exact two-sided sign test across splits (ties dropped, standard convention).
            n_eff = n_neg + n_pos
            sign_p = float(binomtest(min(n_neg, n_pos), n_eff, 0.5,
                                     alternative="two-sided").pvalue) if n_eff > 0 else 1.0
            cell = {
                "baseline": base, "detector": det, "dataset": ds,
                "n_splits": N_REPEATS,
                "per_split_log_ratio": [round(float(x), 6) for x in log_ratios],
                "per_split_ratio": [round(float(np.exp(x)), 6) for x in log_ratios],
                "mean_log_ratio": round(float(np.mean(log_ratios)), 6),
                "min_log_ratio": round(float(np.min(log_ratios)), 6),
                "max_log_ratio": round(float(np.max(log_ratios)), 6),
                "mean_ratio": round(float(np.exp(np.mean(log_ratios))), 6),
                "min_ratio": round(float(np.exp(np.min(log_ratios))), 6),
                "max_ratio": round(float(np.exp(np.max(log_ratios))), 6),
                "ci95_ratio_percentile": [round(float(np.exp(np.percentile(log_ratios, 2.5))), 6),
                                          round(float(np.exp(np.percentile(log_ratios, 97.5))), 6)],
                "n_splits_gwd_smaller": n_neg, "n_splits_gwd_larger": n_pos, "n_ties": n_tie,
                "sign_test_p_two_sided": round(sign_p, 8),
                "any_ratio_crosses_one": bool(np.any(log_ratios >= 0)),
                "mean_cov_gwd": round(float(np.mean(gwd_cov)), 4),
                "mean_cov_baseline": round(float(np.mean(b_cov)), 4),
            }
            cells_out.append(cell)
            flag = "  <<< RATIO CROSSES 1" if cell["any_ratio_crosses_one"] else ""
            print(f"  [{det}/{ds}] vs {base}: mean_ratio={cell['mean_ratio']:.4f} "
                  f"range=[{cell['min_ratio']:.4f},{cell['max_ratio']:.4f}] "
                  f"gwd_smaller={n_neg}/{N_REPEATS} sign_p={sign_p:.2e}{flag}", flush=True)
    out = {
        "label": "EXPLORATORY -- NOT the frozen preregistered Holm-8 confirmatory test",
        "purpose": "across-split robustness of the per-split GWD-vs-baseline set-size ratio",
        "n_repeats": N_REPEATS,
        "split_protocol": "40/20/40 scene split, split-seed = repeat index 0..19 "
                          "(rotcert.splits.repeated_scene_splits, same as G1 headline)",
        "alpha_coverage": ALPHA_COV,
        "calibration": "pooled split-conformal (mondrian_field=None), matching the frozen "
                       "Holm-8 contrast (NOT the per-class Mondrian shipped G1 certificate)",
        "atom": "per-split single (cx,cy)-slice area per construction (pooled => 1 unique "
                "value per cell per split); effective n = n_splits, not n_eval_pairs",
        "cells": cells_out,
        "elapsed_s": round(time.time() - t0, 1),
    }
    outpath = OUT / "holm8_r20_exploratory.json"
    json.dump(out, open(outpath, "w"), indent=1, default=str)
    print(f"\nR20_EXPLORATORY_DONE cells={len(cells_out)} -> {outpath}")


if __name__ == "__main__":
    main()

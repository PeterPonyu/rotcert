#!/usr/bin/env python3
"""Regime-conditional coverage-matched efficiency analysis (2026-07-13).

Question this runner answers
----------------------------
The parent coverage-matched ablation (coverage_matched_2026-07-13) found, at MATCHED
realized coverage, GWD is smaller than naive-coord on DIOR-R (~0.79-0.83x), a WASH on
DOTA (~0.93x), and always LARGER than the hull. The efficiency story was walked back to
"GWD is coverage-efficient only vs naive on DIOR-R." The proposed repair (COMPUTE-PLAN
§5): GWD's matched-coverage advantage should track OBJECT ELONGATION -- it should win on
elongated objects (where a seam-continuous, angle-aware score matters) and be neutral on
compact ones. This runner tests that mechanistically, WITHIN each existing cell, by
running the identical coverage-matched protocol CONDITIONED on an object-geometry regime.

Regime definition (aspect ratio of the matched GT box, convention-free)
-----------------------------------------------------------------------
AR = max(w,h)/min(w,h) of each true-positive's gt_obb (>=1; independent of angle
convention). Primary two-way split at a FIXED, pre-committed cut AR_CUT=3.0 (so strata
are comparable across cells): "compact" (AR<3) vs "elongated" (AR>=3). AR=3 is a standard
elongated-object cutoff in aerial detection. A per-cell MEDIAN split is also reported as a
threshold-robustness check (its cut varies by cell, so it is not cross-cell comparable but
shows the DIRECTION is not an artifact of the 3.0 choice). Regime "all" = no AR
restriction = the parent computation (regression guard).

Protocol (conditioned parent protocol; regression-guarded)
----------------------------------------------------------
Same 4 cells, same 2 baselines, same 40/20/40 repeated scene splits (split-seed = repeat
index) computed ONCE over ALL scenes. For each regime we apply an EXTRA aspect-ratio mask
on top of the calibration/eval scene masks and re-run the parent's exact per-split logic
(imported from coverage_matched_runner: _threshold, baseline_coverage_and_area,
match_alpha): GWD calibrated at nominal alpha=0.10 on the regime's calibration objects ->
q_gwd, A_gwd=pi*q_gwd^2, realized eval coverage c_gwd on the regime's eval objects; each
baseline re-tuned to alpha' so its realized eval JOINT coverage on the regime's eval
objects matches c_gwd; ratio = A_gwd / A_base (<1 => GWD smaller at matched coverage).
A split is used for a regime only if it has >= MIN_CAL calibration and >= MIN_EVAL eval
objects in that regime (recorded).

Honesty
-------
This is POST-HOC and coverage-FAIR, NOT preregistered/confirmatory (identical caveat to
the parent). It repairs the efficiency claim ONLY if elongated ratio < compact ratio holds
in the data. The verdict block leads with whether it does; if it does not, that is stated
first.
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
from rotcert import io as _io, splits as _splits  # noqa: E402
from rotcert.scores import SCORES  # noqa: E402
# Reuse the PARENT's exact helpers so regime="all" is bit-identical to the frozen ablation.
import coverage_matched_runner as parent  # noqa: E402

CELLS = parent.CELLS
BASELINES = parent.BASELINES
ALPHA_GWD = parent.ALPHA_GWD
N_REPEATS = parent.N_REPEATS
AR_CUT = 3.0
MIN_CAL = 100      # regime must have >= this many calibration objects in a split to be used
MIN_EVAL = 100     # and >= this many eval objects (coverage matching needs a stable target)
OUT = ROOT / "coverage_matched_regime_2026-07-13"
FROZEN_PARENT = ROOT / "coverage_matched_2026-07-13" / "results.json"


def aspect_ratio(gt_obb):
    w, h = float(gt_obb[2]), float(gt_obb[3])
    lo = min(abs(w), abs(h))
    return (max(abs(w), abs(h)) / lo) if lo > 1e-9 else float("inf")


def precompute(matched):
    """Parent residual precompute + per-object aspect ratio."""
    gwd = SCORES["gwd"]
    n = len(matched)
    gwd_res = np.empty(n, float)
    base_res = {b: {c: np.empty(n, float) for c in SCORES[b].coord_names} for b in BASELINES}
    ar = np.empty(n, float)
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
        ar[i] = aspect_ratio(m["gt_obb"])
        scene_code[i] = scene_to_code[m["scene_id"]]
    return gwd_res, base_res, ar, scene_code, scenes


def regime_masks(ar):
    """Return {regime_name: boolean object mask}. 'all', fixed AR=3 split, median split."""
    med = float(np.median(ar))
    return {
        "all": np.ones(ar.shape, bool),
        "compact": ar < AR_CUT,
        "elongated": ar >= AR_CUT,
        "compact_medsplit": ar < med,
        "elongated_medsplit": ar >= med,
    }, med


def run_cell(det, ds, path, t0):
    matched = parent.load_tp(path)
    gwd_res, base_res, ar, scene_code, scenes = precompute(matched)
    reps = _splits.repeated_scene_splits(scenes, n_repeats=N_REPEATS, cal_frac=0.4, match_frac=0.2)
    code_of = {s: i for i, s in enumerate(scenes)}
    rmasks, med = regime_masks(ar)
    print(f"\n=== {det}/{ds}: {len(matched)} TP / {len(scenes)} scenes; "
          f"AR mean={ar.mean():.2f} med={med:.2f} fracEl(>= {AR_CUT})={(ar>=AR_CUT).mean():.3f} "
          f"({time.time()-t0:.0f}s) ===", flush=True)

    # accumulators: [regime][baseline] -> lists
    acc = {rg: {b: {"ratio": [], "alpha_prime": [], "cov_base": [], "n_cal": [], "n_eval": []}
                for b in BASELINES} for rg in rmasks}
    gwd_cov = {rg: [] for rg in rmasks}

    for r, rep in enumerate(reps):
        match_codes = np.array([code_of[s] for s in rep["matching"]], dtype=np.int64)
        eval_codes = np.array([code_of[s] for s in rep["eval"]], dtype=np.int64)
        split_match = np.isin(scene_code, match_codes)
        split_eval = np.isin(scene_code, eval_codes)
        for rg, rmask in rmasks.items():
            m_mask = split_match & rmask
            e_mask = split_eval & rmask
            n_cal = int(m_mask.sum()); n_eval = int(e_mask.sum())
            if n_cal < MIN_CAL or n_eval < MIN_EVAL:
                continue
            gwd_cal = np.sort(gwd_res[m_mask])
            q_gwd = parent._threshold(gwd_cal, n_cal, ALPHA_GWD)
            if not math.isfinite(q_gwd):
                continue
            A_gwd = float(math.pi * q_gwd * q_gwd)
            c_gwd = float(np.mean(gwd_res[e_mask] <= q_gwd))
            gwd_cov[rg].append(c_gwd)
            for base in BASELINES:
                score = SCORES[base]
                cal_sorted = {c: np.sort(base_res[base][c][m_mask]) for c in score.coord_names}
                eval_arr = {c: base_res[base][c][e_mask] for c in score.coord_names}
                a_star, cov_m, area_m, _ = parent.match_alpha(score, cal_sorted, n_cal, eval_arr, c_gwd)
                acc[rg][base]["ratio"].append(A_gwd / area_m)
                acc[rg][base]["alpha_prime"].append(a_star)
                acc[rg][base]["cov_base"].append(cov_m)
                acc[rg][base]["n_cal"].append(n_cal)
                acc[rg][base]["n_eval"].append(n_eval)

    rows = []
    for rg in rmasks:
        for base in BASELINES:
            d = acc[rg][base]
            rm = np.array(d["ratio"], float)
            if rm.size == 0:
                rows.append({"detector": det, "dataset": ds, "baseline": base, "regime": rg,
                             "n_splits_used": 0, "note": "no split met MIN_CAL/MIN_EVAL"})
                continue
            rows.append({
                "detector": det, "dataset": ds, "baseline": base, "regime": rg,
                "ar_cut": AR_CUT if rg in ("compact", "elongated") else (round(med, 4) if "medsplit" in rg else None),
                "n_splits_used": int(rm.size),
                "n_obj_regime_mean_cal": round(float(np.mean(d["n_cal"])), 1),
                "n_obj_regime_mean_eval": round(float(np.mean(d["n_eval"])), 1),
                "ratio_matched_mean": round(float(rm.mean()), 6),
                "ratio_matched_median": round(float(np.median(rm)), 6),
                "ratio_matched_range": [round(float(rm.min()), 6), round(float(rm.max()), 6)],
                "n_splits_gwd_smaller": int(np.sum(rm < 1.0)),
                "frac_splits_gwd_smaller": round(float(np.mean(rm < 1.0)), 4),
                "gwd_cov_mean": round(float(np.mean(gwd_cov[rg])), 6),
                "cov_base_matched_mean": round(float(np.mean(d["cov_base"])), 6),
                "cov_match_abs_gap_mean": round(float(np.mean(np.abs(
                    np.array(d["cov_base"]) - np.array(gwd_cov[rg])))), 6),
                "alpha_prime_mean": round(float(np.mean(d["alpha_prime"])), 6),
                "per_split_ratio_matched": [round(float(x), 6) for x in rm],
            })
    return rows, {"ar_mean": round(float(ar.mean()), 4), "ar_median": round(med, 4),
                  "ar_p25": round(float(np.percentile(ar, 25)), 4),
                  "ar_p75": round(float(np.percentile(ar, 75)), 4),
                  "ar_p90": round(float(np.percentile(ar, 90)), 4),
                  "ar_max": round(float(ar.max()), 4),
                  "frac_elongated_ge3": round(float((ar >= AR_CUT).mean()), 4),
                  "n_tp": len(matched), "n_scenes": len(scenes)}


def regression_guard(rows):
    """regime='all' ratio_matched_mean must equal the frozen parent ratio_matched_r20_mean."""
    frozen = json.load(open(FROZEN_PARENT))
    fmap = {(c["detector"], c["dataset"], c["baseline"]): c["ratio_matched_r20_mean"]
            for c in frozen["cells"]}
    checks = []
    max_abs = 0.0
    for row in rows:
        if row["regime"] != "all" or row.get("n_splits_used", 0) == 0:
            continue
        key = (row["detector"], row["dataset"], row["baseline"])
        fval = fmap.get(key)
        if fval is None:
            continue
        diff = abs(row["ratio_matched_mean"] - fval)
        max_abs = max(max_abs, diff)
        checks.append({"cell": "/".join(key), "regime_all_ratio": row["ratio_matched_mean"],
                       "frozen_parent_ratio": fval, "abs_diff": round(diff, 9)})
    return {"n_checked": len(checks), "max_abs_diff": round(max_abs, 9),
            "passed": bool(max_abs < 1e-6), "checks": checks}


def main():
    t0 = time.time()
    all_rows = []
    ar_summ = {}
    for det, ds, path in CELLS:
        rows, arinfo = run_cell(det, ds, path, t0)
        all_rows.extend(rows)
        ar_summ[f"{det}/{ds}"] = arinfo
        # concise per-cell verdict line
        def get(rg, base):
            for x in rows:
                if x["regime"] == rg and x["baseline"] == base and x.get("n_splits_used", 0) > 0:
                    return x
            return None
        for base in BASELINES:
            a = get("all", base); c = get("compact", base); e = get("elongated", base)
            if a and c and e:
                print(f"  {det}/{ds} vs {base}: all={a['ratio_matched_mean']:.3f} "
                      f"compact={c['ratio_matched_mean']:.3f} elongated={e['ratio_matched_mean']:.3f} "
                      f"-> {'ELONG<COMPACT (hyp holds)' if e['ratio_matched_mean'] < c['ratio_matched_mean'] else 'ELONG>=COMPACT (hyp FAILS)'}",
                      flush=True)

    guard = regression_guard(all_rows)
    print(f"\n[regression guard] regime='all' vs frozen parent: "
          f"checked={guard['n_checked']} max_abs_diff={guard['max_abs_diff']} passed={guard['passed']}", flush=True)

    out = {
        "label": "Regime-conditional coverage-matched efficiency analysis; POST-HOC, coverage-FAIR, "
                 "NOT preregistered/confirmatory (extends coverage_matched_2026-07-13)",
        "regime_definition": f"AR=max(w,h)/min(w,h) of the matched GT box; compact=AR<{AR_CUT}, "
                             f"elongated=AR>={AR_CUT} (fixed cut); *_medsplit = per-cell median split; "
                             "all=no AR restriction (regression guard vs parent).",
        "protocol": "Parent 40/20/40 repeated scene splits (seed=repeat idx) computed once over all scenes; "
                    "an extra aspect-ratio mask conditions each split; parent per-split logic "
                    "(_threshold, match_alpha) reused verbatim; a split is used per regime only if "
                    f">= {MIN_CAL} cal and >= {MIN_EVAL} eval objects fall in that regime.",
        "metric": "(cx,cy)-slice area ratio A_gwd/A_base at matched realized coverage (<1 => GWD smaller)",
        "alpha_gwd": ALPHA_GWD, "n_repeats": N_REPEATS, "ar_cut": AR_CUT,
        "min_cal": MIN_CAL, "min_eval": MIN_EVAL,
        "aspect_ratio_by_cell": ar_summ,
        "regression_guard": guard,
        "rows": all_rows,
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "results.json", "w"), indent=1, default=str)
    print(f"\nREGIME_DONE rows={len(all_rows)} elapsed={out['elapsed_s']}s -> {OUT/'results.json'}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""C2 confirmatory Holm-8 head-to-head (FROZEN per apps-design/05 sha256
4764b230... + SIGN-OFF-RECORD-2026-07-11): 2 baseline contrasts (B1
naive-coord+Bonferroni, B2 axis-aligned hull) x 2 detectors x 2 datasets
= 8 one-sided tests (H1: GWD set size < baseline at IDENTICAL NOMINAL
calibration -- NOT matched realized coverage; see the calibration note
below), Holm alpha=0.05.

Calibration protocol (rotcert.audit.set_size_contrast docstring): one
frozen 40/20/40 cal/match/eval scene split per (detector, dataset) cell
(split-seed 0); BOTH constructions are calibrated on the SAME match split
at the IDENTICAL NOMINAL calibration level alpha=0.10 (identical split-
conformal quantile construction). NOTE: this equalizes the *nominal*
level, NOT realized coverage -- realized eval coverage is NOT matched
(GWD ~0.897-0.907 vs union-bound baselines ~0.935-0.950; the baselines
over-cover), so part of GWD's smaller region is the price of baseline
over-coverage; a coverage-matched ablation is the clean follow-up.

DEGENERACY CAVEAT (verified 2026-07-12, holm8_r20_exploratory.py): with
mondrian_field=None (pooled), scores.set_size_cxcy_slice is a pure
function of the calibrator's scalar quantiles and IGNORES pred_obb, so on
this single split EVERY eval detection in a cell carries the IDENTICAL set
size (measured: ORCNN-DOTA GWD 1 unique value 126.58 over 20,872 pairs,
naive 212.51, hull 162.30). Effective n per cell is therefore 1 (a pair
of scalars from one split), not n_eval_pairs: the class-blocked bootstrap
CI is zero-width and the sign-flip permutation p is DETERMINISTIC at the
1/(n_perm+1)=1/2001 floor for ANY negative effect, independent of its
magnitude. The preregistered test thus certifies only the per-split
deterministic inequality pi*q_gwd^2 < 4*q_cx*q_cy on the frozen split; the
substantive content is the effect SIZE (the ratios), not the p-values.
See holm8_r20_exploratory.json for the across-split (n=20) robustness that
supplies the real evidence strength. Either sign publishes.
"""
import json
import sys
import time
from pathlib import Path

ROOT = _portal_repo_root()
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from rotcert import io as _io, certify as _certify, splits as _splits, audit as _audit  # noqa: E402
from rotcert.scores import SCORES, set_size_cxcy_slice  # noqa: E402

CELLS = [
    ("rtmdet", "dota", ROOT / "pilot_results_2026-07-10/pulled/root/autodl-tmp/rotcert_results/pilot/matched.jsonl"),
    ("orcnn", "dota", ROOT / "orcnn_dota_cert_2026-07-11/matched.jsonl"),
    ("orcnn", "dior", ROOT / "dior_cert_results_2026-07-11/orcnn_fixedgt/matched.jsonl"),
    ("rtmdet", "dior", ROOT / "dior_cert_results_2026-07-11/rtmdet/matched.jsonl"),
]
BASELINES = ["naive-coord", "hull"]
ALPHA_COV = 0.10
OUT = ROOT / "holm8_2026-07-12"


def load_tp(path):
    rows = _io.load_jsonl(path)
    return [{"pred_obb": r["pred_obb"], "gt_obb": r["gt_obb"], "class": r["class"],
             "scene_id": r["scene_id"]} for r in rows if r.get("match_type") == "tp"]


def cell_sizes(matched, score_name, match_scenes, eval_scenes):
    match_rows = [m for m in matched if m["scene_id"] in match_scenes]
    eval_rows = [m for m in matched if m["scene_id"] in eval_scenes]
    cert = _certify.g1_calibrate(match_rows, score_name, alpha=ALPHA_COV, mondrian_field=None)
    calibrator = cert["strata"][None]["calibrator"]
    score = SCORES[score_name]
    sizes, classes, covered = [], [], []
    for m in eval_rows:
        s = set_size_cxcy_slice(score_name, calibrator, np.asarray(m["pred_obb"], dtype=float))
        if s is None or not np.isfinite(s) or s <= 0:
            continue
        sizes.append(float(s))
        classes.append(m["class"])
        covered.append(bool(score.covers(calibrator, m["pred_obb"], m["gt_obb"])))
    return np.asarray(sizes), classes, float(np.mean(covered)) if covered else float("nan")


def main():
    t0 = time.time()
    cells_out = []
    for det, ds, path in CELLS:
        matched = load_tp(path)
        scenes = sorted({m["scene_id"] for m in matched})
        reps = _splits.repeated_scene_splits(scenes, n_repeats=1, cal_frac=0.4, match_frac=0.2)
        cal_s, match_s, eval_s = (set(reps[0][k]) for k in ("calibration", "matching", "eval"))
        print(f"[{det}/{ds}] tp={len(matched)} scenes={len(scenes)} "
              f"cal/match/eval={len(cal_s)}/{len(match_s)}/{len(eval_s)}", flush=True)
        g_sizes, g_cls, g_cov = cell_sizes(matched, "gwd", match_s, eval_s)
        for base in BASELINES:
            b_sizes, b_cls, b_cov = cell_sizes(matched, base, match_s, eval_s)
            n = min(len(g_sizes), len(b_sizes))
            assert g_cls[:n] == b_cls[:n], "paired eval rows diverged"
            contrast = _audit.set_size_contrast(
                g_sizes[:n], b_sizes[:n], g_cls[:n], n_perm=2000, n_boot=1000, seed=0)
            # NOTE: contrast["n_classes"] is MISLABELED -- set_size_contrast returns the
            # number of paired eval rows (== n above == n_eval_pairs), not the number of
            # distinct classes. The frozen holm8_result.json carries this mislabel; it is
            # NOT rewritten (frozen artifact). Read "n_classes" there as n_eval_pairs.
            cell = {"baseline": base, "detector": det, "dataset": ds,
                    "eval_coverage_gwd": round(g_cov, 4), "eval_coverage_baseline": round(b_cov, 4),
                    "n_eval_pairs": n, **{k: contrast[k] for k in
                    ("point_log_ratio", "ci_log_ratio", "p_value", "ratio", "n_classes")}}
            cells_out.append(cell)
            print(f"  vs {base}: ratio={cell['ratio']:.4f} p={cell['p_value']:.5f} "
                  f"cov g/b={g_cov:.4f}/{b_cov:.4f}", flush=True)
    holm = _audit.holm8_confirmatory(cells_out, alpha=0.05)
    holm["protocol"] = {"alpha_coverage": ALPHA_COV, "split": "40/20/40 seed0 single",
                       "frozen_per": "apps-design/05 sha256 4764b230 + SIGN-OFF-RECORD-2026-07-11",
                       "elapsed_s": round(time.time() - t0, 1)}
    out = OUT / "holm8_result.json"
    json.dump(holm, open(out, "w"), indent=1, default=str)
    n_rej = sum(1 for r in holm["results"] if r["reject_holm"])
    print(f"HOLM8_DONE family={holm['family_size']} rejected={n_rej} -> {out}")


if __name__ == "__main__":
    main()

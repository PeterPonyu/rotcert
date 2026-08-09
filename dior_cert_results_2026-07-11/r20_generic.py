#!/usr/bin/env python3
"""R=20 scene-split repeats for a DIOR-R GWD G1 certificate (generic).
Usage: r20_generic.py <matched.jsonl> <out.json>
Same protocol as the ORCNN run: 20 repeated scene-level splits (40/20/40, split-seed
= repeat index), global GWD q_hat fit on calibration scenes, marginal coverage measured
on held-out EVAL scenes, BCa CI across the 20 repeats. alpha=0.10.
"""
import json, sys, time
import numpy as np
from scipy.stats import bootstrap as scipy_bootstrap

ROOT = "/home/zeyufu/Desktop/ml-reliability-research/reliability-commons/tools/rotcert"
sys.path.insert(0, ROOT)
from rotcert import io as _io, certify as _certify, splits as _splits
from rotcert.scores import SCORES

MATCHED, OUT = sys.argv[1], sys.argv[2]
ALPHA, R = 0.10, 20
t0 = time.time()
rows = _io.load_jsonl(MATCHED)
tp = [r for r in rows if r.get("match_type") == "tp"]
matched = [{"pred_obb": r["pred_obb"], "gt_obb": r["gt_obb"], "class": r["class"],
            "scene_id": r["scene_id"]} for r in tp]
by_scene = {}
for m in matched:
    by_scene.setdefault(m["scene_id"], []).append(m)
uniq = sorted(by_scene)
print(f"[data] {len(matched)} TP over {len(uniq)} scenes ({time.time()-t0:.0f}s)", flush=True)
score = SCORES["gwd"]
reps = _splits.repeated_scene_splits(uniq, n_repeats=R, cal_frac=0.4, match_frac=0.2)
per = []
for i, part in enumerate(reps):
    cal = [m for s in set(part["calibration"]) for m in by_scene[s]]
    ev = [m for s in set(part["eval"]) for m in by_scene[s]]
    cert = _certify.g1_calibrate(cal, "gwd", alpha=ALPHA, mondrian_field=None)
    calib = cert["strata"][None]["calibrator"]
    cov = float(np.mean([bool(score.covers(calib, np.asarray(m["pred_obb"], float),
                                            np.asarray(m["gt_obb"], float))) for m in ev]))
    per.append({"repeat": i, "coverage": cov, "q_hat": float(calib.q_hat), "n_eval": len(ev)})
    print(f"[r{i:02d}] cov={cov:.4f} q_hat={calib.q_hat:.3f}", flush=True)
covs = np.array([p["coverage"] for p in per])
bca = scipy_bootstrap((covs,), np.mean, method="BCa", n_resamples=9999,
                      confidence_level=0.95, random_state=0)
res = {"experiment": "dior_r_gwd_R20", "alpha": ALPHA, "nominal_coverage": 1 - ALPHA,
       "n_repeats": R, "matched": MATCHED,
       "marginal_coverage_across_repeats": {
           "mean": float(covs.mean()), "std": float(covs.std(ddof=1)),
           "min": float(covs.min()), "max": float(covs.max()),
           "bca_ci_95": [float(bca.confidence_interval.low), float(bca.confidence_interval.high)]},
       "q_hat_across_repeats": {"mean": float(np.mean([p["q_hat"] for p in per])),
                                "std": float(np.std([p["q_hat"] for p in per], ddof=1))},
       "per_repeat": per}
json.dump(res, open(OUT, "w"), indent=2)
m = res["marginal_coverage_across_repeats"]
print(f"[done] R=20 mean={m['mean']:.4f} BCa95=[{m['bca_ci_95'][0]:.4f},{m['bca_ci_95'][1]:.4f}] "
      f"(min {m['min']:.4f} max {m['max']:.4f}) -> {OUT}", flush=True)

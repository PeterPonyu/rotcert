#!/usr/bin/env python3
"""R=20 scene-split repeats for the DIOR-R ORCNN GWD G1 certificate.

Design-mandated out-of-sample protocol (apps-design 05 §4.2): for each of R=20
repeated scene-level splits (split-seed = repeat index, 40/20/40 cal/match/eval),
fit the GLOBAL GWD split-conformal q_hat on the calibration scenes and measure
marginal coverage on the held-out EVAL scenes. Reports per-repeat coverage + a
BCa CI across the 20 repeats (scipy). Same GWD score, same alpha=0.10 as the DOTA
pilot. Writes r20_coverage.json. Does NOT run any GWD-vs-naive head-to-head.
"""
import json, sys, time
import numpy as np
from scipy.stats import bootstrap as scipy_bootstrap


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

def _data_root():
    import os
    from pathlib import Path
    return Path(os.environ.get("DATA_ROOT", Path.home() / "data")).expanduser()

def _portfolio_root():
    """Parent of theme repos when laid out as a portfolio sibling tree."""
    from pathlib import Path
    r = _portal_repo_root()
    parent = r.parent
    markers = ("reliability-commons", "inspect-gate", "materials-mlip-research", "asr-gate")
    if any((parent / m).exists() for m in markers):
        return parent
    return parent

def _autodl_tmp():
    import os
    from pathlib import Path
    return Path(os.environ.get("AUTODL_TMP", "/tmp/autodl-tmp"))

def _conda_root():
    import os
    from pathlib import Path
    return Path(os.environ.get("CONDA_ROOT", Path.home() / "miniconda3")).expanduser()

ROOT = str(_portal_repo_root())
sys.path.insert(0, ROOT)
from rotcert import io as _io, certify as _certify, splits as _splits
from rotcert.scores import SCORES

MATCHED = ROOT + "/dior_cert_results_2026-07-11/orcnn/matched.jsonl"
OUT = ROOT + "/dior_cert_results_2026-07-11/orcnn/r20_coverage.json"
ALPHA = 0.10
R = 20

t0 = time.time()
rows = _io.load_jsonl(MATCHED)
tp = [r for r in rows if r.get("match_type") == "tp"]
matched = [{"pred_obb": r["pred_obb"], "gt_obb": r["gt_obb"], "class": r["class"],
            "scene_id": r["scene_id"]} for r in tp]
by_scene = {}
for m in matched:
    by_scene.setdefault(m["scene_id"], []).append(m)
uniq = sorted(by_scene)
print(f"[data] {len(matched)} TP rows over {len(uniq)} scenes ({time.time()-t0:.0f}s)", flush=True)

score = SCORES["gwd"]
reps = _splits.repeated_scene_splits(uniq, n_repeats=R, cal_frac=0.4, match_frac=0.2)

per_repeat = []
for i, part in enumerate(reps):
    cal_scenes = set(part["calibration"])
    eval_scenes = set(part["eval"])
    cal_rows = [m for s in cal_scenes for m in by_scene[s]]
    eval_rows = [m for s in eval_scenes for m in by_scene[s]]
    cert = _certify.g1_calibrate(cal_rows, "gwd", alpha=ALPHA, mondrian_field=None)
    calib = cert["strata"][None]["calibrator"]
    covered = [bool(score.covers(calib, np.asarray(m["pred_obb"], float),
                                 np.asarray(m["gt_obb"], float))) for m in eval_rows]
    cov = float(np.mean(covered))
    per_repeat.append({"repeat": i, "coverage": cov, "q_hat": float(calib.q_hat),
                       "n_cal": len(cal_rows), "n_eval": len(eval_rows)})
    print(f"[r{i:02d}] cov={cov:.4f} q_hat={calib.q_hat:.3f} n_eval={len(eval_rows)}", flush=True)

covs = np.array([r["coverage"] for r in per_repeat])
bca = scipy_bootstrap((covs,), np.mean, method="BCa", n_resamples=9999,
                      confidence_level=0.95, random_state=0)
result = {
    "experiment": "dior_r_orcnn_gwd_R20_scene_split_repeats",
    "alpha": ALPHA, "nominal_coverage": 1 - ALPHA, "n_repeats": R,
    "score": "gwd", "split": "scene-level 40/20/40 cal/match/eval, seed=repeat index",
    "marginal_coverage_across_repeats": {
        "mean": float(covs.mean()), "std": float(covs.std(ddof=1)),
        "min": float(covs.min()), "max": float(covs.max()),
        "bca_ci_95": [float(bca.confidence_interval.low), float(bca.confidence_interval.high)],
    },
    "q_hat_across_repeats": {"mean": float(np.mean([r["q_hat"] for r in per_repeat])),
                             "std": float(np.std([r["q_hat"] for r in per_repeat], ddof=1))},
    "per_repeat": per_repeat,
}
with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
mc = result["marginal_coverage_across_repeats"]
print(f"\n[done] R=20 marginal coverage mean={mc['mean']:.4f} "
      f"BCa95=[{mc['bca_ci_95'][0]:.4f},{mc['bca_ci_95'][1]:.4f}] "
      f"(min {mc['min']:.4f}, max {mc['max']:.4f}) -> {OUT} ({time.time()-t0:.0f}s)")

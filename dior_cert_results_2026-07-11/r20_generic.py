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

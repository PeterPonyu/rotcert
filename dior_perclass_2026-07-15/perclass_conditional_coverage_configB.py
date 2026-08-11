#!/usr/bin/env python3
"""Per-class (Mondrian) conditional-coverage analysis for the GWD G1 certificate on
the two Config-B DIOR-R cells (RoI Transformer, S2A-Net) -- SAME script pattern as
``perclass_conditional_coverage.py`` (2026-07-15), pointed at the Config-B cell dirs.

POST-HOC / DESCRIPTIVE. Mirrors the paper's Mondrian machinery exactly
(``mondrian_field="class"``) and the out-of-sample R=20 scene-split protocol of
``dior_cert_results_2026-07-11/r20_generic.py`` (identical to the frozen ORCNN/RTMDet
run this script's sibling produced). For each of the 20 DIOR classes we report the OOS
conditional coverage of the per-class GWD certificate, the number of held-out evaluation
TPs the coverage is measured on, the full-data calibration count, the G1
certifiability-floor status (alpha_min = 1/(n_cal+1) vs alpha), and the G2 recall-
certificate status (from the frozen recall.json) -- identical fields to the ORCNN/RTMDet
run so the two results.json blocks merge into one layout.

Protocol per cell (identical to r20_generic.py / perclass_conditional_coverage.py):
  * matched TRUE-POSITIVE pairs only, grouped by scene_id (scene = image for DIOR);
  * 20 repeated scene-level splits (40% calibration / 20% match-holdout / 40% eval),
    split-seed = repeat index;
  * per repeat: g1_calibrate(cal, "gwd", alpha=0.10, mondrian_field="class"), then
    g1_coverage(cert, eval) -> per-class coverage on the held-out eval scenes;
  * aggregate each class across the 20 repeats (mean / std / min coverage, mean n_eval).

Reads ONLY the just-produced Config-B cell dirs (configB_cells_2026-07-15/{roi_trans,s2anet}/
matched.jsonl, cert_gwd.json, recall.json) -- the frozen ORCNN/RTMDet cells and their
results.json entries are untouched. Writes a companion results file
(dior_perclass_2026-07-15/results_configB.json) in the SAME schema as results.json so the
two can be merged; merge_into_results() (called by main) does that merge in place, appending
the two new cells to results.json's "cells" dict without touching the existing two.

Usage: perclass_conditional_coverage_configB.py  (paths hard-coded relative to repo)
"""
import json
import os
import sys
import time

import numpy as np


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

ROOT = "${REPO_ROOT}"
sys.path.insert(0, str(_portal_commons_root()))
sys.path.insert(0, ROOT)

from rotcert import certify as _certify  # noqa: E402
from rotcert import io as _io  # noqa: E402
from rotcert import splits as _splits  # noqa: E402

ALPHA, R = 0.10, 20
CELLS = {
    "dior_roi_trans": "configB_cells_2026-07-15/roi_trans",
    "dior_s2anet": "configB_cells_2026-07-15/s2anet",
}
RESULTS_MAIN = os.path.join(ROOT, "dior_perclass_2026-07-15", "results.json")
OUT = os.path.join(ROOT, "dior_perclass_2026-07-15", "results_configB.json")


def run_cell(cell_dir):
    matched_path = os.path.join(ROOT, cell_dir, "matched.jsonl")
    cert_path = os.path.join(ROOT, cell_dir, "cert_gwd.json")
    recall_path = os.path.join(ROOT, cell_dir, "recall.json")

    t0 = time.time()
    rows = _io.load_jsonl(matched_path)
    tp = [r for r in rows if r.get("match_type") == "tp"]
    matched = [
        {"pred_obb": r["pred_obb"], "gt_obb": r["gt_obb"], "class": r["class"],
         "scene_id": r["scene_id"]}
        for r in tp
    ]
    by_scene = {}
    for m in matched:
        by_scene.setdefault(m["scene_id"], []).append(m)
    uniq = sorted(by_scene)
    classes = sorted({m["class"] for m in matched})
    print(f"[data] {os.path.basename(cell_dir)}: {len(matched)} TP over {len(uniq)} "
          f"scenes, {len(classes)} classes ({time.time()-t0:.0f}s)", flush=True)

    frozen_cert = json.load(open(cert_path))
    n_cal_full = {k: v["n_cal"] for k, v in frozen_cert["strata"].items()}
    q_hat_full = {k: v["calibrator"]["q_hat"] for k, v in frozen_cert["strata"].items()}

    recall = json.load(open(recall_path))
    recall_status = {
        k: ("certified" if v.get("certified") else "refused")
        for k, v in recall["per_class"].items()
    }

    reps = _splits.repeated_scene_splits(uniq, n_repeats=R, cal_frac=0.4, match_frac=0.2)
    cov_by_class = {c: [] for c in classes}
    n_by_class = {c: [] for c in classes}
    overall = []
    for i, part in enumerate(reps):
        cal = [m for s in set(part["calibration"]) for m in by_scene[s]]
        ev = [m for s in set(part["eval"]) for m in by_scene[s]]
        cert = _certify.g1_calibrate(cal, "gwd", alpha=ALPHA, mondrian_field="class")
        cov = _certify.g1_coverage(cert, ev)
        overall.append(cov["overall_coverage"])
        for c in classes:
            ps = cov["per_stratum"].get(c)
            if ps is not None:
                cov_by_class[c].append(ps["coverage"])
                n_by_class[c].append(ps["n"])
        print(f"[r{i:02d}] overall={cov['overall_coverage']:.4f} "
              f"n_out_of_support={cov['n_out_of_support']}", flush=True)

    per_class = {}
    for c in classes:
        covs = np.array(cov_by_class[c], dtype=float)
        ns = np.array(n_by_class[c], dtype=float)
        alpha_min = 1.0 / (n_cal_full[c] + 1)
        per_class[c] = {
            "oos_coverage_mean": float(covs.mean()),
            "oos_coverage_std": float(covs.std(ddof=1)),
            "oos_coverage_min": float(covs.min()),
            "n_eval_mean": float(ns.mean()),
            "n_cal_full": int(n_cal_full[c]),
            "q_hat_full": float(q_hat_full[c]),
            "g1_certifiability_alpha_min": float(alpha_min),
            "g1_certifiability_refused": bool(alpha_min > ALPHA),
            "recall_cert_status": recall_status.get(c, "n/a"),
        }

    overall = np.array(overall, dtype=float)
    at_or_above = sum(1 for c in classes if per_class[c]["oos_coverage_mean"] >= 1 - ALPHA)
    g1_refused = [c for c in classes if per_class[c]["g1_certifiability_refused"]]
    recall_refused = [c for c in classes if per_class[c]["recall_cert_status"] == "refused"]
    return {
        "n_classes": len(classes),
        "n_tp_total": len(matched),
        "n_scenes": len(uniq),
        "overall_oos_coverage_mean": float(overall.mean()),
        "n_classes_at_or_above_nominal": at_or_above,
        "n_classes_below_nominal": len(classes) - at_or_above,
        "min_class_coverage": min(per_class[c]["oos_coverage_mean"] for c in classes),
        "max_class_coverage": max(per_class[c]["oos_coverage_mean"] for c in classes),
        "g1_certifiability_floor_refused": g1_refused,
        "recall_certificate_refused": sorted(recall_refused),
        "recall_certificate_refused_count": len(recall_refused),
        "per_class": per_class,
    }


def main():
    res = {
        "experiment": "dior_perclass_conditional_coverage_R20_configB",
        "kind": "post-hoc / descriptive (per-class Mondrian conditional coverage)",
        "score": "gwd",
        "alpha": ALPHA,
        "nominal_coverage": 1 - ALPHA,
        "n_repeats": R,
        "split": {"cal_frac": 0.4, "match_frac": 0.2, "eval_frac": 0.4,
                  "unit": "scene (=image for DIOR)", "split_seed": "repeat index"},
        "mondrian_field": "class",
        "protocol_mirror": "dior_cert_results_2026-07-11/r20_generic.py + rotcert.certify.g1_coverage"
                            " (identical protocol/script pattern to dior_perclass_2026-07-15/"
                            "perclass_conditional_coverage.py, applied to the Config-B DIOR cells)",
        "cells": {},
    }
    for name, cell_dir in CELLS.items():
        res["cells"][name] = run_cell(cell_dir)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"[done] wrote {OUT}", flush=True)

    # merge into the main results.json layout: append the two new cell blocks,
    # touching nothing else (existing dior_orcnn_fixedgt / dior_rtmdet entries
    # and every other top-level key are preserved byte-for-byte apart from the
    # "cells" dict gaining two keys).
    main_res = json.load(open(RESULTS_MAIN))
    for name in CELLS:
        assert name not in main_res["cells"], f"{name} already present in {RESULTS_MAIN}"
        main_res["cells"][name] = res["cells"][name]
    json.dump(main_res, open(RESULTS_MAIN, "w"), indent=2)
    print(f"[done] merged Config-B cells into {RESULTS_MAIN}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""AOPG DIOR-R reproduction gate (design 05-APP-rotdet-cert.md, K3 + A2 amendment).

COMPUTE-ONLY runner. Recomputes the frozen K3 reproduction gate for the two
in-house DIOR-R detectors against the AOPG published DIOR-R table, using the
FROZEN gate function ``orchestration.phase0.reproduction_gate`` (tol=0.5, the
design's "±0.5"). No manuscript is touched.

Measured DIOR-R *test*-split mAP (single-scale, no-TTA) is read from the
in-house training-run per-class eval tables (the deployed checkpoints, matching
the inference provenance JSONs). Note: the mmrotate DIOR configs evaluate the
`val_evaluator` on `ImageSets/Main/test.txt` (11,738 images) -- so the reported
"val mAP" IS the DIOR-R test-split mAP, the same split AOPG publishes on. This
removes K3's val-vs-test gap for this arm.

Frozen target source (design A2): "the AOPG published DIOR-R table
(jbwang1997/AOPG, Apache-2.0)". That table (AOPG paper arXiv:2110.01931 Table I;
jbwang1997/AOPG repo DIOR-R model-zoo) publishes: RetinaNet-O 57.55,
Faster RCNN-O 59.54, Gliding Vertex 60.06, RoI Transformer 63.87, AOPG 64.41.
The repo's DIOR-R zoo row is exactly one entry: AOPG @ 64.41 -> the frozen
per-source reference target. Neither Oriented R-CNN nor RTMDet is a row.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROTCERT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROTCERT_ROOT / "orchestration"))

# FROZEN gate logic -- imported, not reimplemented.
from phase0 import reproduction_gate, DEFAULT_REPRO_TOL  # noqa: E402

# --- AOPG published DIOR-R table (frozen target source) -----------------------
AOPG_DIOR_TABLE = {
    "RetinaNet-O": 57.55,
    "Faster RCNN-O": 59.54,
    "Gliding Vertex": 60.06,
    "RoI Transformer": 63.87,
    "AOPG": 64.41,  # the jbwang1997/AOPG repo's sole DIOR-R zoo row == the target
}
AOPG_TARGET = AOPG_DIOR_TABLE["AOPG"]  # 64.41, single-scale, R50-FPN, DIOR-R test

# --- In-house reproduced DIOR-R test mAP (primary: training per-class tables) --
# Deployed checkpoints (== inference provenance JSONs), single-scale, no-TTA.
DETECTORS = {
    "oriented_rcnn_r50_1x": {
        "measured_map": 62.61,          # epoch_12 dota/mAP 0.6261 (deployed ckpt)
        "checkpoint": "orcnn_dior/seed_0/epoch_12.pth",
        "config": "oriented-rcnn-le90_r50_fpn_1x_dior.py",
        "schedule": "1x (12 epochs)",
        "source_log": "dior_train_results_2026-07-10 :: orcnn_dior_seed0_train.log "
                      "Epoch(val)[12][11738/11738] dota/mAP: 0.6261",
        "nearest_table_row": "RoI Transformer",  # closest two-stage published method
    },
    "rtmdet_r_l_3x": {
        "measured_map": 68.36,          # epoch_36 dota/mAP 0.6836 (deployed ckpt)
        "checkpoint": "rtmdet_r_dior/seed_0/epoch_36.pth",
        "config": "rotated_rtmdet_l-3x-dior.py",
        "schedule": "3x (36 epochs); best epoch_24=69.65, deployed epoch_36=68.36",
        "source_log": "dior_train_results_2026-07-10 :: rtmdet_r_dior_seed0_resume.log "
                      "Epoch(val)[36][11738/11738] dota/mAP: 0.6836",
        "nearest_table_row": "AOPG",  # exceeds every row in the 2021 AOPG table
    },
}


def main() -> int:
    results = {
        "gate": "K3 reproduction gate (design 05-APP-rotdet-cert.md, A2 amendment)",
        "frozen_rule": "|measured_val_map - published_val_map| <= tol; tol=0.5 mAP (K3 '±0.5')",
        "frozen_target_source": "AOPG published DIOR-R table (jbwang1997/AOPG, Apache-2.0)",
        "aopg_dior_table": AOPG_DIOR_TABLE,
        "aopg_target_used": AOPG_TARGET,
        "tolerance": DEFAULT_REPRO_TOL,
        "eval_split_note": "in-house eval runs on DIOR-R test.txt (11,738 imgs) == AOPG's split",
        "category_caveat": (
            "The AOPG DIOR-R table contains NO Oriented R-CNN or RTMDet row; the "
            "±0.5 identity-reproduction tolerance is defined for same-method "
            "reproduction. Verdicts below apply the frozen function against the "
            "AOPG-source DIOR-R reference (64.41) exactly as frozen; the gap is a "
            "cross-method gap, disclosed not chased (K3)."
        ),
        "detectors": {},
    }

    for name, d in DETECTORS.items():
        gate = reproduction_gate(d["measured_map"], AOPG_TARGET, tol=DEFAULT_REPRO_TOL)
        nearest = d["nearest_table_row"]
        gate_vs_nearest = reproduction_gate(
            d["measured_map"], AOPG_DIOR_TABLE[nearest], tol=DEFAULT_REPRO_TOL
        )
        results["detectors"][name] = {
            **{k: d[k] for k in ("measured_map", "checkpoint", "config",
                                 "schedule", "source_log")},
            "gate_vs_aopg_target": gate,
            "nearest_table_row": nearest,
            "nearest_row_map": AOPG_DIOR_TABLE[nearest],
            "gate_vs_nearest_row": gate_vs_nearest,
        }
        print(f"{name}: measured={d['measured_map']:.2f}  "
              f"vs AOPG 64.41 gap={gate['gap']:+.2f} passed={gate['passed']}  "
              f"| vs {nearest} {AOPG_DIOR_TABLE[nearest]:.2f} "
              f"gap={gate_vs_nearest['gap']:+.2f} passed={gate_vs_nearest['passed']}")

    out = Path(__file__).with_name("aopg_repro_result.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

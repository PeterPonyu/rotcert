#!/usr/bin/env python3
"""Phase-0 staging audit (design §7 Phase 0, §2.4/K5, K3): freeze scene/crop/class
counts (incl. images-per-class for the G2 power-floor check, §2.4), and check the
reproduction gate (published/zoo-consensus VAL mAP within ``ROTCERT_REPRO_TOL``).

Pure python + ``rotcert`` (no mmrotate/torch import here) -- this script reads
ALREADY-STAGED detections/GT JSONL (from ``score_rtmdet.py`` / an mAP-eval script,
box-side) and the raw image directory tree; it never runs inference itself.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from rotcert.io import load_jsonl, populate_scene_ids, validate_gts
from rotcert.ltt import power_floor_n_img

DEFAULT_REPRO_TOL = 0.5  # mAP points (design §4.7/K3: "±0.5")
DEFAULT_BETA = 0.20
DEFAULT_DELTA = 0.05
DEFAULT_GRID_SIZE = 50


def staging_counts(gt_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Scene/crop/class counts + images-per-class (design's G2 power-floor input)."""
    scenes = set()
    crops = set()
    class_counts: Dict[str, int] = defaultdict(int)
    class_scenes: Dict[str, set] = defaultdict(set)

    for r in gt_records:
        scenes.add(r["scene_id"])
        crops.add(r["image_id"])
        class_counts[r["class"]] += 1
        class_scenes[r["class"]].add(r["scene_id"])

    return {
        "n_scenes": len(scenes),
        "n_crops": len(crops),
        "n_gt_instances": len(gt_records),
        "class_instance_counts": dict(class_counts),
        "class_scene_counts": {c: len(s) for c, s in class_scenes.items()},
    }


def g2_power_floor_report(
    class_scene_counts: Dict[str, int],
    beta: float = DEFAULT_BETA,
    delta: float = DEFAULT_DELTA,
    grid_size: int = DEFAULT_GRID_SIZE,
    r_hat_assumed: float = 0.05,
) -> Dict[str, Any]:
    """Per-class a-priori LTT-HB certifiability, using the REALIZED images-per-class
    counts against design §2.4's power-floor formula. This REPLACES the design's
    pre-Phase-0 estimate (~6-9 DOTA classes) with actual counts, per §5 K5's mandate."""
    floor = power_floor_n_img(beta=beta, delta=delta, grid_size=grid_size, r_hat=r_hat_assumed)
    per_class = {}
    n_certifiable = 0
    for cls, n_img in sorted(class_scene_counts.items()):
        certifiable = n_img >= floor["bentkus_floor"]
        per_class[cls] = {"n_img": n_img, "certifiable": certifiable}
        if certifiable:
            n_certifiable += 1
    return {
        "beta": beta, "delta": delta, "grid_size": grid_size, "r_hat_assumed": r_hat_assumed,
        "power_floor": floor, "per_class": per_class, "n_classes_certifiable": n_certifiable,
        "n_classes_total": len(class_scene_counts),
        "k5_note": (
            "K5 fires if n_classes_certifiable falls more than 2 below the frozen "
            "pre-Phase-0 a-priori prediction (~6-9), OR if fewer than 4 classes clear "
            "it at all (design §5)."
        ),
    }


def reproduction_gate(measured_map: float, published_val_map: float, tol: float = DEFAULT_REPRO_TOL) -> Dict[str, Any]:
    """K3 (design §4.7/§5): measured VAL mAP must be within ``tol`` of the
    published/zoo-consensus VAL mAP (single-scale, no-TTA)."""
    gap = measured_map - published_val_map
    passed = abs(gap) <= tol
    return {
        "measured_val_map": measured_map, "published_val_map": published_val_map,
        "tolerance": tol, "gap": gap, "passed": passed,
        "note": "certify reproduced scores, never paper-quoted ones (K3)" if passed else "GATE FAILED: no certification arm runs on this detector until fixed (K3)",
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Phase-0 staging audit: scene/crop/class counts + G2 power floor + reproduction gate")
    p.add_argument("--gt", required=True, help="canonical GT JSONL for the staged val split")
    p.add_argument("--measured-val-map", type=float, required=True)
    p.add_argument("--published-val-map", type=float, required=True)
    p.add_argument("--repro-tol", type=float, default=float(os.environ.get("ROTCERT_REPRO_TOL", DEFAULT_REPRO_TOL)))
    p.add_argument("--beta", type=float, default=DEFAULT_BETA)
    p.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    p.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    p.add_argument("-o", "--out", required=True)
    args = p.parse_args(argv)

    gt_records = populate_scene_ids(validate_gts(load_jsonl(args.gt)))
    counts = staging_counts(gt_records)
    power = g2_power_floor_report(counts["class_scene_counts"], beta=args.beta, delta=args.delta, grid_size=args.grid_size)
    repro = reproduction_gate(args.measured_val_map, args.published_val_map, tol=args.repro_tol)

    result = {"staging_counts": counts, "g2_power_floor": power, "reproduction_gate": repro}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"phase0: n_scenes={counts['n_scenes']} n_crops={counts['n_crops']} "
        f"n_classes_certifiable={power['n_classes_certifiable']}/{power['n_classes_total']} "
        f"repro_gate_passed={repro['passed']} -> {args.out}"
    )
    return 0 if repro["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

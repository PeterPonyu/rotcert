#!/usr/bin/env python3
"""Flatten XHAZE-SUMMARY.json into XHAZE-TABLE.csv (one row per cell and arm; point estimate and descriptive 95% range,
empty range = withheld because a resample was undefined). Read-only on the summary."""
import csv
import json
import math
from pathlib import Path
import os

XH = Path(os.environ.get("ROTCERT_ROOT", Path(__file__).resolve().parents[3])) / "var/planc-results-20261003/analysis/xhaze"
METRICS = [("hcp_cov", "hcp|coverage"), ("pooled_cov", "pooled|coverage"), ("scene_fnr", "fnr"),
           ("e2e_hcp", "hcp|e2e_risk"), ("radius_hcp", "hcp|normalized_radius")]
ARMS = [("clean", "clean|clean")] + [(f"{b} {kind}", f"clean|{b}" if kind == "clean-cal" else f"{b}|{b}")
                                     for b in ("b06", "b12", "b18") for kind in ("clean-cal", "recal")]


def fmt(x):
    if x is None:
        return ""
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return f"{round(x, 5):g}" if abs(x) < 1e-4 else str(round(x, 5))


def main():
    S = json.loads((XH / "XHAZE-SUMMARY.json").read_text())["summary"]
    head = ["cell", "arm"] + [f"{n}{s}" for n, _ in METRICS for s in ("", "_lo", "_hi")]
    with (XH / "XHAZE-TABLE.csv").open("w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(head)
        for cell in sorted(S):
            for name, prefix in ARMS:
                row = [cell, name]
                for _, key in METRICS:
                    e = S[cell][f"{prefix}|{key}"]
                    row += [fmt(e["point"]), fmt(e.get("lower")), fmt(e.get("upper"))]
                w.writerow(row)
    print("rows", sum(1 for _ in (XH / "XHAZE-TABLE.csv").open()) - 1)


if __name__ == "__main__":
    main()

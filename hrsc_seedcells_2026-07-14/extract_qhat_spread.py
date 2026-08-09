#!/usr/bin/env python3
"""Extract the per-seed q_hat spread for the HRSC 3-seed arm (extends SEED-VARIANCE-DIGEST.md).

For each of the six HRSC seed cells the frozen R=20 record stores, per repeat, the
conformal quantile q_hat (the GWD-score region scale in Gaussian-Wasserstein units).
This script reports, per cell, the mean/min/max of q_hat across its 20 scene reseeds,
and the across-seed spread of the per-cell mean q_hat within each detector family --
the seed-sensitive quantity the seed-variance section reports (region size), as opposed
to marginal coverage which split-conformal calibration pins at nominal by construction.

Reads only frozen r20_coverage.json files; writes nothing. Run:
    python extract_qhat_spread.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLS = [
    ("Oriented R-CNN", 0, ROOT / "configA_cert_2026-07-14/hrsc_orcnn/r20_coverage.json"),
    ("Oriented R-CNN", 1, ROOT / "hrsc_seedcells_2026-07-14/orcnn_seed1/r20_coverage.json"),
    ("Oriented R-CNN", 2, ROOT / "hrsc_seedcells_2026-07-14/orcnn_seed2/r20_coverage.json"),
    ("RTMDet-R", 0, ROOT / "configA_cert_2026-07-14/hrsc_rtmdet/r20_coverage.json"),
    ("RTMDet-R", 1, ROOT / "hrsc_seedcells_2026-07-14/rtmdet_seed1/r20_coverage.json"),
    ("RTMDet-R", 2, ROOT / "hrsc_seedcells_2026-07-14/rtmdet_seed2/r20_coverage.json"),
]


def cell_stats(path):
    d = json.load(open(path))
    qs = [r["q_hat"] for r in d["per_repeat"]]
    rep = d["q_hat_across_repeats"]
    return {
        "mean": rep["mean"],
        "std": rep["std"],
        "min": min(qs),
        "max": max(qs),
        "n": len(qs),
    }


def main():
    rows = []
    fam_means = {"Oriented R-CNN": [], "RTMDet-R": []}
    for det, seed, path in CELLS:
        s = cell_stats(path)
        fam_means[det].append(s["mean"])
        rows.append((det, seed, s))
        print(f"{det:16s} seed {seed}  q_hat mean {s['mean']:.4f} "
              f"[{s['min']:.4f}, {s['max']:.4f}]  (std {s['std']:.4f}, n={s['n']})")
    print()
    for det, means in fam_means.items():
        lo, hi = min(means), max(means)
        print(f"{det:16s} across-seed mean-q_hat range: "
              f"[{lo:.4f}, {hi:.4f}]  spread {hi - lo:.4f}  "
              f"({100 * (hi - lo) / (sum(means) / len(means)):.1f}% of family mean)")


if __name__ == "__main__":
    main()

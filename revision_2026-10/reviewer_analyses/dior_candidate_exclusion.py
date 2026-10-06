#!/usr/bin/env python3
"""Exclusion sensitivity for the DIOR-R near-duplicate candidates (2026-10-04, after the second independent review).

The annotation audit (UNIT-AUDIT-v3) flags 63 candidate overlapping pairs among 51 DIOR-R test images, some across the
calibration/evaluation roles. Without claiming they are duplicates, this recomputes the source-weighting comparison of the
four seed-0 DIOR-R detector families with every candidate image removed: the frozen S1 point estimate (mean over the 20
original source splits, r = 0..19) and the archived fixed split (r = 0), class-wise Mondrian pooled and HCP thresholds at
alpha = 0.10. Removing a source keeps every other source's role. The unexcluded values must reproduce the stored point
estimates to 1e-9 percentage points before anything is written. Read-only on frozen trees; writes only
DIOR-CANDIDATE-EXCLUSION.json below var/reviewer-gaps-20261004/unit-audit.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import sys
sys.dont_write_bytecode = True
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

R = Path(os.environ.get("ROTCERT_ROOT", Path(__file__).resolve().parents[2]))
S1 = R / "revision_2026-10/execution/s1"
STORED = R / "manuscripts/revision-from-submission-20260930-v3/complete-data/S1-complete-class-results.csv"
AUDIT = R / "var/reviewer-gaps-20261004/unit-audit/UNIT-AUDIT-v3.json"
OUT = R / "var/reviewer-gaps-20261004/unit-audit/DIOR-CANDIDATE-EXCLUSION.json"
sys.path.insert(0, str(S1))
import core                          # noqa: E402
from run_s1 import CELLS, load_cell  # noqa: E402

DIOR = ("dior-orcnn-s0", "dior-roit-s0", "dior-rtmdet-s0", "dior-s2anet-s0")
NSPLIT = 20


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def macro(metrics):
    cov = metrics[:, :, 0]
    assert not np.isnan(cov).any()
    return cov.mean(axis=1)            # (pooled, hcp)


def main():
    if OUT.exists():
        raise SystemExit("receipt exists")
    audit = json.loads(AUDIT.read_text())
    cand = set(audit["result"]["dior"]["candidate_images_all"])
    cells = {c["name"]: c for c in CELLS}
    rows, inputs = [], {str(AUDIT): sha(AUDIT)}
    for name in DIOR:
        cell = cells[name]
        c, data, universe, _ = load_cell(cell)
        inputs[cell["cache"]] = cell["cache_sha256"]
        names = [str(x) for x in c["scene_names"]]
        drop = np.array([n in cand for n in names])
        base = np.zeros(len(names), dtype=np.int64)
        base[universe] = 1
        excl = base.copy()
        excl[drop] = 0
        per_split = []
        for r in range(NSPLIT):
            roles = core.original_roles(c["scene_names"], universe, r)
            m0 = macro(core.one_inner(data, base, roles, geometry=False)[1])
            m1 = macro(core.one_inner(data, excl, roles, geometry=False)[1])
            dropped = {k: int((drop & (roles == v) & (base > 0)).sum()) for k, v in (("cal", core.CAL), ("design", core.DESIGN), ("eval", core.EVAL))}
            per_split.append(dict(r=r, pooled=float(m0[0]), hcp=float(m0[1]), pooled_excl=float(m1[0]), hcp_excl=float(m1[1]),
                                  dropped_sources=dropped))
        point = dict(pooled=float(np.mean([s["pooled"] for s in per_split])), hcp=float(np.mean([s["hcp"] for s in per_split])),
                     pooled_excl=float(np.mean([s["pooled_excl"] for s in per_split])),
                     hcp_excl=float(np.mean([s["hcp_excl"] for s in per_split])))
        # reproduction of the stored primary point estimate (20 splits) and fixed-role values, class-macro
        stored = list(csv.DictReader(STORED.open()))
        for analysis, ref in (("primary", point), ("fixed-role", per_split[0])):
            rr = [r for r in stored if r["analysis"] == analysis and r["cell"] == name]
            sp, sh = np.mean([float(r["pooling_percent"]) for r in rr]), np.mean([float(r["hcp_percent"]) for r in rr])
            assert abs(sp - 100 * ref["pooled"]) < 1e-9 and abs(sh - 100 * ref["hcp"]) < 1e-9, (name, analysis, sp, sh, ref)
        inputs[str(STORED)] = sha(STORED)
        rows.append(dict(cell=name, candidate_images_in_cache=int(drop.sum()), candidate_tp_bearing=int((drop & (base > 0)).sum()),
                         point=point, point_delta_pp=dict(pooled=100 * (point["pooled_excl"] - point["pooled"]),
                                                          hcp=100 * (point["hcp_excl"] - point["hcp"])),
                         fixed_split=per_split[0],
                         fixed_delta_pp=dict(pooled=100 * (per_split[0]["pooled_excl"] - per_split[0]["pooled"]),
                                             hcp=100 * (per_split[0]["hcp_excl"] - per_split[0]["hcp"])),
                         max_abs_split_delta_pp=dict(pooled=max(abs(100 * (s["pooled_excl"] - s["pooled"])) for s in per_split),
                                                     hcp=max(abs(100 * (s["hcp_excl"] - s["hcp"])) for s in per_split)),
                         stored_point_reproduced=True, splits=per_split))
        print(name, json.dumps({k: rows[-1][k] for k in ("candidate_tp_bearing", "point", "point_delta_pp", "fixed_delta_pp", "max_abs_split_delta_pp")}), flush=True)
    receipt = dict(schema="rotcert-dior-candidate-exclusion-v1", alpha=0.10, splits=NSPLIT, candidate_images=sorted(cand),
                   rows=rows, inputs=inputs, script_sha256=sha(__file__),
                   interpretation="sensitivity only: candidates are annotation-geometry overlaps, not confirmed duplicates")
    OUT.write_text(json.dumps(receipt, indent=1, allow_nan=False) + "\n")
    print(OUT, sha(OUT))


if __name__ == "__main__":
    main()

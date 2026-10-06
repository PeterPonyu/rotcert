#!/usr/bin/env python3
"""DOTA crop-pipeline sensitivity, v3 (2026-10-04, after the second independent review): as v2, but the
evaluation-role ground-truth support counts every evaluation-role source of the common map, including sources without
a true positive in that condition (v2 reused the TP-bearing mask and missed them). Coverage values are unchanged.

v2 docstring follows. DOTA crop-pipeline sensitivity with one common source-ID -> role map (supersedes dota_retiling.py, whose per-condition
call to original_roles(names, universe, 0) re-drew the calibration/design/evaluation roles whenever the TP-bearing source
universe changed, so the three overlaps also differed in their split). Here every condition reuses the role map of the
archived 200-pixel cell (fixed split r = 0, seed 0); TP-bearing sources that have no 200-pixel role are excluded and
counted. The same retrained O-RCNN seed-0 detector, alpha = 0.10, Mondrian-by-class pooled and HCP thresholds from the
frozen S1 core. Re-tiling also changes which tile instances are detected and matched, so this is a crop-pipeline
sensitivity, not an isolated duplicate-object effect. The 200-pixel cell must reproduce the stored fixed-role class
coverages to 1e-9 percentage points. Read-only on frozen trees; writes only DOTA-RETILING-v2.json below
var/reviewer-gaps-20261004/dota-retiling. (2026-10-04.)"""
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
A = R / "revision_2026-10/closeout/analysis"
STORED = R / "manuscripts/revision-from-submission-20260930-v3/complete-data/S1-complete-class-results.csv"
OUT = R / "var/reviewer-gaps-20261004/dota-retiling"
sys.path.insert(0, str(S1))
import core                         # noqa: E402
from run_s1 import load_cell        # noqa: E402

OVERLAPS = (0, 200, 500)
REFERENCE = 200


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cell_for(overlap):
    cache = A / f"orcnn-s0-val{overlap}/orcnn-s0-val{overlap}.npz"
    return dict(name=f"orcnn-s0-val{overlap}", cache=str(cache), cache_sha256=sha(cache))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stored = {r["class"]: r for r in csv.DictReader(STORED.open())
              if r["analysis"] == "fixed-role" and r["cell"] == f"orcnn-s0-val{REFERENCE}"}
    loaded = {o: (cell_for(o),) + tuple(load_cell(cell_for(o))[:3]) for o in OVERLAPS}
    ref_cell, ref_c, _, ref_universe = loaded[REFERENCE]
    ref_names = [str(x) for x in ref_c["scene_names"]]
    ref_roles = core.original_roles(ref_c["scene_names"], ref_universe, 0)
    role_map = {ref_names[i]: int(ref_roles[i]) for i in ref_universe}
    classes = [str(x) for x in ref_c["class_names"]]
    rows, inputs = [], {}
    for overlap in OVERLAPS:
        cell, c, data, universe = loaded[overlap]
        inputs[cell["cache"]] = cell["cache_sha256"]
        names = [str(x) for x in c["scene_names"]]
        assert [str(x) for x in c["class_names"]] == classes
        roles = np.array([role_map.get(n, 0) for n in names], dtype=np.uint8)
        unmapped = [int(i) for i in universe if names[i] not in role_map]
        counts = np.zeros(len(names), dtype=np.int64)
        counts[[int(i) for i in universe if names[i] in role_map]] = 1
        # The confound removed here: roles re-drawn on this condition's own TP-bearing universe.
        redrawn = core.original_roles(c["scene_names"], universe, 0)
        union = sorted(set(role_map) | {names[i] for i in universe})
        own = {names[i]: int(redrawn[i]) for i in universe}
        changed = sum(own.get(n, 0) != role_map.get(n, 0) for n in union)
        changed_both = sum(own[n] != role_map[n] for n in union if n in own and n in role_map)
        q, metrics, support = core.one_inner(data, counts, roles, geometry=False)
        if overlap == REFERENCE:
            assert np.array_equal(roles, ref_roles)
            worst = max(max(abs(100 * metrics[0, k, 0] - float(stored[n]["pooling_percent"])),
                            abs(100 * metrics[1, k, 0] - float(stored[n]["hcp_percent"]))) for k, n in enumerate(classes))
            assert worst < 1e-9, worst
        # every source present in this condition carries exactly its 200-pixel role (0 if it had none)
        assert all(int(roles[i]) == role_map.get(names[i], 0) for i in range(len(names)))
        cov = metrics[:, :, 0]
        defined = ~np.isnan(cov).any(axis=0)
        keep = counts > 0
        tp_keep = keep[c["tp_scene"]]
        role_of_tp = roles[c["tp_scene"]]
        per_class = []
        for k, n in enumerate(classes):
            tp_k = (c["tp_cls"] == k)
            fn_k = (c["fn_cls"] == k)                      # every evaluation-role source, TP-bearing or not
            per_class.append(dict(
                cls=n, pooled=None if np.isnan(cov[0, k]) else float(cov[0, k]),
                hcp=None if np.isnan(cov[1, k]) else float(cov[1, k]),
                cal_sources=int(support[0, k, 0]), eval_sources=int(support[2, k, 0]),
                cal_tp=int((tp_k & tp_keep & (role_of_tp == core.CAL)).sum()),
                eval_tp=int((tp_k & tp_keep & (role_of_tp == core.EVAL)).sum()),
                eval_gt_tile_instances=int((tp_k & tp_keep & (role_of_tp == core.EVAL)).sum()
                                           + (fn_k & (roles[c["fn_scene"]] == core.EVAL)).sum())))
        role_sources = {r: int(((roles == v) & keep).sum()) for r, v in (("cal", core.CAL), ("design", core.DESIGN), ("eval", core.EVAL))}
        rows.append(dict(
            overlap_px=overlap, sources=len(names), tiles=int(c["n_tiles"].astype(int).sum()),
            tp_bearing_sources=int(len(universe)), sources_used=int(keep.sum()), role_sources=role_sources,
            excluded_unmapped_sources=[names[i] for i in unmapped],
            excluded_unmapped_tp_pairs=int(np.isin(c["tp_scene"], unmapped).sum()),
            matched_pairs=int(len(c["tp_scene"])), matched_pairs_used=int(tp_keep.sum()),
            role_changes_if_redrawn=int(changed), role_changes_if_redrawn_common_sources=int(changed_both),
            classes=len(classes), classes_defined=int(defined.sum()),
            pooled_macro=float(np.mean(cov[0, defined])), hcp_macro=float(np.mean(cov[1, defined])),
            pooled_min_class=float(np.min(cov[0, defined])), hcp_min_class=float(np.min(cov[1, defined])),
            hcp_classes_below_090=int((cov[1, defined] < 0.90).sum()),
            pooled_classes_below_090=int((cov[0, defined] < 0.90).sum()),
            per_class=per_class))
        print(json.dumps({k: v for k, v in rows[-1].items() if k != "per_class"}), flush=True)
    receipt = dict(
        schema="rotcert-dota-retiling-v3", supersedes="DOTA-RETILING-v2.json (GT support restricted to TP-bearing sources)",
        split=f"archived fixed split r=0; one source-ID -> role map taken from the {REFERENCE}-pixel cell for every condition",
        alpha=0.10, rows=rows, inputs=inputs,
        reproduction=f"overlap {REFERENCE} reproduces stored fixed-role class coverages to 1e-9 pp",
        role_consistency="asserted: every source in every condition carries its reference-cell role; unmapped TP-bearing sources excluded",
        script_sha256=sha(__file__),
        interpretation=("crop-pipeline sensitivity of object pooling with the split held fixed; re-tiling also changes detected "
                        "and matched tile instances, so no isolated duplicate-object effect is identified; descriptive"))
    path = OUT / "DOTA-RETILING-v3.json"
    if path.exists():
        raise SystemExit("receipt exists")
    path.write_text(json.dumps(receipt, indent=1, allow_nan=False) + "\n")
    print(path, sha(path))


if __name__ == "__main__":
    main()

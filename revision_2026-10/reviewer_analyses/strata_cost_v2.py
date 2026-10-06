#!/usr/bin/env python3
"""Descriptive strata and stage-level cost of the GWD certificate, second version (extends strata_cost.py).

Same frozen setting: four DIOR-R seed-0 detector families, archived fixed split (r = 0), alpha = 0.10, Mondrian-by-class
pooled and HCP thresholds from the frozen S1 core; the stored fixed-role class coverages are reproduced to 1e-9
percentage points before anything is written. Added: (i) orientation regime of the predicted box with the rule of the
orientation figure (near-square if short/long side >= 0.9, checked first; seam if within 5 degrees of the +-90 degree
long-edge seam; otherwise interior); (ii) scene density, quartiles over evaluation pairs of the number of annotated objects
(all classes) in the pair's scene; (iii) per detector and stratum, the number of evaluation pairs and of evaluation scenes,
and a 95% percentile interval from 2,000 resamples of evaluation scenes with the thresholds held fixed (conditional on the
calibration sample). Cost: single-core process time per stage on cached detections (warm, after loading); image decoding,
inference, matching and serialization are not timed. Read-only on frozen trees (no bytecode written); writes only
STRATA-COST-v2.json below var/reviewer-gaps-20261004/strata. (2026-10-04.)"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import sys
sys.dont_write_bytecode = True
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

R = Path(os.environ.get("ROTCERT_ROOT", Path(__file__).resolve().parents[2]))
S1 = R / "revision_2026-10/execution/s1"
STORED = R / "manuscripts/revision-from-submission-20260930-v3/complete-data/S1-complete-class-results.csv"
OUT = R / "var/reviewer-gaps-20261004/strata"
sys.path.insert(0, str(S1))
sys.path.insert(1, str(R))
import core                                   # noqa: E402
from run_s1 import CELLS, load_cell           # noqa: E402
from vendor.gwd import obb_gwd                # noqa: E402
from rotcert.audit import classify_theta_stratum   # noqa: E402  (rule of the orientation figure)

DIOR = ("dior-orcnn-s0", "dior-roit-s0", "dior-rtmdet-s0", "dior-s2anet-s0")
B, SEED = 2000, 20261006


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def quartile_masks(x, ev, labels):
    qs = np.quantile(x[ev], [.25, .5, .75])
    out = {}
    for j, lab in enumerate(labels):
        lo = -np.inf if j == 0 else qs[j - 1]
        hi = np.inf if j == 3 else qs[j]
        out[lab] = ev & (x > lo) & (x <= hi)
    return out, qs.tolist()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    stored = list(csv.DictReader(STORED.open()))
    cells = {c["name"]: c for c in CELLS}
    results, cost, inputs = {}, {}, {str(STORED): sha(STORED)}
    for name in DIOR:
        cell = cells[name]
        t = time.process_time()
        c, data, universe, _ = load_cell(cell)
        t_load = time.process_time() - t
        inputs[cell["cache"]] = cell["cache_sha256"]
        names = np.asarray(c["scene_names"]).astype(str)
        classes = [str(x) for x in c["class_names"]]
        roles = core.original_roles(c["scene_names"], universe, 0)
        counts = np.zeros(len(names), dtype=np.int64)
        counts[universe] = 1
        t = time.process_time()
        q, metrics, support = core.one_inner(data, counts, roles, geometry=True)
        t_inner = time.process_time() - t
        ref = {r["class"]: r for r in stored if r["analysis"] == "fixed-role" and r["cell"] == name}
        worst = max(max(abs(100 * metrics[0, k, 0] - float(ref[n]["pooling_percent"])),
                        abs(100 * metrics[1, k, 0] - float(ref[n]["hcp_percent"]))) for k, n in enumerate(classes))
        assert worst < 1e-9, (name, worst)
        p, g = c["tp_pred"].astype(float), c["tp_gt"].astype(float)
        t = time.process_time()
        score = obb_gwd(p, g)
        t_score = time.process_time() - t
        k_of = c["tp_cls"].astype(int)
        pw, ph = np.maximum(p[:, 2], p[:, 3]), np.minimum(p[:, 2], p[:, 3])
        q_pool, q_hcp = q[0, k_of], q[1, k_of]
        t = time.process_time()
        half = np.full(len(score), np.pi / 2)
        for k in range(len(classes)):
            m = k_of == k
            if m.any():
                half[m] = core.angle_halfwidth(pw[m], ph[m], q[1, k])
        t_proj = time.process_time() - t
        scene = c["tp_scene"].astype(int)
        ev = roles[scene] == core.EVAL
        cov_h, cov_p = score <= q_hcp, score <= q_pool
        nrad = q_hcp / np.sqrt(pw * ph)
        inform = half < np.pi / 2
        gw, gh = np.maximum(g[:, 2], g[:, 3]), np.minimum(g[:, 2], g[:, 3])
        size, ar, conf = np.sqrt(gw * gh), gw / gh, c["tp_conf"].astype(float)
        cal_sources = support[core.CAL - 1, :, 0]
        rank = np.argsort(np.argsort(cal_sources, kind="stable"), kind="stable")
        tertile = np.minimum(2, (3 * rank) // len(classes))
        regime = np.array([classify_theta_stratum(float(a), float(b), float(th)) for a, b, th in p[:, 2:5]])
        objects = np.bincount(scene, minlength=len(names)) + np.bincount(c["fn_scene"].astype(int), minlength=len(names))
        density = objects[scene].astype(float)
        masks = {}
        m, size_edges = quartile_masks(size, ev, ("size Q1 (smallest)", "size Q2", "size Q3", "size Q4 (largest)"))
        masks.update(m)
        for lab, lo, hi in (("aspect < 1.5", 0, 1.5), ("aspect 1.5-3", 1.5, 3), ("aspect >= 3", 3, np.inf)):
            masks[lab] = ev & (ar >= lo) & (ar < hi)
        for lab in ("interior", "boundary", "square"):
            masks["orientation " + lab] = ev & (regime == lab)
        m, dens_edges = quartile_masks(density, ev, ("density Q1 (sparsest)", "density Q2", "density Q3", "density Q4 (densest)"))
        masks.update(m)
        for j, lab in enumerate(("rare classes", "middle classes", "frequent classes")):
            masks[lab] = ev & (tertile[k_of] == j)
        m, conf_edges = quartile_masks(conf, ev, ("confidence Q1 (lowest)", "confidence Q2", "confidence Q3", "confidence Q4 (highest)"))
        masks.update(m)
        masks["all evaluation pairs"] = ev
        # paired resamples of evaluation scenes, thresholds fixed
        eval_scenes = np.flatnonzero((roles == core.EVAL) & (counts > 0))
        pos = np.full(len(names), -1)
        pos[eval_scenes] = np.arange(len(eval_scenes))
        W = rng.multinomial(len(eval_scenes), np.full(len(eval_scenes), 1 / len(eval_scenes)), size=B).astype(float)
        strata = {}
        for lab, mk in masks.items():
            n = int(mk.sum())
            if not n:
                strata[lab] = dict(n=0)
                continue
            sc = pos[scene[mk]]
            assert (sc >= 0).all()
            cnt = np.bincount(sc, minlength=len(eval_scenes)).astype(float)
            hit_h = np.bincount(sc, weights=cov_h[mk].astype(float), minlength=len(eval_scenes))
            hit_p = np.bincount(sc, weights=cov_p[mk].astype(float), minlength=len(eval_scenes))
            den = W @ cnt
            with np.errstate(invalid="ignore", divide="ignore"):
                bh, bp = (W @ hit_h) / den, (W @ hit_p) / den
            ok = den > 0
            strata[lab] = dict(n=n, sources=int((cnt > 0).sum()), hcp=float(cov_h[mk].mean()), pooled=float(cov_p[mk].mean()),
                               hcp_interval=[float(np.quantile(bh[ok], .025)), float(np.quantile(bh[ok], .975))],
                               pooled_interval=[float(np.quantile(bp[ok], .025)), float(np.quantile(bp[ok], .975))],
                               resamples_undefined=int((~ok).sum()),
                               radius_median=float(np.median(nrad[mk])), informative=float(inform[mk].mean()))
        results[name] = dict(strata=strata, reproduction_max_abs_diff_pp=worst, eval_scenes=int(len(eval_scenes)),
                             class_tertiles={classes[k]: int(tertile[k]) for k in range(len(classes))},
                             edges=dict(size=size_edges, density=dens_edges, confidence=conf_edges))
        cost[name] = dict(pairs=int(len(score)), scenes=int(len(names)), load_and_score_s=t_load, score_only_s=t_score,
                          calibrate_evaluate_project_s=t_inner, projections_all_pairs_s=t_proj,
                          scope="warm, cached detections; per stage; excludes decoding, inference, matching and serialization")
        print(name, json.dumps({k: (v["n"], v.get("sources"), round(100 * v.get("hcp", float("nan")), 2)) for k, v in strata.items()}),
              flush=True)
    receipt = dict(schema="rotcert-strata-cost-v2", supersedes="STRATA-COST.json (v1: no orientation/density strata, no support or intervals)",
                   split="archived fixed split r=0", alpha=0.10, bootstrap=dict(B=B, seed=SEED, unit="evaluation scene", thresholds="fixed"),
                   orientation_rule="predicted box; near-square if short/long >= 0.9 (checked first); seam if within 5 deg of +-90 deg; else interior",
                   results=results, cost=cost, inputs=inputs, script_sha256=sha(__file__), host=dict(cpu=os.uname().machine, threads=1),
                   interpretation="descriptive conditional coverage within strata; the certificate is marginal and source-uniform")
    path = OUT / "STRATA-COST-v2.json"
    path.write_text(json.dumps(receipt, indent=1) + "\n")
    print(path, sha(path))


if __name__ == "__main__":
    main()

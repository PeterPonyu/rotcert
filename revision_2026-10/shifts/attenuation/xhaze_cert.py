#!/usr/bin/env python3
"""X-HAZE certification analysis (local CPU; 2026-10-03).

Follows the frozen expansion design (orchestration/expansion_20261001/design.json, "dior-haze"):
  - fixed seed-0 checkpoints, no retraining (seeds 1 and 2 were added afterwards as replication);
  - the frozen accepted clean DIOR-R calibration/evaluation source IDs (sources.jsonl, 5869/5869),
    never repartitioned after haze outcomes;
  - comparisons: clean-cal -> clean-eval, clean-cal -> shifted-eval, same-beta-cal -> same-beta-eval.
Certificate = the paper's own frozen S1 core (Mondrian-by-class pooled and scene-weighted HCP at
alpha = 1/10, GWD score from the S1 vendor copy); matching = the frozen local matcher (IoU >= 0.5,
score >= 0.05 export, greedy by score) including its reference-matcher parity check.
Uncertainty = paired source bootstrap within the fixed roles (descriptive 95% percentile ranges);
a range is reported only if every draw is defined (no draw is discarded), as in the paper.

Stages:  selftest  (re-match one clean cell, must reproduce the frozen matched.jsonl byte for byte)
         match     (hazed detections -> matched.jsonl + cache + receipt)
         analyze   (point estimates + bootstrap per family/seed)
Writes only below var/planc-results-20261003/analysis/xhaze. Frozen trees are only read.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
sys.dont_write_bytecode = True

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROT = Path(os.environ.get("ROTCERT_ROOT", Path(__file__).resolve().parents[3]))
S1 = ROT / "revision_2026-10/execution/s1"
CLOSE = ROT / "revision_2026-10/closeout"
SOURCES = ROT / "revision_2026-10/inputs/bundle/sources.jsonl"
SOURCES_SHA = "3fe7b4843decae8641ab6ec751621aa9a5ed49043c9d6a16c1b699b701fe24b8"
GT = CLOSE / "collected-prefix/run/jobs/dior-orcnn-s0/ground_truth.jsonl"
GT_SHA = "6a2970475bc159f15407fc04e9569a061bf1f43a345187a4582a5b4cd7037185"
PULL = ROT / "var/planc-results-20261003/haze"
OUT = ROT / "var/planc-results-20261003/analysis/xhaze"
FROZEN = {  # frozen inputs, checked at start
    "s1/core.py": (S1 / "core.py", None),
    "s1/vendor/gwd.py": (S1 / "vendor/gwd.py", None),
    "match_local.py": (CLOSE / "match_local.py", "0f16cd5a03de49f401cb8efd08c9d4c2b6a45f84867394db21f6cb0403e1361b"),
}
FAMILIES = ("orcnn", "roit", "rtmdet", "s2anet")
BETAS = ("b06", "b12", "b18")
BETA_VALUE = {"b06": 0.6, "b12": 1.2, "b18": 1.8}
CAL, EVAL = 1, 3
B_DEFAULT = 2000
RNG_ROOT = 20261003


def sha(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


core = load_module("s1_core", S1 / "core.py")
gwd = load_module("s1_gwd", S1 / "vendor/gwd.py")
_matcher = None


def matcher():
    global _matcher
    if _matcher is None:
        _matcher = load_module("frozen_match_local", CLOSE / "match_local.py")
    return _matcher


def frozen_hashes() -> dict:
    out = {}
    for key, (path, expected) in FROZEN.items():
        h = sha(path)
        if expected and h != expected:
            raise RuntimeError(f"frozen input changed: {key}")
        out[key] = h
    if sha(SOURCES) != SOURCES_SHA or sha(GT) != GT_SHA:
        raise RuntimeError("frozen sources/GT changed")
    out["sources.jsonl"], out["ground_truth.jsonl"] = SOURCES_SHA, GT_SHA
    out["xhaze_cert.py"] = sha(__file__)
    return out


# --------------------------------------------------------------------------- matching
def match_to(det_path: Path, out: Path) -> dict:
    """Exact replica of match_local.process() with explicit paths (process() writes into the frozen tree)."""
    m = matcher()
    out.mkdir(parents=True, exist_ok=True)
    t = time.monotonic()
    dets, gts = m.load_group(det_path, "det"), m.load_group(GT, "gt")
    images = sorted(set(dets) | set(gts))
    candidates = [i for i in images if dets.get(i) and gts.get(i)]
    picks = sorted(set(candidates[::max(1, len(candidates) // 12)][:12]
                       + sorted(candidates, key=lambda i: abs(len(gts[i]) - 12))[:3]))
    for i in picks:
        m.assert_equal(dets[i], gts[i])
    counts = collections.Counter()
    by_class = collections.defaultdict(collections.Counter)
    tmp = out / "matched.jsonl.partial"
    with tmp.open("w") as f:
        for image in images:
            d, g = dets.get(image, []), gts.get(image, [])
            scene = (d or g)[0]["scene_id"]
            assert all(x["scene_id"] == scene for x in d + g)
            match = m.accelerated(d, g)
            for mm in match["matches"]:
                di, gi = mm["det_index"], mm["gt_index"]
                row = {"image_id": image, "scene_id": scene, "class": d[di]["class"], "pred_obb": d[di]["obb"],
                       "gt_obb": g[gi]["obb"], "pred_score": d[di]["score"], "iou": mm["iou"], "match_type": "tp"}
                f.write(json.dumps(row) + "\n"); counts["tp"] += 1; by_class[row["class"]]["tp"] += 1
            for di in match["unmatched_det_indices"]:
                row = {"image_id": image, "scene_id": scene, "class": d[di]["class"], "pred_obb": d[di]["obb"],
                       "gt_obb": None, "pred_score": d[di]["score"], "iou": None, "match_type": "fp"}
                f.write(json.dumps(row) + "\n"); counts["fp"] += 1; by_class[row["class"]]["fp"] += 1
            for gi in match["unmatched_gt_indices"]:
                row = {"image_id": image, "scene_id": scene, "class": g[gi]["class"], "pred_obb": None,
                       "gt_obb": g[gi]["obb"], "pred_score": None, "iou": None, "match_type": "fn"}
                f.write(json.dumps(row) + "\n"); counts["fn"] += 1; by_class[row["class"]]["fn"] += 1
    assert counts["tp"] + counts["fp"] == sum(map(len, dets.values()))
    assert counts["tp"] + counts["fn"] == sum(map(len, gts.values()))
    matched = out / "matched.jsonl"
    tmp.replace(matched)
    cache = out / "cache.npz"
    m.build(str(matched), str(cache), scene_rule="image")
    return {"counts": dict(counts), "by_class": {k: dict(v) for k, v in by_class.items()},
            "recall": counts["tp"] / (counts["tp"] + counts["fn"]), "reference_matcher_parity_images": len(picks),
            "matched_sha256": sha(matched), "cache_sha256": sha(cache), "detections_sha256": sha(det_path),
            "ground_truth_sha256": sha(GT), "runtime_seconds": round(time.monotonic() - t, 1)}


def match_job(job: str) -> dict:
    src = PULL / job
    if not (src / ".verified").is_file():
        raise RuntimeError(f"{job}: not byte-verified")
    rec = json.loads((src / "HAZE-RECORD.json").read_text())
    out = OUT / "matched" / job
    receipt = out / "match_receipt.json"
    if receipt.is_file():
        r = json.loads(receipt.read_text())
        assert sha(out / "matched.jsonl") == r["matched_sha256"] and sha(out / "cache.npz") == r["cache_sha256"]
        return r
    r = match_to(src / "detections.jsonl", out)
    assert r["detections_sha256"] == rec["detections_sha256"]
    r.update(job=job, base_job=rec["haze"]["base_job"], beta=rec["haze"]["beta"],
             checkpoint_sha256=rec["haze"]["checkpoint_sha256"], frozen=frozen_hashes())
    receipt.write_text(json.dumps(r, indent=2) + "\n")
    return r


def selftest() -> dict:
    """Re-match the clean O-RCNN s0 detections; the frozen matched.jsonl must be reproduced byte for byte."""
    out = OUT / "selftest" / "dior-orcnn-s0"
    r = match_to(CLOSE / "collected-prefix/run/jobs/dior-orcnn-s0/detections.jsonl", out)
    frozen = json.loads((CLOSE / "analysis/dior-orcnn-s0/match_receipt.json").read_text())
    ok = r["matched_sha256"] == frozen["matched_sha256"]
    z1, z2 = np.load(out / "cache.npz"), np.load(CLOSE / "analysis/dior-orcnn-s0/dior-orcnn-s0.npz")
    arrays_equal = all(np.array_equal(z1[k], z2[k]) for k in z2.files if k != "source_sha256")
    res = {"matched_identical": ok, "cache_arrays_identical": bool(arrays_equal), "receipt": r}
    (OUT / "selftest" / "SELFTEST.json").write_text(json.dumps(res, indent=2) + "\n")
    if not (ok and arrays_equal):
        raise RuntimeError("selftest failed: replica matcher does not reproduce the frozen clean cell")
    return res


# --------------------------------------------------------------------------- analysis
def roles_and_names():
    names, roles = [], []
    for line in SOURCES.open():
        r = json.loads(line)
        names.append(str(r["source_id"]))
        roles.append({"calibration": CAL, "evaluation": EVAL}[r["role"]])
    order = np.argsort(np.array(names), kind="stable")
    names = np.array(names)[order]
    roles = np.array(roles, dtype=np.uint8)[order]
    assert len(set(names)) == len(names) == 11738
    assert (roles == CAL).sum() == 5869 and (roles == EVAL).sum() == 5869
    return names, roles


class Condition:
    """One matched cell on the global source axis."""

    def __init__(self, cache: Path, names: np.ndarray, class_names=None):
        z = np.load(cache, allow_pickle=False)
        cn = [str(c) for c in z["class_names"]]
        if class_names is not None and cn != list(class_names):
            raise ValueError(f"class dictionary differs: {cache}")
        self.class_names = cn
        pos = {n: i for i, n in enumerate(names)}
        sidx = np.array([pos[str(s)] for s in z["scene_names"]], dtype=np.int64)
        ns, nk = len(names), len(cn)
        self.tp_scene, self.tp_cls = sidx[z["tp_scene"]], z["tp_cls"].astype(np.int64)
        self.tp_pred, self.tp_gt = z["tp_pred"], z["tp_gt"]
        self.scores = gwd.obb_gwd(self.tp_pred, self.tp_gt)
        self.data = core.prepare(self.scores, self.tp_scene, self.tp_cls, self.tp_pred, self.tp_gt, ns, nk)
        fn_scene, fn_cls = sidx[z["fn_scene"]], z["fn_cls"].astype(np.int64)
        self.tp_sk = np.zeros((ns, nk), np.int64)
        np.add.at(self.tp_sk, (self.tp_scene, self.tp_cls), 1)
        self.fn_sk = np.zeros((ns, nk), np.int64)
        np.add.at(self.fn_sk, (fn_scene, fn_cls), 1)
        self.gt_sk = self.tp_sk + self.fn_sk
        self.fp_total = int(len(z["fp_scene"]))

    def covered_sk(self, q: np.ndarray) -> np.ndarray:
        """Count of TPs per (source, class) whose score is within the class threshold."""
        cov = self.scores <= q[self.tp_cls]
        out = np.zeros_like(self.tp_sk)
        np.add.at(out, (self.tp_scene[cov], self.tp_cls[cov]), 1)
        return out


def evaluate(data, counts, roles, q, geometry=True):
    """The evaluation half of core.one_inner with externally supplied thresholds q[2, K] (verbatim arithmetic)."""
    nk = len(data)
    metrics = np.full((2, nk, len(core.METRICS) if geometry else 1), np.nan)
    for k, d in enumerate(data):
        ew = counts * (roles == EVAL)
        present = (ew > 0) & (d.sizes > 0)
        if not present.any():
            continue
        mask = ew[d.sources] > 0
        sc = d.sources[mask]
        objweights = ew[sc] / d.sizes[sc]
        denominator = float(ew[present].sum())
        for method in range(2):
            qq = q[method, k]
            metrics[method, k, 0] = np.dot(objweights, d.scores[mask] <= qq) / denominator
            if geometry:
                half = core.angle_halfwidth(d.widths[mask], d.heights[mask], qq)
                angle_covered = np.where(d.square_gt[mask], half == np.pi / 2, d.angle_errors[mask] <= half)
                joint = (d.center_errors[mask] <= qq) & angle_covered
                values = (math.isinf(qq), qq / d.scales[mask], half, half < np.pi / 2, joint)
                for j, v in enumerate(values, 1):
                    metrics[method, k, j] = float(v) if np.isscalar(v) else np.dot(objweights, v) / denominator
    return metrics


def gt_level(cond: Condition, counts, roles, q):
    """Scene-weighted (source-uniform) all-GT miss rate and end-to-end risk per class, eval role.
    Returns fnr[K], e2e[2, K] (pooled, HCP thresholds) and object-level recall."""
    ew = (counts * (roles == EVAL)).astype(float)
    pos = cond.gt_sk > 0
    w = ew[:, None] * pos
    den = w.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac_fn = np.where(pos, cond.fn_sk / np.maximum(cond.gt_sk, 1), 0.0)
        fnr = np.where(den > 0, (w * frac_fn).sum(0) / den, np.nan)
        e2e = np.full((2, cond.gt_sk.shape[1]), np.nan)
        for method in range(2):
            cov = cond.covered_sk(q[method])
            frac_ok = np.where(pos, cov / np.maximum(cond.gt_sk, 1), 0.0)
            e2e[method] = np.where(den > 0, 1.0 - (w * frac_ok).sum(0) / den, np.nan)
    tp = (ew[:, None] * cond.tp_sk).sum()
    gt = (ew[:, None] * cond.gt_sk).sum()
    return fnr, e2e, tp / gt


def one_draw(conds: dict, counts, roles):
    """All readouts for one multiplicity vector; conds = {'clean': C, 'b06': C, ...}."""
    rec = {}
    q_clean, m_clean, _ = core.one_inner(conds["clean"].data, counts, roles, geometry=True)
    fnr, e2e, rec_obj = gt_level(conds["clean"], counts, roles, q_clean)
    rec["clean|clean"] = dict(q=q_clean, m=m_clean, fnr=fnr, e2e=e2e, recall=rec_obj)
    for b in BETAS:
        if b not in conds:
            continue
        c = conds[b]
        q_b, m_b, _ = core.one_inner(c.data, counts, roles, geometry=True)
        fnr_b, e2e_b, rec_b = gt_level(c, counts, roles, q_b)
        rec[f"{b}|{b}"] = dict(q=q_b, m=m_b, fnr=fnr_b, e2e=e2e_b, recall=rec_b)
        m_x = evaluate(c.data, counts, roles, q_clean, geometry=True)
        _, e2e_x, _ = gt_level(c, counts, roles, q_clean)
        rec[f"clean|{b}"] = dict(q=q_clean, m=m_x, fnr=fnr_b, e2e=e2e_x, recall=rec_b)
    return rec


MACROS = ("coverage", "infinite_fraction", "normalized_radius", "angle_halfwidth",
          "informative_angle_fraction", "joint_center_angle")


def flatten(rec: dict) -> dict:
    """Scalar readouts per (calibration|evaluation) arm, method in (pooled, hcp). Strict class means."""
    out = {}
    for arm, r in rec.items():
        for mi, method in enumerate(("pooled", "hcp")):
            for j, name in enumerate(MACROS):
                out[f"{arm}|{method}|{name}"] = float(np.mean(r["m"][mi, :, j]))
            out[f"{arm}|{method}|e2e_risk"] = float(np.mean(r["e2e"][mi]))
            out[f"{arm}|{method}|ship_coverage"] = float(r["m"][mi, SHIP, 0])
            out[f"{arm}|{method}|q_ship_px"] = float(r["q"][mi, SHIP])
        out[f"{arm}|fnr"] = float(np.mean(r["fnr"]))
        out[f"{arm}|recall_object"] = float(r["recall"])
    return out


SHIP = None


def cell_conditions(family: str, seed: int, names):
    global SHIP
    base = f"dior-{family}-s{seed}"
    clean_cache = CLOSE / "analysis" / base / f"{base}.npz"
    frozen = json.loads((CLOSE / "analysis" / base / "match_receipt.json").read_text())
    if sha(clean_cache) != frozen["cache_sha256"] or frozen["ground_truth_sha256"] != GT_SHA:
        raise RuntimeError(f"{base}: clean cache does not match its frozen receipt")
    conds = {"clean": Condition(clean_cache, names)}
    cn = conds["clean"].class_names
    SHIP = cn.index("ship")
    inputs = {"clean": {"cache": str(clean_cache), "cache_sha256": frozen["cache_sha256"],
                        "detections_sha256": frozen["detections_sha256"]}}
    clean_ckpt = json.loads((CLOSE / "collected-prefix/run/jobs" / base / "run.json").read_text())["checkpoint_sha256"]
    for b in BETAS:
        job = f"{base}-haze-{b}"
        rdir = OUT / "matched" / job
        if not (rdir / "match_receipt.json").is_file():
            continue
        r = json.loads((rdir / "match_receipt.json").read_text())
        if sha(rdir / "cache.npz") != r["cache_sha256"]:
            raise RuntimeError(f"{job}: cache hash mismatch")
        if r["checkpoint_sha256"] != clean_ckpt:
            raise RuntimeError(f"{job}: haze checkpoint differs from the clean base job")
        conds[b] = Condition(rdir / "cache.npz", names, cn)
        inputs[b] = {"cache_sha256": r["cache_sha256"], "detections_sha256": r["detections_sha256"],
                     "checkpoint_sha256": r["checkpoint_sha256"]}
    return conds, inputs, clean_ckpt


def resample(rng, roles):
    counts = np.zeros(len(roles), dtype=np.int64)
    for role in (CAL, EVAL):
        ids = np.flatnonzero(roles == role)
        counts += np.bincount(rng.choice(ids, len(ids), replace=True), minlength=len(roles))
    return counts


def analyze_task(args):
    family, seed, start, stop, B = args
    names, roles = roles_and_names()
    conds, _, _ = cell_conditions(family, seed, names)
    rng = np.random.default_rng(np.random.SeedSequence([RNG_ROOT, FAMILIES.index(family), seed]))
    rows = []
    for b in range(B):
        counts = resample(rng, roles)     # the stream is consumed in full order so chunks are reproducible
        if b < start or b >= stop:
            continue
        rows.append(flatten(one_draw(conds, counts, roles)))
    return family, seed, start, rows


def composition(conds: dict, roles, q_clean_hcp):
    """Point-estimate decomposition on GT objects detected in both conditions (eval role, HCP clean threshold)."""
    out = {}
    c0 = conds["clean"]
    key0 = {(int(s), int(k), *np.round(g, 4)): i for i, (s, k, g) in enumerate(zip(c0.tp_scene, c0.tp_cls, c0.tp_gt))}
    ev0 = roles[c0.tp_scene] == EVAL
    for b in BETAS:
        if b not in conds:
            continue
        c = conds[b]
        shared_clean, shared_haze = [], []
        for i, (s, k, g) in enumerate(zip(c.tp_scene, c.tp_cls, c.tp_gt)):
            j = key0.get((int(s), int(k), *np.round(g, 4)))
            if j is not None and roles[s] == EVAL:
                shared_clean.append(j); shared_haze.append(i)
        shared_clean, shared_haze = np.array(shared_clean), np.array(shared_haze)
        lost = np.setdiff1d(np.flatnonzero(ev0), shared_clean)
        q0 = q_clean_hcp[c0.tp_cls]
        out[b] = {
            "eval_TP_clean": int(ev0.sum()), "eval_TP_haze": int((roles[c.tp_scene] == EVAL).sum()),
            "shared_TP": int(len(shared_clean)), "lost_TP": int(len(lost)),
            "object_cov_shared_clean_pred": float(np.mean(c0.scores[shared_clean] <= q0[shared_clean])),
            "object_cov_shared_haze_pred": float(np.mean(c.scores[shared_haze] <= q_clean_hcp[c.tp_cls[shared_haze]])),
            "object_cov_lost_clean_pred": float(np.mean(c0.scores[lost] <= q0[lost])) if len(lost) else None,
            "median_gwd_shared_clean": float(np.median(c0.scores[shared_clean])),
            "median_gwd_shared_haze": float(np.median(c.scores[shared_haze])),
        }
    return out


def ext_percentile(col, p):
    """Linear percentile on the extended reals: +inf draws are kept, never filtered (inf-inf -> inf)."""
    x = np.sort(col)
    h = (len(x) - 1) * p / 100.0
    lo, hi = x[int(math.floor(h))], x[int(math.ceil(h))]
    return float(lo) if lo == hi else float(lo + (hi - lo) * (h - math.floor(h)))


def summarize(draws, point, B):
    keys = sorted(point)
    arr = np.array([[d[k] for k in keys] for d in draws])
    res = {}
    for j, k in enumerate(keys):
        col = arr[:, j]
        complete = bool(np.all(~np.isnan(col)))
        entry = {"point": point[k], "draws": int(len(col)), "nan_draws": int(np.isnan(col).sum()),
                 "inf_draws": int(np.isinf(col).sum())}
        if complete and len(col) == B:
            entry.update(lower=ext_percentile(col, 2.5), upper=ext_percentile(col, 97.5))
        else:
            entry.update(lower=None, upper=None, status="UNAVAILABLE_UNDEFINED_DRAWS")
        res[k] = entry
    # paired differences against clean|clean
    for arm in [a for a in {k.split("|")[0] + "|" + k.split("|")[1] for k in keys} if a != "clean|clean"]:
        for method in ("pooled", "hcp"):
            for name in ("coverage", "e2e_risk"):
                a, c = f"{arm}|{method}|{name}", f"clean|clean|{method}|{name}"
                if a not in point:
                    continue
                d = arr[:, keys.index(a)] - arr[:, keys.index(c)]
                e = {"point": point[a] - point[c]}
                if not np.isnan(d).any() and len(d) == B:
                    e.update(lower=ext_percentile(d, 2.5), upper=ext_percentile(d, 97.5))
                else:
                    e.update(lower=None, upper=None, status="UNAVAILABLE_UNDEFINED_DRAWS")
                res[f"diff|{a}"] = e
    return res


def run_analysis(workers: int, B: int, cells):
    names, roles = roles_and_names()
    meta = {"frozen": frozen_hashes(), "B": B, "rng_root": RNG_ROOT, "alpha": "1/10",
            "split": "frozen accepted clean cal5869/eval5869 (sources.jsonl)",
            "bootstrap": "paired source resampling within each fixed role; descriptive 2.5/97.5 percentiles; "
                         "a range is withheld if any draw is undefined", "cells": {}}
    tasks, points = [], {}
    for family, seed in cells:
        conds, inputs, ckpt = cell_conditions(family, seed, names)
        if any(b not in conds for b in BETAS):
            print(f"skip {family}-s{seed}: missing", [b for b in BETAS if b not in conds], flush=True)
            continue
        ones = np.ones(len(names), dtype=np.int64)
        rec = one_draw(conds, ones, roles)
        # validity gate: evaluate() must reproduce one_inner exactly on the same-condition arm
        m_check = evaluate(conds["clean"].data, ones, roles, rec["clean|clean"]["q"], geometry=True)
        if not np.array_equal(m_check, rec["clean|clean"]["m"], equal_nan=True):
            raise RuntimeError("evaluate() does not reproduce core.one_inner")
        points[(family, seed)] = flatten(rec)
        comp = composition(conds, roles, rec["clean|clean"]["q"][1])
        classes = {}
        for arm, r in rec.items():
            classes[arm] = {cn: {"q_pooled": float(r["q"][0, k]), "q_hcp": float(r["q"][1, k]),
                                 "cov_pooled": float(r["m"][0, k, 0]), "cov_hcp": float(r["m"][1, k, 0]),
                                 "fnr": float(r["fnr"][k])}
                            for k, cn in enumerate(conds["clean"].class_names)}
        meta["cells"][f"{family}-s{seed}"] = {"inputs": inputs, "checkpoint_sha256": ckpt,
                                              "composition": comp, "class_point": classes}
        chunk = max(1, B // 4)
        tasks += [(family, seed, s, min(B, s + chunk), B) for s in range(0, B, chunk)]
    results = collections.defaultdict(list)
    with ProcessPoolExecutor(workers) as ex:
        for family, seed, start, rows in ex.map(analyze_task, tasks):
            results[(family, seed)].append((start, rows))
            print(f"done {family}-s{seed} draws {start}..{start + len(rows) - 1}", flush=True)
    summary = {}
    for key, parts in results.items():
        draws = [r for _, rows in sorted(parts) for r in rows]
        assert len(draws) == B
        summary[f"{key[0]}-s{key[1]}"] = summarize(draws, points[key], B)
        np.savez_compressed(OUT / f"draws-{key[0]}-s{key[1]}.npz",
                            keys=np.array(sorted(points[key])),
                            values=np.array([[d[k] for k in sorted(points[key])] for d in draws]))
    meta["summary"] = summary
    (OUT / "XHAZE-SUMMARY.json").write_text(json.dumps(meta, indent=1, allow_nan=True) + "\n")
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["selftest", "match", "analyze"])
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--B", type=int, default=B_DEFAULT)
    p.add_argument("--cells", nargs="*", default=None, help="family-seed, e.g. roit-s1")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "selftest":
        print(json.dumps({k: v for k, v in selftest().items() if k != "receipt"}))
    elif a.stage == "match":
        frozen_hashes()
        jobs = sorted(d.name for d in PULL.iterdir() if (d / ".verified").is_file())
        with ProcessPoolExecutor(a.workers) as ex:
            for r in ex.map(match_job, jobs):
                print(r["job"], r["counts"], round(r["recall"], 4), flush=True)
    else:
        cells = [(c.rsplit("-s", 1)[0], int(c.rsplit("-s", 1)[1])) for c in a.cells] if a.cells else \
            [(f, s) for f in FAMILIES for s in (0, 1, 2)]
        run_analysis(a.workers, a.B, cells)


if __name__ == "__main__":
    main()

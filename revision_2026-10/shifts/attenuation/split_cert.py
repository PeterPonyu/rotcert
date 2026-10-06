#!/usr/bin/env python3
"""Official-split certification readout for datasets without audited source scenes (2026-10-03).

Calibration = the official val split, evaluation = the official test split (frozen expansion design for DroneVehicle).
Each image is its own source unit: these datasets carry no audited scene/flight identity (the export writes scene_id =
null), so the readout is image-level and descriptive, NOT a source-scene certificate; within-split frame dependence makes
the bootstrap ranges optimistic. Matching, certificate, metrics and bootstrap are exactly those of xhaze_cert.py
(frozen local matcher, frozen S1 core, GWD, alpha = 1/10, paired within-role source resampling).

  dv:   one condition per modality (rgb, ir) and seed; image IDs are shared by an RGB/IR pair, so one resample serves both
        modalities (paired contrast) while each modality keeps its own ground truth.
  rsar: one condition per detector family (code-path check only; RSAR stays uncertified until its source units exist).
Writes only below var/planc-results-20261003/analysis/<dataset>.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
sys.dont_write_bytecode = True

import argparse
import collections
import importlib.util
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("xhaze_cert", HERE / "xhaze_cert.py")
X = importlib.util.module_from_spec(spec)
spec.loader.exec_module(X)

ROT = X.ROT
CLOSE = X.CLOSE
OUTROOT = ROT / "var/planc-results-20261003/analysis"
DV_PULL = ROT / "var/planc-results-20261003/dv"
CAL, EVAL = X.CAL, X.EVAL


def cells(dataset):
    if dataset == "dv":
        out = {}
        for seed in (0, 1):
            for mod in ("rgb", "ir"):
                base = f"dv-orcnn-{mod}-s{seed}"
                out[f"{mod}-s{seed}"] = (DV_PULL / f"{base}-val", DV_PULL / base)
        return out
    return {fam: (CLOSE / f"collected-prefix/run/jobs/rsar-{fam}-s0-val", CLOSE / f"collected-prefix/run/jobs/rsar-{fam}-s0-test")
            for fam in ("orcnn", "roit", "s2anet")}


def load_rows(path: Path, kind: str, prefix: str):
    m = X.matcher()
    validate = m.rio.validate_detection if kind == "det" else m.rio.validate_gt
    out = collections.defaultdict(list)
    for i, line in enumerate(path.open()):
        row = validate(json.loads(line), tag=f"{kind}[{i}]")
        if row.get("scene_id"):
            raise ValueError("this readout is for exports without audited scenes")
        row["scene_id"] = prefix + str(row["image_id"])
        out[row["image_id"]].append(row)
    return out


def match_split(job_dir: Path, prefix: str, out: Path) -> dict:
    """match_local.process() logic with image-level scene IDs (prefix keeps val and test IDs apart)."""
    m = X.matcher()
    receipt = out / "match_receipt.json"
    if receipt.is_file():
        r = json.loads(receipt.read_text())
        assert X.sha(out / "matched.jsonl") == r["matched_sha256"] and X.sha(out / "cache.npz") == r["cache_sha256"]
        return r
    out.mkdir(parents=True, exist_ok=True)
    t = time.monotonic()
    dets, gts = load_rows(job_dir / "detections.jsonl", "det", prefix), load_rows(job_dir / "ground_truth.jsonl", "gt", prefix)
    images = sorted(set(dets) | set(gts))
    candidates = [i for i in images if dets.get(i) and gts.get(i)]
    picks = sorted(set(candidates[::max(1, len(candidates) // 12)][:12]
                       + sorted(candidates, key=lambda i: abs(len(gts[i]) - 12))[:3]))
    for i in picks:
        m.assert_equal(dets[i], gts[i])
    counts = collections.Counter()
    tmp = out / "matched.jsonl.partial"
    with tmp.open("w") as f:
        for image in images:
            d, g = dets.get(image, []), gts.get(image, [])
            scene = prefix + str(image)
            match = m.accelerated(d, g)
            for mm in match["matches"]:
                di, gi = mm["det_index"], mm["gt_index"]
                f.write(json.dumps({"image_id": image, "scene_id": scene, "class": d[di]["class"], "pred_obb": d[di]["obb"],
                                    "gt_obb": g[gi]["obb"], "pred_score": d[di]["score"], "iou": mm["iou"],
                                    "match_type": "tp"}) + "\n"); counts["tp"] += 1
            for di in match["unmatched_det_indices"]:
                f.write(json.dumps({"image_id": image, "scene_id": scene, "class": d[di]["class"], "pred_obb": d[di]["obb"],
                                    "gt_obb": None, "pred_score": d[di]["score"], "iou": None, "match_type": "fp"}) + "\n")
                counts["fp"] += 1
            for gi in match["unmatched_gt_indices"]:
                f.write(json.dumps({"image_id": image, "scene_id": scene, "class": g[gi]["class"], "pred_obb": None,
                                    "gt_obb": g[gi]["obb"], "pred_score": None, "iou": None, "match_type": "fn"}) + "\n")
                counts["fn"] += 1
    assert counts["tp"] + counts["fp"] == sum(map(len, dets.values()))
    assert counts["tp"] + counts["fn"] == sum(map(len, gts.values()))
    tmp.replace(out / "matched.jsonl")
    m.build(str(out / "matched.jsonl"), str(out / "cache.npz"), scene_rule="image")
    r = {"job_dir": str(job_dir), "counts": dict(counts), "recall": counts["tp"] / (counts["tp"] + counts["fn"]),
         "images": len(images), "reference_matcher_parity_images": len(picks),
         "matched_sha256": X.sha(out / "matched.jsonl"), "cache_sha256": X.sha(out / "cache.npz"),
         "detections_sha256": X.sha(job_dir / "detections.jsonl"), "ground_truth_sha256": X.sha(job_dir / "ground_truth.jsonl"),
         "runtime_seconds": round(time.monotonic() - t, 1)}
    receipt.write_text(json.dumps(r, indent=2) + "\n")
    return r


def axis_for(caches):
    """Union of source names over all caches of a group; role from the val/ or test/ prefix."""
    names = set()
    for c in caches:
        names |= {str(s) for s in np.load(c, allow_pickle=False)["scene_names"]}
    names = np.array(sorted(names))
    roles = np.array([CAL if n.startswith("val/") else EVAL for n in names], dtype=np.uint8)
    assert all(n.startswith(("val/", "test/")) for n in names)
    return names, roles


def build_group(dataset, out, keys):
    """One shared source axis per group (DV: the two modalities of one seed; RSAR: one family)."""
    caches = {}
    for key in keys:
        val_dir, test_dir = cells(dataset)[key]
        caches[key] = []
        for prefix, d in (("val/", val_dir), ("test/", test_dir)):
            r = match_split(d, prefix, out / "matched" / f"{key}-{prefix[:-1]}")
            caches[key].append(out / "matched" / f"{key}-{prefix[:-1]}" / "cache.npz")
    merged = {}
    for key, (cv, ct) in caches.items():
        merged[key] = out / "matched" / f"{key}-merged.npz"
        if not merged[key].is_file():
            a, b = np.load(cv, allow_pickle=False), np.load(ct, allow_pickle=False)
            if list(a["class_names"]) != list(b["class_names"]):
                raise ValueError(f"{key}: val/test class dictionaries differ")
            sn = np.concatenate([a["scene_names"], b["scene_names"]])
            off = len(a["scene_names"])
            z = {k: a[k] for k in ("class_names",)}
            z["scene_names"] = sn
            for k in ("tp_scene", "fn_scene", "fp_scene"):
                z[k] = np.concatenate([a[k], b[k] + off])
            for k in ("tp_cls", "tp_pred", "tp_gt", "tp_conf", "tp_iou", "fn_cls", "fp_cls", "fp_conf"):
                z[k] = np.concatenate([a[k], b[k]])
            z["n_tiles"] = np.concatenate([a["n_tiles"], b["n_tiles"]])
            np.savez_compressed(merged[key], **z)
    return merged


def group_point_and_draws(args):
    dataset, keys, B, start, stop, seed_tuple = args
    out = OUTROOT / dataset
    merged = {k: out / "matched" / f"{k}-merged.npz" for k in keys}
    names, roles = axis_for(list(merged.values()))
    conds = {k: X.Condition(merged[k], names) for k in keys}
    rng = np.random.default_rng(np.random.SeedSequence(list(seed_tuple)))
    rows = []
    ones = np.ones(len(names), dtype=np.int64)
    for b in range(-1, B):
        counts = ones if b < 0 else X.resample(rng, roles)
        if b >= 0 and (b < start or b >= stop):
            continue
        rec = {}
        for k, c in conds.items():
            q, mtr, _ = X.core.one_inner(c.data, counts, roles, geometry=True)
            fnr, e2e, rec_obj = X.gt_level(c, counts, roles, q)
            for mi, method in enumerate(("pooled", "hcp")):
                for j, name in enumerate(X.MACROS):
                    rec[f"{k}|{method}|{name}"] = float(np.mean(mtr[mi, :, j]))
                rec[f"{k}|{method}|e2e_risk"] = float(np.mean(e2e[mi]))
            rec[f"{k}|fnr"] = float(np.mean(fnr))
            rec[f"{k}|recall_object"] = float(rec_obj)
            if b < 0:
                rec[f"{k}|classes"] = {cn: {"q_hcp": float(q[1, i]), "cov_hcp": float(mtr[1, i, 0]),
                                            "cov_pooled": float(mtr[0, i, 0]), "fnr": float(fnr[i])}
                                       for i, cn in enumerate(c.class_names)}
        if b < 0 and start > 0:
            continue
        rows.append((b, rec))
    return keys, start, rows


def run(dataset, B, workers):
    out = OUTROOT / dataset
    out.mkdir(parents=True, exist_ok=True)
    keys_all = list(cells(dataset))
    groups = [[f"rgb-s{s}", f"ir-s{s}"] for s in (0, 1)] if dataset == "dv" else [[k] for k in keys_all]
    for g in groups:
        build_group(dataset, out, g)
    tasks = []
    for gi, g in enumerate(groups):
        chunk = max(1, B // 4)
        tasks += [(dataset, g, B, s, min(B, s + chunk), (20261003, 77, gi)) for s in range(0, B, chunk)]
    collected = collections.defaultdict(list)
    with ProcessPoolExecutor(workers) as ex:
        for keys, start, rows in ex.map(group_point_and_draws, tasks):
            collected[tuple(keys)] += rows
            print("done", keys, start, flush=True)
    summary = {}
    for keys, rows in collected.items():
        rows.sort(key=lambda t: t[0])
        point = rows[0][1]
        draws = [r for b, r in rows if b >= 0]
        assert rows[0][0] == -1 and len(draws) == B
        scalar = {k: v for k, v in point.items() if not k.endswith("|classes")}
        res = X.summarize(draws, scalar, B)
        if dataset == "dv":
            s = keys[0].split("-")[1]
            for name in ("hcp|coverage", "pooled|coverage", "fnr", "hcp|e2e_risk", "hcp|normalized_radius"):
                d = np.array([r[f"ir-{s}|{name}"] - r[f"rgb-{s}|{name}"] for r in draws])
                e = {"point": point[f"ir-{s}|{name}"] - point[f"rgb-{s}|{name}"]}
                if not np.isnan(d).any():
                    e.update(lower=X.ext_percentile(d, 2.5), upper=X.ext_percentile(d, 97.5))
                else:
                    e.update(lower=None, upper=None, status="UNAVAILABLE_UNDEFINED_DRAWS")
                res[f"ir_minus_rgb|{s}|{name}"] = e
        res["classes"] = {k.split("|")[0]: v for k, v in point.items() if k.endswith("|classes")}
        summary["+".join(keys)] = res
    meta = {"dataset": dataset, "B": B, "frozen": X.frozen_hashes(),
            "unit": "image (no audited source scenes); descriptive, not a source-scene certificate",
            "roles": "official val = calibration, official test = evaluation", "summary": summary}
    (out / f"{dataset.upper()}-SUMMARY.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=["dv", "rsar"])
    p.add_argument("--B", type=int, default=2000)
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    run(a.dataset, a.B, a.workers)


if __name__ == "__main__":
    main()

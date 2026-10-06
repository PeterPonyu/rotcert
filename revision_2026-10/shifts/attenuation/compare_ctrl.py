#!/usr/bin/env python3
"""Compare each clean pipeline-control replay (dior-<fam>-s<seed>-ctrl) with the clean detections exported at the end of
training (frozen closeout copy). Byte identity is checked first; otherwise per-image numeric differences are reported.
Read-only; prints one JSON line per pair and writes CTRL-COMPARISON.json next to the haze analysis."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
import os

import numpy as np

ROT = Path(os.environ.get("ROTCERT_ROOT", Path(__file__).resolve().parents[3]))
CLEAN = ROT / "revision_2026-10/closeout/collected-prefix/run/jobs"
PULL = ROT / "var/planc-results-20261003/haze"
OUT = ROT / "var/planc-results-20261003/analysis/xhaze/CTRL-COMPARISON.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def by_image(path):
    out = defaultdict(list)
    for line in Path(path).open():
        r = json.loads(line)
        out[r["image_id"]].append((r["class"], float(r["score"]), *map(float, r["obb"])))
    return out


def compare(base: str) -> dict:
    ctrl = PULL / f"{base}-ctrl"
    if not (ctrl / ".verified").is_file():
        return {"base": base, "status": "NOT_PULLED"}
    a, b = CLEAN / base / "detections.jsonl", ctrl / "detections.jsonl"
    res = {"base": base, "clean_sha256": sha(a), "ctrl_sha256": sha(b)}
    rec = json.loads((ctrl / "HAZE-RECORD.json").read_text())
    run = json.loads((CLEAN / base / "run.json").read_text())
    res["same_checkpoint"] = rec["haze"]["checkpoint_sha256"] == run["checkpoint_sha256"]
    if res["clean_sha256"] == res["ctrl_sha256"]:
        res["status"] = "BYTE_IDENTICAL"
        return res
    da, db = by_image(a), by_image(b)
    n_a, n_b = sum(map(len, da.values())), sum(map(len, db.values()))
    diff_imgs, max_score, max_box, count_mismatch = 0, 0.0, 0.0, 0
    for img in set(da) | set(db):
        x, y = sorted(da.get(img, [])), sorted(db.get(img, []))
        if x == y:
            continue
        diff_imgs += 1
        if len(x) != len(y) or [t[0] for t in x] != [t[0] for t in y]:
            count_mismatch += 1
            continue
        ax, ay = np.array([t[1:] for t in x]), np.array([t[1:] for t in y])
        max_score = max(max_score, float(np.abs(ax[:, 0] - ay[:, 0]).max()))
        max_box = max(max_box, float(np.abs(ax[:, 1:] - ay[:, 1:]).max()))
    res.update(status="NUMERIC_DIFFERENCES", detections_clean=n_a, detections_ctrl=n_b, images_differing=diff_imgs,
               images_with_count_or_class_mismatch=count_mismatch, max_abs_score_diff=max_score, max_abs_obb_diff=max_box)
    return res


def main():
    bases = [f"dior-{f}-s{s}" for s in (0, 1, 2) for f in ("orcnn", "roit", "rtmdet", "s2anet")]
    results = [compare(b) for b in bases]
    for r in results:
        print(json.dumps(r))
    OUT.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())

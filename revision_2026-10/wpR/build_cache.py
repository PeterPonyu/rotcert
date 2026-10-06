#!/usr/bin/env python3
"""matched.jsonl -> compact .npz cache (read-only on the source; sha256 recorded).

Usage: build_cache.py --cell <cell_id>   (reads cells.json)  |  --all

Manifest source: --cells-file, else $ROTCERT_CELLS, else the legacy cells.json.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from rotcert import io as rio  # noqa: E402


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(src, out, scene_rule="image"):
    rows = rio.load_jsonl(src)
    if scene_rule == "image":
        for r in rows:
            r.setdefault("scene_id", r["image_id"])
            if not r.get("scene_id"):
                r["scene_id"] = r["image_id"]
    else:
        rows = rio.populate_scene_ids(rows)
    scene_names = sorted({str(r["scene_id"]) for r in rows})
    class_names = sorted({str(r["class"]) for r in rows})
    sidx = {s: i for i, s in enumerate(scene_names)}
    cidx = {c: i for i, c in enumerate(class_names)}
    tiles = {}
    for r in rows:
        tiles.setdefault(str(r["scene_id"]), set()).add(str(r["image_id"]))
    tp = [r for r in rows if r.get("match_type") == "tp"]
    fn = [r for r in rows if r.get("match_type") == "fn"]
    fp = [r for r in rows if r.get("match_type") == "fp"]
    arr = dict(
        tp_scene=np.array([sidx[str(r["scene_id"])] for r in tp], dtype=np.int32),
        tp_cls=np.array([cidx[str(r["class"])] for r in tp], dtype=np.int32),
        tp_pred=np.array([r["pred_obb"] for r in tp], dtype=float).reshape(-1, 5),
        tp_gt=np.array([r["gt_obb"] for r in tp], dtype=float).reshape(-1, 5),
        tp_conf=np.array([float(r["pred_score"]) for r in tp], dtype=float),
        tp_iou=np.array([float(r["iou"]) for r in tp], dtype=float),
        fn_scene=np.array([sidx[str(r["scene_id"])] for r in fn], dtype=np.int32),
        fn_cls=np.array([cidx[str(r["class"])] for r in fn], dtype=np.int32),
        fp_scene=np.array([sidx[str(r["scene_id"])] for r in fp], dtype=np.int32),
        fp_cls=np.array([cidx[str(r["class"])] for r in fp], dtype=np.int32),
        fp_conf=np.array([float(r["pred_score"]) for r in fp], dtype=float),
        scene_names=np.array(scene_names),
        class_names=np.array(class_names),
        n_tiles=np.array([len(tiles[s]) for s in scene_names], dtype=np.int32),
        source_sha256=np.array(_sha256(src)),
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--cells-file", help="defaults to $ROTCERT_CELLS, else the legacy cells.json")
    a = ap.parse_args()
    cells_file = a.cells_file or os.environ.get("ROTCERT_CELLS") or str(HERE / "cells.json")
    cells = json.loads(Path(cells_file).read_text())
    todo = cells if a.all else [c for c in cells if c["cell_id"] == a.cell]
    for c in todo:
        got = _sha256(c["path"])
        if got != c["sha256"]:
            raise SystemExit(f"sha256 mismatch for {c['cell_id']}: {got} != {c['sha256']}")
        out = HERE.parent / "cache" / f"{c['cell_id']}.npz"
        build(c["path"], str(out), scene_rule=c["scene_rule"])
        print("cached", c["cell_id"], "->", out)


if __name__ == "__main__":
    main()

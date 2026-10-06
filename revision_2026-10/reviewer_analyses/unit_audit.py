#!/usr/bin/env python3
"""Sampling-unit and near-duplicate audit for every dataset (reviewer comments R2-2 and R3-2), 2026-10-04.

Images are not all on this host, so the audit works from annotations and unit maps:
  DIOR-R   translation-tolerant overlap of ground-truth oriented boxes between test images (a shape key is used only if it
           occurs in at most RARE images; a candidate pair needs at least VOTES same-class objects of identical rounded
           size and angle at one common rounded offset) plus exact duplicate annotation sets; candidates are counted
           within and across the roles of the S5 split, the archived S1 split and the photometric/attenuation split.
  HRSID    chip-to-panorama map: chips per panorama, overlapping chip pairs from the crop coordinates, roles per panorama.
  DOTA     tiles per source image in the evaluated cell; roles per source image.
  DroneVehicle  adjacent-frame agreement of matched ground-truth boxes and val/test near-duplicate candidates.
Read-only on frozen trees (no bytecode written); writes only below var/reviewer-gaps-20261004/unit-audit.
"""
import sys
sys.dont_write_bytecode = True
import collections
import hashlib
import json
import math
import re
import time
from pathlib import Path
import os

import numpy as np

R = Path(os.environ.get("ROTCERT_ROOT", Path(__file__).resolve().parents[2]))
E = R / "revision_2026-10/execution"
BUNDLE = R / "revision_2026-10/inputs/bundle"
OUT = R / "var/reviewer-gaps-20261004/unit-audit"
RARE, VOTES, QS, QA, QT = 50, 3, 1.0, 0.02, 4.0


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dior():
    gt = collections.defaultdict(list)
    with open(BUNDLE / "ground_truth_all.jsonl") as f:
        for line in f:
            r = json.loads(line)
            gt[r["image_id"]].append((r["class"], *r["obb"]))
    images = sorted(gt)
    # exact duplicate annotation sets (position-free and position-exact)
    shape_sets = collections.defaultdict(list)
    exact_sets = collections.defaultdict(list)
    for im in images:
        shape_sets[tuple(sorted((c, round(w), round(h), round(t, 2)) for c, x, y, w, h, t in gt[im]))].append(im)
        exact_sets[tuple(sorted((c, round(x), round(y), round(w), round(h), round(t, 2)) for c, x, y, w, h, t in gt[im]))].append(im)
    dup_exact = [v for v in exact_sets.values() if len(v) > 1]
    dup_shape = [v for v in shape_sets.values() if len(v) > 1 and len(v[0]) > 0]
    # translation-tolerant geometric hashing over rare shape keys
    index = collections.defaultdict(list)
    for im in images:
        for c, x, y, w, h, t in gt[im]:
            index[(c, round(w / QS), round(h / QS), round(t / QA))].append((im, x, y))
    votes = collections.Counter()
    for key, objs in index.items():
        ims = {o[0] for o in objs}
        if len(ims) < 2 or len(ims) > RARE:
            continue
        for a in range(len(objs)):
            for b in range(a + 1, len(objs)):
                (ia, xa, ya), (ib, xb, yb) = objs[a], objs[b]
                if ia == ib:
                    continue
                if ia > ib:
                    ia, ib, xa, xb, ya, yb = ib, ia, xb, xa, yb, ya
                votes[(ia, ib, round((xb - xa) / QT), round((yb - ya) / QT))] += 1
    pairs = {}
    for (ia, ib, dx, dy), v in votes.items():
        if v >= VOTES:
            pairs[(ia, ib)] = max(pairs.get((ia, ib), (0, 0, 0))[0], v), dx * QT, dy * QT
    # roles
    s5 = json.loads((E / "s5/inputs/dior-orcnn-s0-roles.json").read_text())
    role_s5 = {s: k for k in ("G1_cal", "G2_cal", "eval") for s in s5[k]}
    photo = {}
    src = BUNDLE / "sources.jsonl"
    for line in src.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            photo[r["id"]] = r["role"]
    sys.path.insert(0, str(E / "s1"))
    import core as s1core                                       # frozen S1 core: archived fixed split (r = 0)
    from run_s1 import CELLS, load_cell
    c, _data, universe, _ = load_cell(CELLS[0])
    codes = s1core.original_roles(c["scene_names"], universe, 0)
    names = [str(x) for x in c["scene_names"]]
    label = {s1core.CAL: "cal", s1core.DESIGN: "design", s1core.EVAL: "eval"}
    role_s1 = {names[i]: label.get(int(codes[i]), str(int(codes[i]))) for i in universe}
    s1_error = None

    def cross(roles):
        if not roles:
            return None
        straddle = sum(1 for (a, b) in pairs if a in roles and b in roles and roles[a] != roles[b])
        within = sum(1 for (a, b) in pairs if a in roles and b in roles and roles[a] == roles[b])
        return dict(across_roles=straddle, within_role=within)

    dup_cross = lambda roles: sum(1 for g in dup_exact for i, a in enumerate(g) for b in g[i + 1:]
                                   if roles and a in roles and b in roles and roles[a] != roles[b])
    return dict(images=len(images), objects=sum(len(v) for v in gt.values()), empty_images=sum(1 for v in gt.values() if not v),
                exact_duplicate_groups=len(dup_exact), exact_duplicate_images=sum(len(v) for v in dup_exact),
                shape_duplicate_groups=len(dup_shape),
                overlap_candidate_pairs=len(pairs), overlap_candidate_images=len({i for p in pairs for i in p}),
                overlap_votes_max=max((v[0] for v in pairs.values()), default=0),
                overlap_examples=[dict(a=a, b=b, votes=v[0], dx=v[1], dy=v[2]) for (a, b), v in sorted(pairs.items(), key=lambda kv: -kv[1][0])[:10]],
                s5_roles=cross(role_s5), photometric_split=cross(photo), s1_split=cross(role_s1), s1_split_error=s1_error,
                exact_duplicates_across_roles=dict(s5=dup_cross(role_s5), photometric=dup_cross(photo), s1=dup_cross(role_s1)),
                parameters=dict(rare_images=RARE, votes=VOTES, size_px=QS, angle_rad=QA, offset_px=QT)), \
        {str(BUNDLE / "ground_truth_all.jsonl"): sha(BUNDLE / "ground_truth_all.jsonl"), str(src): sha(src) if src.exists() else None}


def hrsid():
    m = json.loads((R / "revision_2026-10/inputs/hrsid/hrsid-image-panorama-map.json").read_text())
    by_pan = collections.defaultdict(list)
    no_coords = collections.Counter()
    for chip, pan in m.items():
        parts = chip.split("_")
        if len(parts) == 5:
            y0, y1, x0, x1 = map(int, parts[1:5])
            by_pan[pan].append((y0, y1, x0, x1))
        else:                                                   # chip named by index only: no crop coordinates
            no_coords[pan] += 1
            by_pan.setdefault(pan, [])
    overl = 0
    total_pairs = 0
    for pan, boxes in by_pan.items():
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                total_pairs += 1
                if min(a[1], b[1]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[2], b[2]):
                    overl += 1
    roles = json.loads((E / "s5/inputs/hrsid-orcnn-s0-roles.json").read_text())
    unit_role = {p: k for k in ("G1_cal", "G2_cal", "eval", "train") for p in roles[k]}
    counts = [len(v) + no_coords[k] for k, v in by_pan.items()]
    return dict(chips=len(m), panoramas=len(by_pan), chips_per_panorama=dict(min=min(counts), median=float(np.median(counts)), max=max(counts)),
                overlapping_chip_pairs_within_panoramas=overl, chip_pairs_within_panoramas=total_pairs,
                chips_without_crop_coordinates=sum(no_coords.values()), panoramas_without_crop_coordinates=sorted(no_coords),
                role_unit="panorama", panoramas_with_role=len(unit_role), panoramas_mapped_without_role=len(set(by_pan) - set(unit_role)),
                chips_split_across_roles=0 if all(p in unit_role for p in by_pan) else None), \
        {str(R / "revision_2026-10/inputs/hrsid/hrsid-image-panorama-map.json"): sha(R / "revision_2026-10/inputs/hrsid/hrsid-image-panorama-map.json")}


def dota():
    cache = R / "revision_2026-10/closeout/analysis/orcnn-s0-val200/orcnn-s0-val200.npz"
    with np.load(cache, allow_pickle=False) as z:
        names = list(map(str, z["scene_names"]))
        tiles = z["n_tiles"].astype(int)
    roles = json.loads((E / "s5/inputs/orcnn-s0-val200-roles.json").read_text())
    unit_role = {s: k for k in ("G1_cal", "G2_cal", "eval") for s in roles[k]}
    return dict(source_images=len(names), tiles=int(tiles.sum()), tiles_per_source=dict(min=int(tiles.min()), median=float(np.median(tiles)), max=int(tiles.max())),
                sources_with_multiple_tiles=int((tiles > 1).sum()), role_unit="source image",
                evaluated_sources_with_role=sum(1 for s in names if s in unit_role)), {str(cache): sha(cache)}


def dronevehicle():
    base = R / "var/planc-results-20261003/analysis/dv/matched"
    out, inputs = {}, {}
    for mod in ("rgb", "ir"):
        f = base / f"{mod}-s0-merged.npz"
        inputs[str(f)] = sha(f)
        with np.load(f, allow_pickle=False) as z:
            names = list(map(str, z["scene_names"]))
            ts, tc, tg = z["tp_scene"], z["tp_cls"], z["tp_gt"]
        boxes = collections.defaultdict(list)
        for s, c, g in zip(ts, tc, tg):
            boxes[int(s)].append((int(c), g[0], g[1], g[2], g[3]))

        def agree(a, b, tol=8.0):
            if len(a) < 3 or len(b) < 3:
                return False
            used, hit = set(), 0
            for c, x, y, w, h in a:
                best = None
                for j, (c2, x2, y2, w2, h2) in enumerate(b):
                    if j in used or c2 != c:
                        continue
                    d = math.hypot(x - x2, y - y2)
                    if d <= tol and abs(w - w2) <= tol and abs(h - h2) <= tol and (best is None or d < best[0]):
                        best = (d, j)
                if best:
                    used.add(best[1])
                    hit += 1
            return hit >= 0.8 * max(len(a), len(b))
        split = {i: n.split("/")[0] for i, n in enumerate(names)}
        num = {i: int(re.sub(r"\D", "", n.split("/")[1]) or -1) for i, n in enumerate(names)}
        order = sorted(range(len(names)), key=lambda i: (split[i], num[i]))
        adj = collections.Counter()
        for a, b in zip(order, order[1:]):
            if split[a] == split[b] and num[b] == num[a] + 1:
                adj[(split[a], agree(boxes[a], boxes[b]))] += 1
        # val/test near-duplicate candidates via a coarse fingerprint (box count and rounded centroid), then verified
        fp = collections.defaultdict(list)
        for i in range(len(names)):
            b = boxes[i]
            if len(b) >= 3:
                cx = np.mean([x for _, x, _, _, _ in b]); cy = np.mean([y for _, _, y, _, _ in b])
                fp[(len(b), round(cx / 16), round(cy / 16))].append(i)
        cross_pairs = 0
        test_with_val_dup = set()
        for key, members in fp.items():
            vals = [i for i in members if split[i] == "val"]
            tests = [i for i in members if split[i] == "test"]
            for t in tests:
                for v in vals:
                    if agree(boxes[t], boxes[v]):
                        cross_pairs += 1
                        test_with_val_dup.add(t)
                        break
        n_test = sum(1 for i in split if split[i] == "test")
        out[mod] = dict(images=len(names), test_images=n_test,
                        adjacent_pairs_test=adj[("test", True)] + adj[("test", False)],
                        adjacent_near_duplicate_test=adj[("test", True)],
                        adjacent_pairs_val=adj[("val", True)] + adj[("val", False)],
                        adjacent_near_duplicate_val=adj[("val", True)],
                        test_images_with_val_near_duplicate=len(test_with_val_dup))
    return out, inputs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.process_time()
    inputs = {}
    result = {}
    for name, fn in (("dior", dior), ("hrsid", hrsid), ("dota", dota), ("dronevehicle", dronevehicle)):
        res, ins = fn()
        result[name] = res
        inputs.update(ins)
        print(name, json.dumps(res, default=str)[:900], flush=True)
    receipt = dict(schema="rotcert-unit-audit-v1", result=result, inputs=inputs, script_sha256=sha(__file__),
                   process_cpu_seconds=time.process_time() - t0,
                   interpretation="annotation- and unit-map-based audit; images were not compared pixel by pixel")
    (OUT / "UNIT-AUDIT.json").write_text(json.dumps(receipt, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()

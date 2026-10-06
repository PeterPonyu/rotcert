#!/usr/bin/env python3
"""Third sampling-unit audit (supersedes unit_audit_v2.py for DIOR-R reporting and the DroneVehicle screen),
2026-10-04, after the second independent review.

Changes:
  DIOR-R  the full list of candidate pairs is stored with its connected components (v2 kept ten examples), so that a
          sensitivity analysis can exclude every candidate image; counting is unchanged (one-to-one votes, offset from the
          same bucket as the votes).
  DroneVehicle  near-identity of two annotation sets is decided by a maximum one-to-one correspondence of same-class
          boxes within 8 pixels in center and size (v2 used a greedy, order-dependent assignment), with a recorded check
          that shuffling the box order never changes a decision. The search windows are centroid-cell neighbourhoods of
          16-pixel cells (the cell and its neighbours one or three cells away), not exact pixel distances. Flagged pairs
          are candidates from annotations, not image-confirmed duplicates.
Read-only on frozen trees (no bytecode written); writes only UNIT-AUDIT-v3.json below var/reviewer-gaps-20261004/unit-audit.
"""
import sys
sys.dont_write_bytecode = True
import collections
import importlib.util
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v1 = _load('unit_audit')
v2 = _load('unit_audit_v2')
R, E, BUNDLE, OUT = v1.R, v1.E, v1.BUNDLE, v1.OUT
RARE, VOTES, QS, QA, QT = v1.RARE, v1.VOTES, v1.QS, v1.QA, v1.QT
sha, max_matching = v1.sha, v2.max_matching
TOL, SHARE, CELL = 8.0, 0.8, 16


def components(pairs):
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[find(a)] = find(b)
    groups = collections.defaultdict(list)
    for x in list(parent):
        groups[find(x)].append(x)
    return sorted(sorted(g) for g in groups.values())


def dior():
    res, ins = v2.dior()
    # rebuild the full pair list with the v2 rule (same code path, all pairs kept)
    gt = collections.defaultdict(list)
    with open(BUNDLE / "ground_truth_all.jsonl") as f:
        for line in f:
            r = json.loads(line)
            gt[r["image_id"]].append((r["class"], *r["obb"]))
    index = collections.defaultdict(list)
    for im in sorted(gt):
        for j, (c, x, y, w, h, t) in enumerate(gt[im]):
            index[(c, round(w / QS), round(h / QS), round(t / QA))].append((im, j, x, y))
    edges = collections.defaultdict(set)
    for objs in index.values():
        ims = {o[0] for o in objs}
        if len(ims) < 2 or len(ims) > RARE:
            continue
        for a in range(len(objs)):
            for b in range(a + 1, len(objs)):
                (ia, ja, xa, ya), (ib, jb, xb, yb) = objs[a], objs[b]
                if ia == ib:
                    continue
                if ia > ib:
                    ia, ib, ja, jb, xa, xb, ya, yb = ib, ia, jb, ja, xb, xa, yb, ya
                edges[(ia, ib, round((xb - xa) / QT), round((yb - ya) / QT))].add((ja, jb))
    pairs = {}
    for (ia, ib, dx, dy), e in edges.items():
        v = max_matching(e)
        if v >= VOTES and v > pairs.get((ia, ib), (0,))[0]:
            pairs[(ia, ib)] = (v, dx * QT, dy * QT)
    assert len(pairs) == res["overlap_candidate_pairs"], (len(pairs), res["overlap_candidate_pairs"])
    comps = components(list(pairs))
    res["candidate_pairs_all"] = [dict(a=a, b=b, votes=v, dx=dx, dy=dy) for (a, b), (v, dx, dy) in sorted(pairs.items())]
    res["candidate_images_all"] = sorted({i for p in pairs for i in p})
    res["candidate_components"] = comps
    res["candidate_component_sizes"] = sorted((len(c) for c in comps), reverse=True)
    return res, ins


def agree_max(a, b, tol=TOL, share=SHARE):
    """Near-identity by a maximum one-to-one correspondence of same-class boxes within tol in center and size."""
    if len(a) < 3 or len(b) < 3:
        return False
    e = [(i, j) for i, (c, x, y, w, h) in enumerate(a) for j, (c2, x2, y2, w2, h2) in enumerate(b)
         if c == c2 and math.hypot(x - x2, y - y2) <= tol and abs(w - w2) <= tol and abs(h - h2) <= tol]
    return max_matching(e) >= share * max(len(a), len(b))


def dronevehicle():
    base = R / "var/planc-results-20261003/analysis/dv/matched"
    out, inputs, by_name = {}, {}, {}
    rnd = random.Random(20261004)
    for mod in ("rgb", "ir"):
        f = base / f"{mod}-s0-merged.npz"
        inputs[str(f)] = sha(f)
        with np.load(f, allow_pickle=False) as z:
            names = list(map(str, z["scene_names"]))
            ts, tc, tg, fs = z["tp_scene"], z["tp_cls"], z["tp_gt"], z["fn_scene"]
        boxes = collections.defaultdict(list)
        for s, c, g in zip(ts, tc, tg):
            boxes[int(s)].append((int(c), float(g[0]), float(g[1]), float(g[2]), float(g[3])))
        by_name[mod] = {names[i]: boxes[i] for i in range(len(names))}
        split = {i: n.split("/")[0] for i, n in enumerate(names)}
        num = {i: int(re.sub(r"\D", "", n.split("/")[1]) or -1) for i, n in enumerate(names)}
        order = sorted(range(len(names)), key=lambda i: (split[i], num[i]))
        adj = collections.Counter()
        for a, b in zip(order, order[1:]):
            if split[a] == split[b] and num[b] == num[a] + 1:
                adj[(split[a], agree_max(boxes[a], boxes[b]))] += 1
        cen = {i: (np.mean([x for _, x, _, _, _ in boxes[i]]), np.mean([y for _, _, y, _, _ in boxes[i]]))
               for i in range(len(names)) if len(boxes[i]) >= 3}
        grid = collections.defaultdict(list)
        for i, (cx, cy) in cen.items():
            if split[i] == "val":
                grid[(round(cx / CELL), round(cy / CELL))].append(i)
        windows, perm_checks, perm_changes = {}, 0, 0
        for reach in (1, 3):
            compared, flagged = 0, []
            for t, (cx, cy) in cen.items():
                if split[t] != "test":
                    continue
                gx, gy = round(cx / CELL), round(cy / CELL)
                for dx in range(-reach, reach + 1):
                    for dy in range(-reach, reach + 1):
                        for v in grid.get((gx + dx, gy + dy), ()):
                            lt, lv = len(boxes[t]), len(boxes[v])
                            if min(lt, lv) < SHARE * max(lt, lv):
                                continue
                            compared += 1
                            hit = agree_max(boxes[t], boxes[v])
                            if hit or rnd.random() < 0.02:          # permutation-invariance check on all hits + 2% sample
                                bt, bv = boxes[t][:], boxes[v][:]
                                rnd.shuffle(bt); rnd.shuffle(bv)
                                perm_checks += 1
                                perm_changes += int(agree_max(bt, bv) != hit)
                            if hit:
                                flagged.append((names[t], names[v], len(boxes[t]), len(boxes[v])))
            windows[f"cells_{reach}"] = dict(cell_px=CELL, neighbour_cells=reach, candidate_comparisons=compared,
                                             flagged=sorted(flagged))
        test_ids = [i for i in split if split[i] == "test"]
        val_ids = [i for i in split if split[i] == "val"]
        fn_per = np.bincount(fs, minlength=len(names))
        out[mod] = dict(images=len(names), test_images=len(test_ids), val_images=len(val_ids),
                        test_images_screened=sum(1 for t in test_ids if t in cen),
                        test_images_not_screened_fewer_than_3_matched=sum(1 for t in test_ids if t not in cen),
                        unmatched_gt_not_used=dict(test=int(fn_per[test_ids].sum()), val=int(fn_per[val_ids].sum())),
                        adjacent_pairs_test=adj[("test", True)] + adj[("test", False)],
                        adjacent_near_identical_test=adj[("test", True)],
                        adjacent_pairs_val=adj[("val", True)] + adj[("val", False)],
                        adjacent_near_identical_val=adj[("val", True)],
                        search_windows=windows, permutation_checks=perm_checks, permutation_decision_changes=perm_changes)
        print(mod, {k: (w["candidate_comparisons"], len(w["flagged"])) for k, w in windows.items()},
              "perm", perm_checks, perm_changes, flush=True)
    for mod, other in (("rgb", "ir"), ("ir", "rgb")):
        checks = []
        for w in out[mod]["search_windows"].values():
            for t, v, _, _ in w["flagged"]:
                a, b = by_name[other].get(t, []), by_name[other].get(v, [])
                checks.append(dict(test=t, val=v, other_modality=other, matched_boxes=[len(a), len(b)], agrees=agree_max(a, b)))
        out[mod]["flagged_checked_in_other_modality"] = checks
    union = sorted({(a, b) for m in ("rgb", "ir") for w in out[m]["search_windows"].values() for a, b, _, _ in w["flagged"]})
    out["distinct_candidate_frame_pairs"] = [list(p) for p in union]
    return out, inputs


def main():
    t0 = time.process_time()
    inputs = {str(HERE / "unit_audit.py"): sha(HERE / "unit_audit.py"), str(HERE / "unit_audit_v2.py"): sha(HERE / "unit_audit_v2.py")}
    result = {}
    for name, fn in (("dior", dior), ("hrsid", v1.hrsid), ("dota", v1.dota), ("dronevehicle", dronevehicle)):
        res, ins = fn()
        result[name] = res
        inputs.update(ins)
    receipt = dict(schema="rotcert-unit-audit-v3", supersedes="UNIT-AUDIT-v2.json", result=result, inputs=inputs,
                   script_sha256=sha(__file__), process_cpu_seconds=time.process_time() - t0,
                   interpretation=("annotation- and unit-map-based audit; images were not compared pixel by pixel; "
                                   "DroneVehicle candidates come from detector-matched boxes only and are not image-confirmed"))
    path = OUT / "UNIT-AUDIT-v3.json"
    if path.exists():
        raise SystemExit("receipt exists")
    path.write_text(json.dumps(receipt, indent=1, default=str, allow_nan=False) + "\n")
    d, v = result["dior"], result["dronevehicle"]
    print("dior pairs", d["overlap_candidate_pairs"], "images", len(d["candidate_images_all"]), "components", d["candidate_component_sizes"])
    print("dv distinct pairs", len(v["distinct_candidate_frame_pairs"]), v["distinct_candidate_frame_pairs"])
    print(path, sha(path))


if __name__ == "__main__":
    main()

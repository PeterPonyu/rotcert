#!/usr/bin/env python3
"""Second sampling-unit audit (supersedes unit_audit.py for DIOR-R and DroneVehicle; HRSID and DOTA-v1.0 are rerun with
the first script's unchanged functions), 2026-10-04.

Fixes:
  DIOR-R  a candidate offset now counts a maximum one-to-one correspondence between the boxes of the two images (one box
          cannot vote twice), and the reported votes and offset of a pair come from the same offset bucket.
  DroneVehicle  the screen still sees matched ground-truth boxes only (the unmatched boxes' geometry is not in the local
          records), but the fingerprint now also searches the eight neighbouring centroid buckets and any box count within
          the agreement rule's reach, and the exclusions are counted: test images with fewer than three matched boxes, and
          unmatched ground-truth objects. The result is a restricted matched-box screen, not a duplicate census.
Read-only on frozen trees (no bytecode written); writes only UNIT-AUDIT-v2.json below var/reviewer-gaps-20261004/unit-audit.
"""
import sys
sys.dont_write_bytecode = True
import collections
import hashlib
import importlib.util
import json
import math
import re
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("unit_audit_v1", HERE / "unit_audit.py")
v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1)
R, E, BUNDLE, OUT = v1.R, v1.E, v1.BUNDLE, v1.OUT
RARE, VOTES, QS, QA, QT = v1.RARE, v1.VOTES, v1.QS, v1.QA, v1.QT
sha = v1.sha


def max_matching(edges):
    """Maximum bipartite matching size (augmenting paths); edges: iterable of (left, right)."""
    adj = collections.defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
    match_r = {}

    def augment(a, seen):
        for b in adj[a]:
            if b in seen:
                continue
            seen.add(b)
            if b not in match_r or augment(match_r[b], seen):
                match_r[b] = a
                return True
        return False

    return sum(augment(a, set()) for a in list(adj))


def dior():
    gt = collections.defaultdict(list)
    with open(BUNDLE / "ground_truth_all.jsonl") as f:
        for line in f:
            r = json.loads(line)
            gt[r["image_id"]].append((r["class"], *r["obb"]))
    images = sorted(gt)
    exact_sets = collections.defaultdict(list)
    for im in images:
        exact_sets[tuple(sorted((c, round(x), round(y), round(w), round(h), round(t, 2)) for c, x, y, w, h, t in gt[im]))].append(im)
    dup_exact = [v for v in exact_sets.values() if len(v) > 1]
    index = collections.defaultdict(list)
    for im in images:
        for j, (c, x, y, w, h, t) in enumerate(gt[im]):
            index[(c, round(w / QS), round(h / QS), round(t / QA))].append((im, j, x, y))
    edges = collections.defaultdict(set)                         # (ia, ib, dx, dy) -> {(box in ia, box in ib)}
    for key, objs in index.items():
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
    raw_votes = {k: len(v) for k, v in edges.items()}
    pairs, raw_pairs = {}, set()
    for (ia, ib, dx, dy), e in edges.items():
        if raw_votes[(ia, ib, dx, dy)] >= VOTES:
            raw_pairs.add((ia, ib))
        v = max_matching(e)
        if v >= VOTES and v > pairs.get((ia, ib), (0,))[0]:
            pairs[(ia, ib)] = (v, dx * QT, dy * QT)                 # votes and offset from the same bucket
    s5 = json.loads((E / "s5/inputs/dior-orcnn-s0-roles.json").read_text())
    role_s5 = {s: k for k in ("G1_cal", "G2_cal", "eval") for s in s5[k]}
    src = BUNDLE / "sources.jsonl"
    photo = {}
    for line in src.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            photo[r["id"]] = r["role"]
    sys.path.insert(0, str(E / "s1"))
    import core as s1core
    from run_s1 import CELLS, load_cell
    c, _data, universe, _ = load_cell(CELLS[0])
    codes = s1core.original_roles(c["scene_names"], universe, 0)
    names = [str(x) for x in c["scene_names"]]
    label = {s1core.CAL: "cal", s1core.DESIGN: "design", s1core.EVAL: "eval"}
    role_s1 = {names[i]: label.get(int(codes[i]), str(int(codes[i]))) for i in universe}

    def cross(roles):
        return dict(across_roles=sum(1 for (a, b) in pairs if a in roles and b in roles and roles[a] != roles[b]),
                    within_role=sum(1 for (a, b) in pairs if a in roles and b in roles and roles[a] == roles[b]))

    dup_cross = lambda roles: sum(1 for g in dup_exact for i, a in enumerate(g) for b in g[i + 1:]
                                   if a in roles and b in roles and roles[a] != roles[b])
    examples = sorted(pairs.items(), key=lambda kv: (-kv[1][0], kv[0]))[:10]
    # self-check: every reported example offset really has the reported one-to-one votes
    for (a, b), (v, dx, dy) in examples:
        assert max_matching(edges[(a, b, round(dx / QT), round(dy / QT))]) == v
    return dict(images=len(images), objects=sum(len(v) for v in gt.values()),
                exact_duplicate_groups=len(dup_exact), exact_duplicate_images=sum(len(v) for v in dup_exact),
                overlap_candidate_pairs=len(pairs), overlap_candidate_images=len({i for p in pairs for i in p}),
                first_run_pairs_without_one_to_one=len(raw_pairs),
                overlap_votes_max=max((v[0] for v in pairs.values()), default=0),
                overlap_examples=[dict(a=a, b=b, votes=v, dx=dx, dy=dy) for (a, b), (v, dx, dy) in examples],
                s5_roles=cross(role_s5), photometric_split=cross(photo), s1_split=cross(role_s1),
                exact_duplicates_across_roles=dict(s5=dup_cross(role_s5), photometric=dup_cross(photo), s1=dup_cross(role_s1)),
                parameters=dict(rare_images=RARE, votes=VOTES, size_px=QS, angle_rad=QA, offset_px=QT, one_to_one=True)), \
        {str(BUNDLE / "ground_truth_all.jsonl"): sha(BUNDLE / "ground_truth_all.jsonl"), str(src): sha(src)}


def agree_rule(a, b, tol=8.0):
    """Near-identical: at least 80% of the boxes of the larger set agree in class and within tol px in center and size."""
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


def dronevehicle():
    base = R / "var/planc-results-20261003/analysis/dv/matched"
    out, inputs, boxes_by_name = {}, {}, {}
    for mod in ("rgb", "ir"):
        f = base / f"{mod}-s0-merged.npz"
        inputs[str(f)] = sha(f)
        with np.load(f, allow_pickle=False) as z:
            names = list(map(str, z["scene_names"]))
            ts, tc, tg, fs = z["tp_scene"], z["tp_cls"], z["tp_gt"], z["fn_scene"]
        boxes = collections.defaultdict(list)
        for s, c, g in zip(ts, tc, tg):
            boxes[int(s)].append((int(c), g[0], g[1], g[2], g[3]))

        agree = agree_rule

        split = {i: n.split("/")[0] for i, n in enumerate(names)}
        num = {i: int(re.sub(r"\D", "", n.split("/")[1]) or -1) for i, n in enumerate(names)}
        order = sorted(range(len(names)), key=lambda i: (split[i], num[i]))
        adj = collections.Counter()
        for a, b in zip(order, order[1:]):
            if split[a] == split[b] and num[b] == num[a] + 1:
                adj[(split[a], agree(boxes[a], boxes[b]))] += 1
        # val index by rounded centroid; each test image is compared with every val image in the 3 x 3 neighbouring
        # buckets whose box count the agreement rule can reach (smaller/larger >= 0.8)
        cen = {}
        for i in range(len(names)):
            b = boxes[i]
            if len(b) >= 3:
                cen[i] = (np.mean([x for _, x, _, _, _ in b]), np.mean([y for _, _, y, _, _ in b]))
        grid = collections.defaultdict(list)
        for i, (cx, cy) in cen.items():
            if split[i] == "val":
                grid[(round(cx / 16), round(cy / 16))].append(i)
        windows = {}
        for reach in (1, 3):                                    # +-16 px and +-48 px centroid windows
            compared, flagged = 0, []
            for t, (cx, cy) in cen.items():
                if split[t] != "test":
                    continue
                gx, gy = round(cx / 16), round(cy / 16)
                for dx in range(-reach, reach + 1):
                    for dy in range(-reach, reach + 1):
                        for v in grid.get((gx + dx, gy + dy), ()):
                            lt, lv = len(boxes[t]), len(boxes[v])
                            if min(lt, lv) < 0.8 * max(lt, lv):
                                continue
                            compared += 1
                            if agree(boxes[t], boxes[v]):
                                flagged.append((names[t], names[v], len(boxes[t]), len(boxes[v])))
            windows[str(16 * reach)] = dict(candidate_comparisons=compared, flagged=sorted(flagged))
        cross_pairs = len(windows["16"]["flagged"])
        test_hit = {t for t, _, _, _ in windows["16"]["flagged"]}
        compared = windows["16"]["candidate_comparisons"]
        boxes_by_name[mod] = {names[i]: boxes[i] for i in range(len(names))}
        test_ids = [i for i in split if split[i] == "test"]
        val_ids = [i for i in split if split[i] == "val"]
        fn_per = np.bincount(fs, minlength=len(names))
        out[mod] = dict(images=len(names), test_images=len(test_ids), val_images=len(val_ids),
                        test_images_screened=sum(1 for t in test_ids if t in cen),
                        test_images_not_screened_fewer_than_3_matched=sum(1 for t in test_ids if t not in cen),
                        val_images_not_screened_fewer_than_3_matched=sum(1 for v in val_ids if v not in cen),
                        unmatched_gt_not_used=dict(test=int(fn_per[test_ids].sum()), val=int(fn_per[val_ids].sum()),
                                                   all=int(fn_per.sum())),
                        matched_gt_used=dict(test=int(sum(len(boxes[t]) for t in test_ids)), val=int(sum(len(boxes[v]) for v in val_ids))),
                        candidate_comparisons=compared, val_test_pairs_flagged=cross_pairs,
                        test_images_with_val_near_duplicate=len(test_hit),
                        adjacent_pairs_test=adj[("test", True)] + adj[("test", False)],
                        adjacent_near_duplicate_test=adj[("test", True)],
                        adjacent_pairs_val=adj[("val", True)] + adj[("val", False)],
                        adjacent_near_duplicate_val=adj[("val", True)], search_windows_px=windows)
    # every flagged pair checked in the other modality with the same rule (paired frames share their names)
    for mod, other in (("rgb", "ir"), ("ir", "rgb")):
        checks = []
        for w in out[mod]["search_windows_px"].values():
            for t, v, _, _ in w["flagged"]:
                a, b = boxes_by_name[other].get(t, []), boxes_by_name[other].get(v, [])
                checks.append(dict(test=t, val=v, other_modality=other, matched_boxes=[len(a), len(b)], agrees=agree_rule(a, b)))
        out[mod]["flagged_checked_in_other_modality"] = checks
    return out, inputs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.process_time()
    inputs, result = {str(HERE / "unit_audit.py"): sha(HERE / "unit_audit.py")}, {}
    for name, fn in (("dior", dior), ("hrsid", v1.hrsid), ("dota", v1.dota), ("dronevehicle", dronevehicle)):
        res, ins = fn()
        result[name] = res
        inputs.update(ins)
        print(name, json.dumps(res, default=str)[:1200], flush=True)
    receipt = dict(schema="rotcert-unit-audit-v2", supersedes="UNIT-AUDIT.json (v1: DIOR-R votes without one-to-one "
                   "correspondence, mismatched offsets; DroneVehicle exact-bucket fingerprint)", result=result, inputs=inputs,
                   script_sha256=sha(__file__), process_cpu_seconds=time.process_time() - t0,
                   interpretation=("annotation- and unit-map-based audit; images were not compared pixel by pixel; the "
                                   "DroneVehicle screen uses matched ground-truth boxes only and is not a duplicate census"))
    path = OUT / "UNIT-AUDIT-v2.json"
    path.write_text(json.dumps(receipt, indent=1, default=str) + "\n")
    print(path, sha(path))


if __name__ == "__main__":
    main()

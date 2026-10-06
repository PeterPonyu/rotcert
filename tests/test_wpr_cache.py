import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "revision_2026-10" / "wpR"))


def _write(tmp_path):
    rows = [
        {"image_id": "P0001__1024__0___0", "class": "ship", "pred_obb": [10, 10, 8, 2, 0.1],
         "gt_obb": [10, 11, 8, 2, 0.1], "pred_score": 0.9, "iou": 0.8, "match_type": "tp"},
        {"image_id": "P0001__1024__824___0", "class": "ship", "pred_obb": [5, 5, 4, 2, 0.0],
         "gt_obb": None, "pred_score": 0.3, "iou": None, "match_type": "fp"},
        {"image_id": "P0002__1024__0___0", "class": "plane", "pred_obb": None,
         "gt_obb": [50, 50, 20, 20, 0.0], "pred_score": None, "iou": None, "match_type": "fn"},
    ]
    p = tmp_path / "matched.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_build_cache_roundtrip(tmp_path):
    import build_cache
    import common
    src = _write(tmp_path)
    out = tmp_path / "c.npz"
    build_cache.build(str(src), str(out), scene_rule="dota_crop")
    c = common.load_npz(str(out))
    assert list(c["scene_names"]) == ["P0001", "P0002"]
    assert c["tp_pred"].shape == (1, 5)
    assert c["fn_scene"].tolist() == [1]
    assert c["fp_conf"].tolist() == [0.3]
    assert c["n_tiles"].tolist() == [2, 1]
    s = common.gwd_scores(c)
    assert s.shape == (1,) and s[0] > 0


def test_iter_splits_partitions_universe_and_matches_legacy():
    import common
    from rotcert.splits import three_way_scene_split
    names = np.array([f"P{i:04d}" for i in range(60)])
    universe = np.arange(0, 60, 2)  # only even codes hold TPs
    parts = list(common.iter_splits(names, universe, 3))
    assert len(parts) == 3
    for r, cal, des, ev in parts:
        allidx = np.concatenate([cal, des, ev])
        assert sorted(allidx.tolist()) == universe.tolist()
        legacy = three_way_scene_split([str(names[i]) for i in universe], cal_frac=0.4, match_frac=0.2, seed=r)
        assert sorted(names[cal].tolist()) == legacy["calibration"]
        assert sorted(names[ev].tolist()) == legacy["eval"]


def test_scene_slices():
    import common
    sc = np.array([2, 0, 2, 1, 0])
    order = np.argsort(sc, kind="mergesort")
    b = common.scene_slices(sc, 3)
    assert [sorted(order[b[i]:b[i + 1]].tolist()) for i in range(3)] == [[1, 4], [3], [0, 2]]

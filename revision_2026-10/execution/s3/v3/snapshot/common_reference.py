"""Shared helpers for the revision re-analysis runners."""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from rotcert.gwd import obb_gwd  # noqa: E402
from rotcert.splits import three_way_scene_split  # noqa: E402

CACHE = HERE.parent / "cache"
ALPHA, BETA, DELTA = 0.10, 0.20, 0.05
CAL_FRAC, DESIGN_FRAC = 0.4, 0.2


def cells_path():
    """Legacy cells.json by default; override with --cells-file (per-runner) or
    ROTCERT_CELLS (global). The legacy default is unchanged when neither is set."""
    override = os.environ.get("ROTCERT_CELLS")
    return Path(override) if override else (HERE / "cells.json")


def cells():
    return json.loads(cells_path().read_text())


def cell(cell_id):
    return next(c for c in cells() if c["cell_id"] == cell_id)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_cache(cell_id):
    return load_npz(str(CACHE / f"{cell_id}.npz"))


def gwd_scores(c):
    return obb_gwd(c["tp_pred"], c["tp_gt"])


def m_bound(c, cell_meta):
    """Hard a-priori bound on TPs per scene: max_per_img x max tiles per scene."""
    return int(cell_meta["max_per_img"]) * int(c["n_tiles"].max())


def tp_universe(c):
    return np.unique(c["tp_scene"]).astype(np.int64)


def gt_universe(c):
    return np.unique(np.concatenate([c["tp_scene"], c["fn_scene"]])).astype(np.int64)


def iter_splits(scene_names, universe_codes, R, cal_frac=CAL_FRAC, design_frac=DESIGN_FRAC):
    """Legacy-compatible scene splits: three_way_scene_split on the actual scene names."""
    names = [str(scene_names[i]) for i in universe_codes]
    lookup = {str(scene_names[i]): int(i) for i in universe_codes}
    for r in range(R):
        p = three_way_scene_split(names, cal_frac=cal_frac, match_frac=design_frac, seed=r)
        yield (
            r,
            np.array([lookup[n] for n in p["calibration"]], dtype=np.int64),
            np.array([lookup[n] for n in p["matching"]], dtype=np.int64),
            np.array([lookup[n] for n in p["eval"]], dtype=np.int64),
        )


def scene_slices(scene_codes, n_codes):
    """Bounds b with rows of scene i = order[b[i]:b[i+1]], order = argsort(scene_codes, stable)."""
    sc_sorted = np.sort(np.asarray(scene_codes), kind="mergesort")
    return np.searchsorted(sc_sorted, np.arange(n_codes + 1), side="left")


def mask_for(scene_codes, scene_idx):
    sel = np.zeros(int(scene_codes.max(initial=-1)) + 1 if scene_codes.size else 0, dtype=bool)
    keep = scene_idx[scene_idx < sel.size]
    sel[keep] = True
    return sel[scene_codes] if scene_codes.size else np.zeros(0, dtype=bool)


def dump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=float))

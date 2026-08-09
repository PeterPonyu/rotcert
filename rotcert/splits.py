"""Scene-level 3-way repeated splits (design §4.2, the C1 crop-overlap-leakage fix).

**The unit of exchangeability, splitting, and resampling is the SOURCE SCENE (the
original DOTA image), never the crop.** DOTA is tiled into 1024x1024 crops with 200px
overlap, so a single physical object can appear in >= 2 adjacent crops; splitting by
crop would place the same object in both calibration and evaluation, destroying
exchangeability and silently inflating coverage (design §4.2, "CRITICAL"). Every function
here operates on scene ids, never crop ids, and :func:`three_way_scene_split` REFUSES
(raises :class:`SplitError`) if asked to split anything that looks like a bare crop
table without scene ids resolvable.

Scene-id extraction from crop filenames
------------------------------------------
DOTA's mmrotate split-crop naming convention embeds the source image id as a prefix,
e.g. ``P0006__1024__0___0.png`` -> scene id ``P0006`` (image id, then crop
window/offset suffix separated by double underscores). :data:`DEFAULT_SCENE_ID_REGEX`
implements exactly this; DIOR-R images are already single-tile (scene == image, no
crop suffix), so the same regex degrades to the identity for a bare ``00001.jpg``-style
id (no ``__`` present -> the whole stem is returned as the scene id). The pattern is
configurable (``scene_id_regex``) precisely because it is a naming convention, not a
mathematical fact -- verify against the actual staged filenames at Phase 0 before
trusting the default on a new dataset.

Repeated splits (R = 20, design §4.2/§4.3)
--------------------------------------------
:func:`three_way_scene_split` performs ONE calibration(40%)/matching(20%)/eval(40%)
split of the given scene ids; :func:`repeated_scene_splits` calls it ``R`` times with
split-seed = repeat index (0..R-1), stratifying by an optional per-scene label (e.g. a
scene-level boundary/square/interior theta-stratum majority label) where scene counts
permit. No scene appears in two parts of the same repeat by construction (each repeat
partitions the FULL scene id set exactly once).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "SplitError",
    "DEFAULT_SCENE_ID_REGEX",
    "scene_id_from_crop_filename",
    "three_way_scene_split",
    "repeated_scene_splits",
    "assert_scene_level_splits",
]

# Matches the mmrotate DOTA split-crop convention: <image_id>__<tile_size>__<x>___<y>
# (e.g. "P0006__1024__0___0"). Group 1 is the source scene id. Falls back to the whole
# stem when no "__" is present (DIOR-R / any single-tile dataset: scene == image).
DEFAULT_SCENE_ID_REGEX = re.compile(r"^([^_]+(?:_[^_]+)*?)(?:__\d+__.*)?$")


class SplitError(ValueError):
    """Raised on any scene-level-split precondition violation (design §4.2 C1 guard)."""


def scene_id_from_crop_filename(
    filename: str, scene_id_regex: "re.Pattern[str]" = DEFAULT_SCENE_ID_REGEX
) -> str:
    """Extract the source-scene id from a crop (or plain-image) filename stem.

    Strips a directory path and file extension first, then applies
    ``scene_id_regex``. Raises :class:`SplitError` if the regex does not match at all
    (should not happen with the default pattern, which always matches via its
    fallback group, but a custom ``scene_id_regex`` might not).
    """
    stem = filename.rsplit("/", 1)[-1]
    if "." in stem:
        stem = stem.rsplit(".", 1)[0]
    m = scene_id_regex.match(stem)
    if m is None or m.group(1) is None:
        raise SplitError(
            f"scene_id_from_crop_filename: pattern {scene_id_regex.pattern!r} did not "
            f"match filename stem {stem!r}"
        )
    return m.group(1)


def three_way_scene_split(
    scene_ids: Sequence[str],
    cal_frac: float = 0.4,
    match_frac: float = 0.2,
    seed: int = 0,
    strata: Optional[Sequence[Any]] = None,
) -> Dict[str, List[str]]:
    """One calibration/matching/eval partition of ``scene_ids`` (design §4.2).

    Parameters
    ----------
    scene_ids:
        Unique source-scene ids to partition (duplicates raise :class:`SplitError` --
        callers should de-duplicate at the crop-to-scene aggregation step upstream, not
        rely on this function to silently collapse them).
    cal_frac, match_frac:
        Calibration and matching fractions; eval gets the remainder
        (``1 - cal_frac - match_frac``, default 0.4/0.2/0.4 per design §4.2).
    seed:
        Split-seed (== repeat index under :func:`repeated_scene_splits`).
    strata:
        Optional per-scene stratum label, same length/order as ``scene_ids``
        (e.g. class or boundary/square/interior majority label); when given, the
        partition is done WITHIN each stratum separately (so class/stratum
        proportions are preserved across cal/match/eval), falling back to an
        unstratified split for any stratum with too few scenes to place at least one
        scene in each of the three parts (recorded in the returned ``degraded_strata``
        list rather than silently dropping the stratification).

    Returns
    -------
    dict
        ``calibration``, ``matching``, ``eval`` (lists of scene ids, each sorted for
        determinism-of-representation) and ``degraded_strata`` (list of stratum labels
        that fell back to unstratified placement).
    """
    scene_ids = list(scene_ids)
    if len(scene_ids) != len(set(scene_ids)):
        raise SplitError(
            "three_way_scene_split: duplicate scene ids passed in -- de-duplicate at "
            "the crop-to-scene aggregation step (this function operates on unique "
            "scenes only, design §4.2)"
        )
    if not 0.0 < cal_frac < 1.0 or not 0.0 < match_frac < 1.0 or cal_frac + match_frac >= 1.0:
        raise SplitError("three_way_scene_split: cal_frac + match_frac must be in (0, 1)")
    n = len(scene_ids)
    if n < 3:
        raise SplitError(
            f"three_way_scene_split: need >= 3 scenes for a 3-way split, got {n}"
        )

    rng = np.random.default_rng(seed)

    def _split_group(ids: List[str]) -> Dict[str, List[str]]:
        ids = sorted(ids)  # deterministic pre-shuffle order
        perm = rng.permutation(len(ids))
        shuffled = [ids[i] for i in perm]
        m = len(shuffled)
        n_cal = max(1, int(round(cal_frac * m))) if m >= 3 else 0
        n_match = max(1, int(round(match_frac * m))) if m >= 3 else 0
        # Guarantee at least one scene per part when m >= 3; clip overflow into eval.
        if n_cal + n_match >= m:
            n_cal = max(1, m - 2)
            n_match = 1
        cal = shuffled[:n_cal]
        match = shuffled[n_cal : n_cal + n_match]
        ev = shuffled[n_cal + n_match :]
        return {"calibration": cal, "matching": match, "eval": ev}

    if strata is None:
        parts = _split_group(scene_ids)
        degraded: List[Any] = []
    else:
        strata = list(strata)
        if len(strata) != n:
            raise SplitError("three_way_scene_split: strata must match scene_ids length")
        by_stratum: Dict[Any, List[str]] = {}
        for sid, st in zip(scene_ids, strata):
            by_stratum.setdefault(st, []).append(sid)

        cal_all: List[str] = []
        match_all: List[str] = []
        eval_all: List[str] = []
        degraded = []
        undersized: List[str] = []
        for st, ids in by_stratum.items():
            if len(ids) < 3:
                undersized.extend(ids)
                degraded.append(st)
                continue
            sub = _split_group(ids)
            cal_all.extend(sub["calibration"])
            match_all.extend(sub["matching"])
            eval_all.extend(sub["eval"])
        if undersized:
            sub = _split_group(undersized)
            cal_all.extend(sub["calibration"])
            match_all.extend(sub["matching"])
            eval_all.extend(sub["eval"])
        parts = {"calibration": cal_all, "matching": match_all, "eval": eval_all}

    return {
        "calibration": sorted(parts["calibration"]),
        "matching": sorted(parts["matching"]),
        "eval": sorted(parts["eval"]),
        "degraded_strata": degraded,
    }


def repeated_scene_splits(
    scene_ids: Sequence[str],
    n_repeats: int = 20,
    cal_frac: float = 0.4,
    match_frac: float = 0.2,
    strata: Optional[Sequence[Any]] = None,
) -> List[Dict[str, List[str]]]:
    """``n_repeats`` independent :func:`three_way_scene_split` calls, split-seed =
    repeat index (design §4.2/§4.3, R = 20 default). Returns a list of split dicts."""
    if n_repeats < 1:
        raise SplitError("repeated_scene_splits: n_repeats must be >= 1")
    return [
        three_way_scene_split(
            scene_ids, cal_frac=cal_frac, match_frac=match_frac, seed=r, strata=strata
        )
        for r in range(n_repeats)
    ]


def assert_scene_level_splits(
    records: Sequence[Dict[str, Any]], scene_field: str = "scene_id", id_field: str = "image_id"
) -> None:
    """Refusal guard: raise :class:`SplitError` if ``records`` (e.g. a detections
    table) has no resolvable scene id for calibration -- i.e. every row is missing
    ``scene_field`` -- so a caller cannot silently fall through to a crop-level split
    when scene ids are simply absent from the input (design's binding "no hardcoding /
    scene-level discipline in every certified path" rule)."""
    if not records:
        raise SplitError("assert_scene_level_splits: records must be non-empty")
    missing = [r.get(id_field, "<unknown>") for r in records if not r.get(scene_field)]
    if len(missing) == len(records):
        raise SplitError(
            f"assert_scene_level_splits: no record carries a '{scene_field}' -- "
            "calibration REFUSES to fall back to crop/image-level splitting (design "
            "§4.2 C1: crop-level splits destroy exchangeability by placing the same "
            "physical object's overlapping crops in different split parts). Populate "
            f"'{scene_field}' (e.g. via scene_id_from_crop_filename) before splitting."
        )

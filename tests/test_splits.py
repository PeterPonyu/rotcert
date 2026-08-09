"""Tests for rotcert.splits: scene-level (never crop-level) 3-way repeated splits."""

from __future__ import annotations

import pytest

from rotcert.splits import (
    SplitError,
    assert_scene_level_splits,
    repeated_scene_splits,
    scene_id_from_crop_filename,
    three_way_scene_split,
)


class TestSceneIdExtraction:
    def test_dota_crop_convention(self):
        assert scene_id_from_crop_filename("P0006__1024__0___0.png") == "P0006"

    def test_dota_crop_convention_with_path(self):
        assert scene_id_from_crop_filename("/data/dota/P1234__1024__200___400.png") == "P1234"

    def test_dior_r_single_tile_fallback(self):
        assert scene_id_from_crop_filename("00001.jpg") == "00001"

    def test_no_extension(self):
        assert scene_id_from_crop_filename("P0007__1024__0___200") == "P0007"


class TestThreeWaySceneSplit:
    def test_partition_covers_all_scenes_exactly_once(self):
        ids = [f"P{i:04d}" for i in range(100)]
        split = three_way_scene_split(ids, seed=0)
        all_parts = split["calibration"] + split["matching"] + split["eval"]
        assert sorted(all_parts) == sorted(ids)
        assert len(set(all_parts)) == len(all_parts)

    def test_approximate_fractions(self):
        ids = [f"P{i:04d}" for i in range(1000)]
        split = three_way_scene_split(ids, cal_frac=0.4, match_frac=0.2, seed=0)
        assert abs(len(split["calibration"]) - 400) < 20
        assert abs(len(split["matching"]) - 200) < 20
        assert abs(len(split["eval"]) - 400) < 20

    def test_duplicate_scene_ids_raises(self):
        with pytest.raises(SplitError):
            three_way_scene_split(["a", "a", "b"])

    def test_too_few_scenes_raises(self):
        with pytest.raises(SplitError):
            three_way_scene_split(["a", "b"])

    def test_bad_fracs_raise(self):
        with pytest.raises(SplitError):
            three_way_scene_split(["a", "b", "c", "d"], cal_frac=0.6, match_frac=0.6)

    def test_stratified_covers_all_scenes(self):
        ids = [f"P{i:04d}" for i in range(150)]
        strata = ["a" if i % 3 == 0 else "b" for i in range(150)]
        split = three_way_scene_split(ids, seed=1, strata=strata)
        all_parts = split["calibration"] + split["matching"] + split["eval"]
        assert sorted(all_parts) == sorted(ids)

    def test_undersized_stratum_degrades_gracefully(self):
        ids = [f"P{i:04d}" for i in range(20)]
        strata = ["rare"] * 2 + ["common"] * 18
        split = three_way_scene_split(ids, seed=2, strata=strata)
        assert "rare" in split["degraded_strata"]
        all_parts = split["calibration"] + split["matching"] + split["eval"]
        assert sorted(all_parts) == sorted(ids)


class TestRepeatedSceneSplits:
    def test_r_repeats_returned(self):
        ids = [f"P{i:04d}" for i in range(50)]
        reps = repeated_scene_splits(ids, n_repeats=20)
        assert len(reps) == 20

    def test_repeats_differ(self):
        ids = [f"P{i:04d}" for i in range(50)]
        reps = repeated_scene_splits(ids, n_repeats=5)
        cal_sets = [tuple(r["calibration"]) for r in reps]
        assert len(set(cal_sets)) > 1

    def test_every_repeat_is_a_full_partition(self):
        ids = [f"P{i:04d}" for i in range(50)]
        reps = repeated_scene_splits(ids, n_repeats=5)
        for r in reps:
            all_parts = r["calibration"] + r["matching"] + r["eval"]
            assert sorted(all_parts) == sorted(ids)

    def test_zero_repeats_raises(self):
        with pytest.raises(SplitError):
            repeated_scene_splits(["a", "b", "c"], n_repeats=0)


class TestAssertSceneLevelSplits:
    def test_refuses_when_no_scene_id_present(self):
        with pytest.raises(SplitError):
            assert_scene_level_splits([{"image_id": "x"}, {"image_id": "y"}])

    def test_passes_when_at_least_one_scene_id_present(self):
        assert_scene_level_splits([{"scene_id": "P1", "image_id": "x"}, {"image_id": "y"}])

    def test_empty_records_raises(self):
        with pytest.raises(SplitError):
            assert_scene_level_splits([])

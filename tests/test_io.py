"""Tests for rotcert.io: canonical detections/GT schema validation + scene-id
population (and the design's binding scene-level-discipline rule)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rotcert.io import (
    SchemaError,
    load_jsonl,
    populate_scene_ids,
    to_jsonable,
    validate_detection,
    validate_detections,
    validate_gt,
    validate_gts,
    write_jsonl,
)
from rotcert.splits import assert_scene_level_splits


class TestValidateDetection:
    def test_valid_record_normalized(self):
        rec = {"image_id": "P0006__1024__0___0", "class": "ship", "obb": [1, 2, 3, 4, 0.1], "score": 0.9}
        out = validate_detection(rec)
        assert out["scene_id"] is None
        assert out["obb"] == [1.0, 2.0, 3.0, 4.0, 0.1]

    def test_missing_field_raises(self):
        with pytest.raises(SchemaError):
            validate_detection({"image_id": "x", "class": "a", "obb": [1, 2, 3, 4, 0]})

    def test_bad_obb_length_raises(self):
        with pytest.raises(SchemaError):
            validate_detection({"image_id": "x", "class": "a", "obb": [1, 2, 3], "score": 0.5})

    def test_negative_wh_raises(self):
        with pytest.raises(SchemaError):
            validate_detection({"image_id": "x", "class": "a", "obb": [1, 2, -3, 4, 0], "score": 0.5})


class TestValidateGt:
    def test_valid_record(self):
        out = validate_gt({"image_id": "x", "class": "a", "obb": [1, 2, 3, 4, 0]})
        assert out["gt_id"] is None

    def test_missing_field_raises(self):
        with pytest.raises(SchemaError):
            validate_gt({"image_id": "x", "obb": [1, 2, 3, 4, 0]})


class TestValidateBatches:
    def test_validate_detections_empty_raises(self):
        with pytest.raises(SchemaError):
            validate_detections([])

    def test_validate_gts_empty_raises(self):
        with pytest.raises(SchemaError):
            validate_gts([])

    def test_batch_normalizes_all(self):
        recs = [{"image_id": f"x{i}", "class": "a", "obb": [1, 2, 3, 4, 0], "score": 0.5} for i in range(3)]
        out = validate_detections(recs)
        assert len(out) == 3


class TestPopulateSceneIds:
    def test_fills_from_dota_crop_convention(self):
        recs = [{"image_id": "P0006__1024__0___0", "class": "a", "obb": [1, 2, 3, 4, 0], "score": 0.5}]
        out = populate_scene_ids(recs)
        assert out[0]["scene_id"] == "P0006"

    def test_leaves_explicit_scene_id_untouched(self):
        recs = [{"image_id": "P0006__1024__0___0", "scene_id": "OVERRIDE", "class": "a", "obb": [1, 2, 3, 4, 0], "score": 0.5}]
        out = populate_scene_ids(recs)
        assert out[0]["scene_id"] == "OVERRIDE"


class TestJsonlRoundtrip:
    def test_write_and_load(self, tmp_path: Path):
        recs = [{"a": 1, "b": [1.0, 2.0]}, {"a": 2, "b": [3.0]}]
        path = tmp_path / "out.jsonl"
        write_jsonl(path, recs)
        loaded = load_jsonl(path)
        assert loaded == recs

    def test_load_empty_raises(self, tmp_path: Path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        with pytest.raises(SchemaError):
            load_jsonl(path)

    def test_load_bad_json_raises(self, tmp_path: Path):
        path = tmp_path / "bad.jsonl"
        path.write_text("{not valid json}\n")
        with pytest.raises(SchemaError):
            load_jsonl(path)

    def test_to_jsonable_numpy_types(self):
        import numpy as np

        obj = {"a": np.float64(1.5), "b": np.array([1, 2, 3]), "c": np.bool_(True)}
        out = to_jsonable(obj)
        assert out == {"a": 1.5, "b": [1, 2, 3], "c": True}


class TestSceneLevelDisciplineRefusal:
    """Binding rule: calibration REFUSES crop-level splits when scene ids are
    available (i.e. when scene ids are entirely ABSENT, the tool must not silently
    fall back to image_id-as-scene without the caller explicitly populating it)."""

    def test_detections_without_scene_id_refuse_split_level_assertion(self):
        dets = validate_detections(
            [{"image_id": "P0006__1024__0___0", "class": "a", "obb": [1, 2, 3, 4, 0], "score": 0.9}]
        )
        # validate_detection alone does NOT populate scene_id -- assert_scene_level_splits
        # must catch the still-missing scene_id and refuse rather than silently treating
        # image_id (a CROP id) as the scene.
        with pytest.raises(Exception):
            assert_scene_level_splits(dets)

    def test_after_populate_scene_ids_split_assertion_passes(self):
        dets = validate_detections(
            [{"image_id": "P0006__1024__0___0", "class": "a", "obb": [1, 2, 3, 4, 0], "score": 0.9}]
        )
        dets = populate_scene_ids(dets)
        assert_scene_level_splits(dets)  # should not raise

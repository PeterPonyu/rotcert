"""Tests for rotcert.matching: rotated/hull IoU + the preregistered greedy rule."""

from __future__ import annotations

import numpy as np
import pytest

from rotcert.matching import greedy_match, hull_iou, iou_matrix, obb_to_polygon, rotated_iou


class TestPolygonAndIoU:
    def test_self_iou_is_one(self):
        obb = [10.0, 10.0, 20.0, 10.0, 0.3]
        assert rotated_iou(obb, obb) == pytest.approx(1.0)

    def test_disjoint_boxes_iou_zero(self):
        a = [0.0, 0.0, 5.0, 5.0, 0.0]
        b = [1000.0, 1000.0, 5.0, 5.0, 0.0]
        assert rotated_iou(a, b) == pytest.approx(0.0)

    def test_polygon_area_matches_box_area(self):
        obb = [0.0, 0.0, 20.0, 10.0, 0.0]
        poly = obb_to_polygon(obb)
        assert poly.area == pytest.approx(200.0)

    def test_polygon_area_invariant_to_rotation(self):
        for theta_deg in [0, 15, 45, 89]:
            poly = obb_to_polygon([0.0, 0.0, 20.0, 10.0, np.deg2rad(theta_deg)])
            assert poly.area == pytest.approx(200.0, abs=1e-6)

    def test_zero_area_box_gives_zero_iou(self):
        a = [0.0, 0.0, 0.0, 0.0, 0.0]
        b = [0.0, 0.0, 10.0, 10.0, 0.0]
        assert rotated_iou(a, b) == 0.0

    def test_hull_iou_of_axis_aligned_box_equals_rotated_iou(self):
        # For an already-axis-aligned box, hull == polygon, so hull_iou(A,A)==1.
        a = [0.0, 0.0, 20.0, 10.0, 0.0]
        assert hull_iou(a, a) == pytest.approx(1.0)

    def test_iou_matrix_shape_and_diagonal(self):
        preds = np.array([[0.0, 0.0, 10.0, 10.0, 0.0], [50.0, 50.0, 10.0, 10.0, 0.0]])
        gts = np.array([[0.0, 0.0, 10.0, 10.0, 0.0], [50.0, 50.0, 10.0, 10.0, 0.0]])
        m = iou_matrix(preds, gts, iou_metric="rotated")
        assert m.shape == (2, 2)
        assert m[0, 0] == pytest.approx(1.0)
        assert m[1, 1] == pytest.approx(1.0)
        assert m[0, 1] == pytest.approx(0.0)

    def test_iou_matrix_bad_metric_raises(self):
        preds = np.array([[0.0, 0.0, 10.0, 10.0, 0.0]])
        gts = np.array([[0.0, 0.0, 10.0, 10.0, 0.0]])
        with pytest.raises(ValueError):
            iou_matrix(preds, gts, iou_metric="bogus")


class TestGreedyMatch:
    def _mk(self, obb, score, cls):
        return {"obb": obb, "score": score, "class": cls}

    def test_simple_one_to_one_match(self):
        dets = [self._mk([10, 10, 20, 10, 0.0], 0.9, "a")]
        gts = [{"obb": [10, 10, 20, 10, 0.0], "class": "a"}]
        m = greedy_match(dets, gts, iou_thr=0.5)
        assert len(m["matches"]) == 1
        assert m["matches"][0] == {"det_index": 0, "gt_index": 0, "iou": pytest.approx(1.0)}
        assert m["unmatched_det_indices"] == []
        assert m["unmatched_gt_indices"] == []

    def test_class_mismatch_no_match(self):
        dets = [self._mk([10, 10, 20, 10, 0.0], 0.9, "a")]
        gts = [{"obb": [10, 10, 20, 10, 0.0], "class": "b"}]
        m = greedy_match(dets, gts, iou_thr=0.5)
        assert m["matches"] == []
        assert m["unmatched_det_indices"] == [0]
        assert m["unmatched_gt_indices"] == [0]

    def test_below_threshold_no_match(self):
        dets = [self._mk([0, 0, 5, 5, 0.0], 0.9, "a")]
        gts = [{"obb": [100, 100, 5, 5, 0.0], "class": "a"}]
        m = greedy_match(dets, gts, iou_thr=0.5)
        assert m["matches"] == []
        assert m["unmatched_det_indices"] == [0]
        assert m["unmatched_gt_indices"] == [0]

    def test_confidence_priority_higher_score_matches_first(self):
        # Two detections overlapping the SAME gt reasonably; higher-score det should
        # win the match, lower-score det becomes an unmatched FP.
        gts = [{"obb": [10, 10, 20, 10, 0.0], "class": "a"}]
        dets = [
            self._mk([10.5, 10.5, 19, 9, 0.0], 0.4, "a"),
            self._mk([10, 10, 20, 10, 0.0], 0.95, "a"),
        ]
        m = greedy_match(dets, gts, iou_thr=0.3)
        assert len(m["matches"]) == 1
        assert m["matches"][0]["det_index"] == 1  # the higher-confidence det
        assert m["unmatched_det_indices"] == [0]

    def test_one_gt_matches_at_most_one_det(self):
        gts = [{"obb": [10, 10, 20, 10, 0.0], "class": "a"}]
        dets = [
            self._mk([10, 10, 20, 10, 0.0], 0.9, "a"),
            self._mk([10, 10, 20, 10, 0.0], 0.8, "a"),
        ]
        m = greedy_match(dets, gts, iou_thr=0.5)
        assert len(m["matches"]) == 1
        assert m["unmatched_det_indices"] == [1]

    def test_deterministic_tie_break_by_gt_index(self):
        # Two GTs with identical IoU to one detection; lower gt_index should win.
        gts = [
            {"obb": [10, 10, 20, 10, 0.0], "class": "a"},
            {"obb": [10, 10, 20, 10, 0.0], "class": "a"},
        ]
        dets = [self._mk([10, 10, 20, 10, 0.0], 0.9, "a")]
        m = greedy_match(dets, gts, iou_thr=0.5)
        assert m["matches"][0]["gt_index"] == 0

    def test_hull_metric_path_runs(self):
        gts = [{"obb": [10, 10, 20, 10, np.deg2rad(40)], "class": "a"}]
        dets = [self._mk([10, 10, 20, 10, np.deg2rad(40)], 0.9, "a")]
        m = greedy_match(dets, gts, iou_thr=0.5, iou_metric="hull")
        assert len(m["matches"]) == 1

    def test_bad_iou_thr_raises(self):
        with pytest.raises(ValueError):
            greedy_match([], [], iou_thr=1.5)

    def test_empty_dets_all_gt_unmatched(self):
        gts = [{"obb": [10, 10, 20, 10, 0.0], "class": "a"}]
        m = greedy_match([], gts, iou_thr=0.5)
        assert m["matches"] == []
        assert m["unmatched_gt_indices"] == [0]

    def test_empty_gts_all_det_unmatched(self):
        dets = [self._mk([10, 10, 20, 10, 0.0], 0.9, "a")]
        m = greedy_match(dets, [], iou_thr=0.5)
        assert m["matches"] == []
        assert m["unmatched_det_indices"] == [0]

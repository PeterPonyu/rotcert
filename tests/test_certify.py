"""Tests for rotcert.certify: G1 per-Mondrian-cell calibration, G2 certified FNR, and
the honest-uncertainty refusal rules (design §3.3)."""

from __future__ import annotations

import numpy as np
import pytest

from rotcert.certify import (
    CertifyError,
    g1_calibrate,
    g1_coverage,
    g2_certify_fnr,
    g2_certify_fnr_mondrian,
    image_risk_matrix,
)


def _matched(gen_matched_pairs, rng, n, cls="ship"):
    preds, gts = gen_matched_pairs(rng, n)
    return [{"pred_obb": p.tolist(), "gt_obb": g.tolist(), "class": cls} for p, g in zip(preds, gts)]


class TestG1Calibrate:
    def test_unknown_score_raises(self, gen_matched_pairs, rng):
        with pytest.raises(CertifyError):
            g1_calibrate(_matched(gen_matched_pairs, rng, 20), "bogus", alpha=0.1)

    def test_empty_matched_raises(self):
        with pytest.raises(CertifyError):
            g1_calibrate([], "gwd", alpha=0.1)

    def test_marginal_cell_coverage_near_target(self, gen_matched_pairs, rng):
        cal = _matched(gen_matched_pairs, rng, 800)
        ev = _matched(gen_matched_pairs, rng, 2000)
        cert = g1_calibrate(cal, "gwd", alpha=0.1)
        assert cert["refused"] == []
        cov = g1_coverage(cert, ev)
        assert abs(cov["overall_coverage"] - 0.9) < 0.04

    def test_mondrian_stratification_by_class(self, gen_matched_pairs, rng):
        cal_a = _matched(gen_matched_pairs, rng, 400, cls="ship")
        cal_b = _matched(gen_matched_pairs, rng, 400, cls="harbor")
        cert = g1_calibrate(cal_a + cal_b, "gwd", alpha=0.1, mondrian_field="class")
        assert set(cert["strata"].keys()) == {"ship", "harbor"}

    def test_small_n_stratum_refused(self, gen_matched_pairs, rng):
        cal = _matched(gen_matched_pairs, rng, 5)
        cert = g1_calibrate(cal, "gwd", alpha=0.1, mondrian_field="class")
        assert len(cert["refused"]) == 1
        assert cert["refused"][0]["alpha_min"] > 0.1

    def test_bonferroni_score_calibrates(self, gen_matched_pairs, rng):
        cal = _matched(gen_matched_pairs, rng, 300)
        cert = g1_calibrate(cal, "naive-coord", alpha=0.1)
        assert cert["strata"][None]["set_size_cxcy"] is not None

    def test_out_of_support_eval_rows_counted(self, gen_matched_pairs, rng):
        cal = _matched(gen_matched_pairs, rng, 300, cls="ship")
        ev_other = _matched(gen_matched_pairs, rng, 50, cls="never-seen")
        cert = g1_calibrate(cal, "gwd", alpha=0.1, mondrian_field="class")
        cov = g1_coverage(cert, ev_other)
        assert cov["n_out_of_support"] == 50


class TestImageRiskMatrix:
    def test_all_matched_gives_zero_risk_at_lambda_zero(self):
        scenes = [[0.9, 0.8, 0.95]]
        grid = [0.0, 0.5, 1.0]
        risk = image_risk_matrix(scenes, grid)
        assert risk[0, 0] == pytest.approx(0.0)

    def test_all_missed_gives_risk_one(self):
        scenes = [[None, None]]
        grid = [0.0, 0.5]
        risk = image_risk_matrix(scenes, grid)
        assert np.all(risk == 1.0)

    def test_empty_gt_scene_raises(self):
        with pytest.raises(CertifyError):
            image_risk_matrix([[]], [0.0, 0.5])

    def test_monotone_nondecreasing_in_lambda(self):
        rng = np.random.default_rng(0)
        scenes = [[float(rng.uniform(0, 1)) if rng.random() > 0.1 else None for _ in range(10)] for _ in range(50)]
        grid = np.linspace(0, 1, 20)
        risk = image_risk_matrix(scenes, grid)
        assert np.all(np.diff(risk, axis=1) >= -1e-12)


class TestG2CertifyFnr:
    def _gen_scenes(self, rng, n_scenes, n_gt=5, miss_rate=0.05):
        scenes = []
        for _ in range(n_scenes):
            confs = [None if rng.random() < miss_rate else float(rng.uniform(0.2, 0.99)) for _ in range(n_gt)]
            scenes.append(confs)
        return scenes

    def test_certifies_with_enough_images(self):
        rng = np.random.default_rng(0)
        scenes = self._gen_scenes(rng, 500)
        res = g2_certify_fnr(scenes, beta=0.2, delta=0.05)
        assert res["certified"]
        assert not res["refused"]

    def test_refuses_below_power_floor(self):
        rng = np.random.default_rng(1)
        scenes = self._gen_scenes(rng, 5)
        res = g2_certify_fnr(scenes, beta=0.2, delta=0.05)
        assert res["refused"]
        assert res["lambda_star"] is None
        assert "power floor" in res["reason"]

    def test_empty_scenes_raises(self):
        with pytest.raises(CertifyError):
            g2_certify_fnr([], beta=0.2, delta=0.05)

    def test_all_unconditional_miss_raises(self):
        with pytest.raises(CertifyError):
            g2_certify_fnr([[None, None, None]] * 500, beta=0.2, delta=0.05)


class TestG2Mondrian:
    def test_pooled_fallback_present_for_refused_class(self):
        rng = np.random.default_rng(0)

        def gen(n):
            return [[None if rng.random() < 0.05 else float(rng.uniform(0.2, 0.99)) for _ in range(5)] for _ in range(n)]

        by_class = {"common": gen(400), "rare": gen(5)}
        res = g2_certify_fnr_mondrian(by_class, beta=0.2, delta=0.05)
        assert res["per_class"]["common"]["certified"]
        assert res["per_class"]["rare"]["refused"]
        assert res["pooled_marginal"]["certified"]
        assert res["n_classes_certified"] == 1

    def test_empty_raises(self):
        with pytest.raises(CertifyError):
            g2_certify_fnr_mondrian({}, beta=0.2, delta=0.05)

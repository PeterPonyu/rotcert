"""Tests for rotcert.ltt: HB/EB p-values, the power floor arithmetic, and the
per-image risk-matrix LTT certificate."""

from __future__ import annotations

import numpy as np
import pytest

from rotcert.ltt import build_lambda_grid, eb_pvalue, hb_pvalue, ltt_certify_matrix, power_floor_n_img


class TestPValues:
    def test_hb_pvalue_high_when_mean_below_alpha(self):
        y = np.full(500, 0.01)
        assert hb_pvalue(y, alpha=0.2) < 0.01

    def test_hb_pvalue_near_one_when_mean_above_alpha(self):
        y = np.full(500, 0.5)
        assert hb_pvalue(y, alpha=0.2) == pytest.approx(1.0)

    def test_eb_pvalue_tighter_than_hb_low_variance(self):
        rng = np.random.default_rng(0)
        y = np.clip(0.01 + rng.normal(scale=0.001, size=2000), 0, 1)
        p_hb = hb_pvalue(y, alpha=0.2)
        p_eb = eb_pvalue(y, alpha=0.2)
        assert p_eb <= p_hb + 1e-9

    def test_eb_pvalue_one_when_n_is_one(self):
        assert eb_pvalue(np.array([0.01]), alpha=0.2) == 1.0

    def test_out_of_bounds_raises(self):
        with pytest.raises(ValueError):
            hb_pvalue(np.array([1.5, 0.2]), alpha=0.2)
        with pytest.raises(ValueError):
            eb_pvalue(np.array([-0.1, 0.2]), alpha=0.2)

    def test_bad_alpha_raises(self):
        with pytest.raises(ValueError):
            hb_pvalue(np.array([0.1, 0.2]), alpha=1.5)


class TestBuildLambdaGrid:
    def test_grid_ascending_unique(self):
        rng = np.random.default_rng(0)
        scores = rng.uniform(0, 1, 1000)
        grid = build_lambda_grid(scores, n_grid=50)
        assert np.all(np.diff(grid) >= 0)
        assert len(np.unique(grid)) == len(grid)

    def test_caps_below_full_max(self):
        scores = np.arange(100, dtype=float)
        grid = build_lambda_grid(scores, n_grid=50, min_accept_frac=0.1)
        assert grid.max() < scores.max()


class TestPowerFloor:
    def test_matches_design_worked_examples(self):
        # design §2.4: G ~= 50 thresholds, delta=0.05 -> ln(G/delta) = ln(1000) ~= 6.9;
        # headroom .15/.10/.05 -> Hoeffding ~155/345/1380, Bentkus ~90/180/700.
        pf15 = power_floor_n_img(beta=0.20, delta=0.05, grid_size=50, r_hat=0.05)
        pf10 = power_floor_n_img(beta=0.20, delta=0.05, grid_size=50, r_hat=0.10)
        pf05 = power_floor_n_img(beta=0.20, delta=0.05, grid_size=50, r_hat=0.15)
        assert 140 <= pf15["hoeffding_floor"] <= 175
        assert 320 <= pf10["hoeffding_floor"] <= 380
        assert 1300 <= pf05["hoeffding_floor"] <= 1450
        assert pf15["bentkus_floor"] < pf15["hoeffding_floor"]

    def test_infinite_when_no_headroom(self):
        pf = power_floor_n_img(beta=0.1, delta=0.05, grid_size=100, r_hat=0.2)
        assert pf["hoeffding_floor"] == float("inf")
        assert pf["bentkus_floor"] == float("inf")

    def test_bad_inputs_raise(self):
        with pytest.raises(ValueError):
            power_floor_n_img(beta=1.5, delta=0.05, grid_size=10, r_hat=0.1)
        with pytest.raises(ValueError):
            power_floor_n_img(beta=0.2, delta=0.05, grid_size=0, r_hat=0.1)


class TestLttCertifyMatrix:
    def test_certifies_when_risk_controlled(self):
        rng = np.random.default_rng(0)
        n_img, K = 400, 25
        grid = np.linspace(0.1, 0.9, K)
        risk = np.clip(0.02 + grid[None, :] * 0.15 + rng.normal(scale=0.02, size=(n_img, K)), 0, 1)
        res = ltt_certify_matrix(risk, grid, beta=0.20, delta=0.05)
        assert res["certified"]
        assert res["realized_risk"] <= 0.20 + 0.03

    def test_vacuous_when_risk_too_high(self):
        rng = np.random.default_rng(1)
        n_img, K = 400, 25
        grid = np.linspace(0.1, 0.9, K)
        risk = np.clip(0.6 + rng.normal(scale=0.05, size=(n_img, K)), 0, 1)
        res = ltt_certify_matrix(risk, grid, beta=0.20, delta=0.05)
        assert not res["certified"]
        assert res["lambda_star"] is None

    def test_fixed_sequence_procedure_runs(self):
        rng = np.random.default_rng(2)
        n_img, K = 300, 20
        grid = np.linspace(0.1, 0.9, K)
        risk = np.clip(0.02 + grid[None, :] * 0.1 + rng.normal(scale=0.02, size=(n_img, K)), 0, 1)
        res = ltt_certify_matrix(risk, grid, beta=0.20, delta=0.05, procedure="fixed-sequence")
        assert "certified" in res

    def test_shape_mismatch_raises(self):
        risk = np.zeros((10, 5))
        with pytest.raises(ValueError):
            ltt_certify_matrix(risk, np.linspace(0, 1, 4), beta=0.2, delta=0.05)

    def test_out_of_bounds_risk_raises(self):
        risk = np.full((5, 3), 1.5)
        with pytest.raises(ValueError):
            ltt_certify_matrix(risk, np.linspace(0, 1, 3), beta=0.2, delta=0.05)

    def test_hb_pvalue_variant_runs(self):
        rng = np.random.default_rng(3)
        n_img, K = 300, 20
        grid = np.linspace(0.1, 0.9, K)
        risk = np.clip(0.02 + grid[None, :] * 0.1 + rng.normal(scale=0.02, size=(n_img, K)), 0, 1)
        res = ltt_certify_matrix(risk, grid, beta=0.20, delta=0.05, p_value="hb")
        assert "certified" in res

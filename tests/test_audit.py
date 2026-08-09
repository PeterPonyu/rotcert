"""Tests for rotcert.audit: theta-stratum classification, scene-clustered bootstrap
coverage CIs, the confirmatory Holm-8, and K1 premise-death."""

from __future__ import annotations

import numpy as np
import pytest

from rotcert.audit import (
    classify_theta_stratum,
    coverage_ci,
    holm8_confirmatory,
    k1_premise_death,
    set_size_contrast,
    stratum_coverage_table,
)


class TestClassifyThetaStratum:
    def test_boundary(self):
        assert classify_theta_stratum(20, 10, np.deg2rad(88)) == "boundary"
        assert classify_theta_stratum(20, 10, np.deg2rad(-89)) == "boundary"

    def test_interior(self):
        assert classify_theta_stratum(20, 10, np.deg2rad(0)) == "interior"
        assert classify_theta_stratum(20, 10, np.deg2rad(40)) == "interior"

    def test_square_takes_priority_over_boundary(self):
        assert classify_theta_stratum(20, 19.5, np.deg2rad(89)) == "square"

    def test_nonpositive_w_raises(self):
        with pytest.raises(ValueError):
            classify_theta_stratum(0, 0, 0.0)


class TestCoverageCi:
    def test_ci_contains_point(self):
        rng = np.random.default_rng(0)
        covered = rng.random(500) < 0.9
        scenes = rng.integers(0, 40, 500).astype(str)
        ci = coverage_ci(covered, scenes, n_boot=500)
        assert ci["ci"][0] <= ci["point"] <= ci["ci"][1]
        assert ci["n_scenes"] <= 40

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            coverage_ci([], [])

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            coverage_ci([True, False], ["a"])


class TestStratumCoverageTable:
    def test_per_stratum_breakdown(self):
        rng = np.random.default_rng(0)
        rows = []
        for i in range(300):
            st = rng.choice(["boundary", "square", "interior"])
            rows.append({"covered": bool(rng.random() < 0.9), "theta_stratum": st, "scene_id": str(rng.integers(0, 30))})
        table = stratum_coverage_table(rows, n_boot=300)
        assert set(table.keys()) <= {"boundary", "square", "interior"}
        for st, ci in table.items():
            assert ci["ci"][0] <= ci["point"] <= ci["ci"][1]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            stratum_coverage_table([])


class TestSetSizeContrast:
    def test_detects_systematic_smaller_primary(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(8, 12, 20)
        b = rng.uniform(15, 25, 20)
        labels = [f"c{i}" for i in range(20)]
        res = set_size_contrast(a, b, labels, n_perm=1000, seed=0)
        assert res["ratio"] < 1.0
        assert res["p_value"] < 0.05

    def test_no_difference_gives_high_p(self):
        rng = np.random.default_rng(1)
        a = rng.uniform(10, 20, 20)
        b = rng.uniform(10, 20, 20)
        labels = [f"c{i}" for i in range(20)]
        res = set_size_contrast(a, b, labels, n_perm=1000, seed=1)
        assert res["p_value"] > 0.05

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError):
            set_size_contrast([1, 2], [1, 2, 3], ["a", "b", "c"])

    def test_nonpositive_raises(self):
        with pytest.raises(ValueError):
            set_size_contrast([0, 1], [1, 1], ["a", "b"])


class TestHolm8Confirmatory:
    def test_family_size_matches_roster(self):
        cells = [{"p_value": p} for p in [0.001, 0.01, 0.2, 0.5, 0.03, 0.04, 0.9, 0.001]]
        res = holm8_confirmatory(cells)
        assert res["family_size"] == 8
        assert len(res["results"]) == 8

    def test_family_size_is_never_hardcoded(self):
        cells = [{"p_value": p} for p in [0.001, 0.5, 0.9]]
        res = holm8_confirmatory(cells)
        assert res["family_size"] == 3

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            holm8_confirmatory([])


class TestK1PremiseDeath:
    def test_gwd_wins_no_premise_death(self):
        holm = [{"ratio": 0.7, "reject_holm": True} for _ in range(8)]
        res = k1_premise_death(boundary_square_gap_pp=0.5, holm_results=holm, b1_gap_pp=8.0)
        assert not res["premise_death"]

    def test_null_scenario_is_premise_death(self):
        holm = [{"ratio": 0.99, "reject_holm": False} for _ in range(8)]
        res = k1_premise_death(boundary_square_gap_pp=0.5, holm_results=holm, b1_gap_pp=1.0)
        assert res["premise_death"]

    def test_gwd_undercovers_blocks_premise_death_even_if_no_inflation(self):
        # If GWD itself fails the boundary/square gap check, that's a DIFFERENT
        # failure mode (K2 coverage sanity), not K1 premise-death -- the function
        # should not call it premise-death just because B1 also looks fine.
        holm = [{"ratio": 0.99, "reject_holm": False} for _ in range(8)]
        res = k1_premise_death(boundary_square_gap_pp=5.0, holm_results=holm, b1_gap_pp=1.0)
        assert not res["gwd_holds"]
        assert not res["premise_death"]

    def test_empty_holm_results_raises(self):
        with pytest.raises(ValueError):
            k1_premise_death(0.5, [], b1_gap_pp=1.0)

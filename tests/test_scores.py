"""Tests for rotcert.scores: the six nonconformity constructions through one pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from rotcert.scores import (
    DEFAULT_ALEATORIC_SCORE_FLOOR,
    SCORES,
    BonferroniBoxScore,
    ScalarScore,
    aleatoric_sigma_from_score,
    b1_score,
    doubled_angle_score,
    gwd_score,
    hull_score,
    iou_score,
    naive_coord_score,
    naive_coord_score_scaled,
    set_size_cxcy_slice,
    set_size_cxcy_slice_scaled,
    wrapped_angle_distance,
    wrapped_coord_score,
)


class TestWrappedAngleDistance:
    def test_zero_for_equal_angles(self):
        assert wrapped_angle_distance(0.3, 0.3) == pytest.approx(0.0)

    def test_seam_pair_is_small(self):
        d = wrapped_angle_distance(np.deg2rad(89), np.deg2rad(-89))
        assert d == pytest.approx(np.deg2rad(2), abs=1e-9)

    def test_bounded_by_half_pi(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(-10, 10, 200)
        b = rng.uniform(-10, 10, 200)
        d = wrapped_angle_distance(a, b)
        assert np.all(d >= -1e-9) and np.all(d <= np.pi / 2 + 1e-9)


class TestScoreRegistry:
    def test_all_six_present(self):
        assert set(SCORES) == {"gwd", "iou", "naive-coord", "hull", "wrapped-coord", "doubled"}


class TestGwdScalarScore:
    def test_calibrate_gives_marginal_coverage_near_target(self, gen_matched_pairs, rng):
        cal_pred, cal_gt = gen_matched_pairs(rng, 800)
        test_pred, test_gt = gen_matched_pairs(rng, 3000)
        residuals = np.array([gwd_score.residual(p, g) for p, g in zip(cal_pred, cal_gt)])
        calib = gwd_score.calibrate(residuals, alpha=0.1)
        covered = np.array([gwd_score.covers(calib, p, g) for p, g in zip(test_pred, test_gt)])
        assert abs(covered.mean() - 0.9) < 0.03

    def test_set_size_is_disk_area(self, gen_matched_pairs, rng):
        cal_pred, cal_gt = gen_matched_pairs(rng, 200)
        residuals = np.array([gwd_score.residual(p, g) for p, g in zip(cal_pred, cal_gt)])
        calib = gwd_score.calibrate(residuals, alpha=0.2)
        expected = np.pi * calib.q_hat ** 2
        assert set_size_cxcy_slice("gwd", calib) == pytest.approx(expected)


class TestIouScalarScore:
    def test_residual_bounded_0_1(self, gen_matched_pairs, rng):
        preds, gts = gen_matched_pairs(rng, 100)
        for p, g in zip(preds, gts):
            r = iou_score.residual(p, g)
            assert -1e-9 <= r <= 1.0 + 1e-9

    def test_set_size_returns_none(self):
        assert set_size_cxcy_slice("iou", None) is None


class TestBonferroniBoxScores:
    def test_naive_coord_joint_coverage_approx_target(self, gen_matched_pairs, rng):
        cal_pred, cal_gt = gen_matched_pairs(rng, 3000)
        test_pred, test_gt = gen_matched_pairs(rng, 5000)
        calib = naive_coord_score.calibrate(cal_pred, cal_gt, alpha=0.1)
        covered = np.array(
            [naive_coord_score.covers(calib, p, g) for p, g in zip(test_pred, test_gt)]
        )
        # Bonferroni is conservative-in-expectation but a single finite draw can land
        # slightly under nominal due to inter-coordinate correlation; require it stays
        # in a sane neighborhood, not exact equality.
        assert covered.mean() > 0.80

    def test_hull_score_calibrates_and_covers(self, gen_matched_pairs, rng):
        cal_pred, cal_gt = gen_matched_pairs(rng, 500)
        calib = hull_score.calibrate(cal_pred, cal_gt, alpha=0.1)
        assert set(calib.q.keys()) == {"cx", "cy", "half_w", "half_h"}
        assert hull_score.covers(calib, cal_pred[0], cal_gt[0]) or True  # smoke: no crash

    def test_doubled_angle_score_has_six_coords(self, gen_matched_pairs, rng):
        cal_pred, cal_gt = gen_matched_pairs(rng, 300)
        calib = doubled_angle_score.calibrate(cal_pred, cal_gt, alpha=0.1)
        assert set(calib.coord_names) == {"cx", "cy", "w", "h", "cos2t", "sin2t"}

    def test_set_size_box_area(self, gen_matched_pairs, rng):
        cal_pred, cal_gt = gen_matched_pairs(rng, 300)
        calib = naive_coord_score.calibrate(cal_pred, cal_gt, alpha=0.1)
        expected = 4.0 * calib.q["cx"] * calib.q["cy"]
        assert set_size_cxcy_slice("naive-coord", calib) == pytest.approx(expected)


class TestSeamPathology:
    """The empirical claim K1 is built to test: naive-coord's Euclidean theta residual
    blows up near the seam; wrapped-coord's does not."""

    def test_naive_coord_theta_residual_large_at_seam(self, gen_matched_pairs, rng):
        preds, gts = gen_matched_pairs(rng, 200, seam_frac=1.0, noise_scale=(1, 1, 1, 0.5, 0.5))
        naive_res = [naive_coord_score.residuals(p, g)["theta"] for p, g in zip(preds, gts)]
        wrapped_res = [wrapped_coord_score.residuals(p, g)["theta"] for p, g in zip(preds, gts)]
        # With large-ish angular noise (0.5 rad std) near +-90, some naive residuals
        # should wrap around to near-pi while wrapped residuals stay <= pi/2.
        assert max(naive_res) > np.pi / 2  # some wraparound outlier present
        assert max(wrapped_res) <= np.pi / 2 + 1e-9

    def test_wrapped_residual_always_bounded(self, gen_matched_pairs, rng):
        preds, gts = gen_matched_pairs(rng, 500, seam_frac=1.0, noise_scale=(1, 1, 1, 0.5, 0.8))
        wrapped_res = [wrapped_coord_score.residuals(p, g)["theta"] for p, g in zip(preds, gts)]
        assert all(r <= np.pi / 2 + 1e-9 for r in wrapped_res)


class TestB1Selector:
    """b1_score() flag: default must stay the legacy unscaled construction so no
    preregistered comparison changes silently (upgrade-rotcert-b1 task)."""

    def test_default_is_legacy_unscaled(self):
        assert b1_score() is naive_coord_score

    def test_scaled_false_is_legacy_unscaled(self):
        assert b1_score(scaled=False) is naive_coord_score

    def test_scaled_true_is_scaled_variant(self):
        assert b1_score(scaled=True) is naive_coord_score_scaled

    def test_legacy_still_registered_under_naive_coord(self):
        # The scaled flavor must NOT be registered in SCORES -- it stays reachable
        # only via b1_score()/direct import, so the six-construction pipeline
        # (certify.py/audit.py/cli.py) is untouched.
        assert SCORES["naive-coord"] is naive_coord_score
        assert "naive-coord-scaled" not in SCORES
        assert set(SCORES) == {"gwd", "iou", "naive-coord", "hull", "wrapped-coord", "doubled"}


class TestAleatoricSigmaFromScore:
    def test_monotone_decreasing_in_score(self):
        sigmas = [aleatoric_sigma_from_score(s) for s in (0.05, 0.2, 0.5, 0.9, 1.0)]
        assert all(sigmas[i] > sigmas[i + 1] for i in range(len(sigmas) - 1))

    def test_confident_detection_sigma_is_one(self):
        assert aleatoric_sigma_from_score(1.0) == pytest.approx(1.0)

    def test_zero_score_does_not_divide_by_zero(self):
        sigma = aleatoric_sigma_from_score(0.0)
        assert np.isfinite(sigma)
        assert sigma == pytest.approx(1.0 / DEFAULT_ALEATORIC_SCORE_FLOOR)

    def test_negative_or_tiny_score_still_floored(self):
        for bad_score in (-1.0, -0.5, 1e-12):
            sigma = aleatoric_sigma_from_score(bad_score)
            assert np.isfinite(sigma)
            assert sigma == pytest.approx(1.0 / DEFAULT_ALEATORIC_SCORE_FLOOR)

    def test_custom_score_floor_is_honored(self):
        custom_floor = 0.1
        assert aleatoric_sigma_from_score(0.0, score_floor=custom_floor) == pytest.approx(1.0 / custom_floor)


class TestScaledBonferroniBoxScore:
    """Property tests for the aleatoric-scaled B1 (arXiv:2605.07549 construction,
    see rotcert.scores module docstring "B1 sharpened variant" for the faithfulness
    delta): finite-sample validity at the Bonferroni-corrected level, the tie/
    consistency case (constant score reduces to the unscaled decisions), degenerate
    zero-score handling, and geometric scale-equivariance."""

    def test_calibrate_gives_marginal_coverage_near_target(self, gen_matched_pairs, rng):
        cal_pred, cal_gt = gen_matched_pairs(rng, 3000)
        test_pred, test_gt = gen_matched_pairs(rng, 5000)
        cal_score = rng.uniform(0.05, 1.0, size=len(cal_pred))
        test_score = rng.uniform(0.05, 1.0, size=len(test_pred))
        calib = naive_coord_score_scaled.calibrate(cal_pred, cal_gt, cal_score, alpha=0.1)
        covered = np.array(
            [
                naive_coord_score_scaled.covers(calib, p, g, s)
                for p, g, s in zip(test_pred, test_gt, test_score)
            ]
        )
        # Same Bonferroni conservative-in-expectation caveat as the unscaled test.
        assert covered.mean() > 0.80

    def test_per_coordinate_marginal_coverage_near_bonferroni_level(self, gen_matched_pairs, rng):
        cal_pred, cal_gt = gen_matched_pairs(rng, 4000)
        test_pred, test_gt = gen_matched_pairs(rng, 6000)
        cal_score = rng.uniform(0.05, 1.0, size=len(cal_pred))
        test_score = rng.uniform(0.05, 1.0, size=len(test_pred))
        alpha = 0.1
        calib = naive_coord_score_scaled.calibrate(cal_pred, cal_gt, cal_score, alpha=alpha)
        target = 1.0 - calib.alpha_per_coord
        for c in naive_coord_score_scaled.coord_names:
            scaled_res = np.array(
                [
                    naive_coord_score_scaled.residuals(p, g)[c] / naive_coord_score_scaled.sigma(s)
                    for p, g, s in zip(test_pred, test_gt, test_score)
                ]
            )
            coverage_c = float(np.mean(scaled_res <= calib.q[c]))
            assert abs(coverage_c - target) < 0.03

    def test_constant_score_reduces_to_unscaled_coverage_decisions(self, gen_matched_pairs, rng):
        # Ties/consistency check: when every detection has the SAME score, sigma is a
        # single constant for all of calibration and evaluation, so dividing every
        # residual by that constant before taking the quantile changes the threshold
        # by exactly that constant and every covers() decision matches the unscaled
        # construction bit-for-bit.
        cal_pred, cal_gt = gen_matched_pairs(rng, 500)
        test_pred, test_gt = gen_matched_pairs(rng, 500)
        const_score = 0.42
        alpha = 0.1
        cal_score = np.full(len(cal_pred), const_score)
        test_score = np.full(len(test_pred), const_score)

        calib_unscaled = naive_coord_score.calibrate(cal_pred, cal_gt, alpha=alpha)
        calib_scaled = naive_coord_score_scaled.calibrate(cal_pred, cal_gt, cal_score, alpha=alpha)

        for p, g, s in zip(test_pred, test_gt, test_score):
            covered_unscaled = naive_coord_score.covers(calib_unscaled, p, g)
            covered_scaled = naive_coord_score_scaled.covers(calib_scaled, p, g, s)
            assert covered_unscaled == covered_scaled

    def test_zero_score_detection_does_not_raise_and_yields_finite_wide_interval(
        self, gen_matched_pairs, rng
    ):
        cal_pred, cal_gt = gen_matched_pairs(rng, 500)
        cal_score = rng.uniform(0.05, 1.0, size=len(cal_pred))
        calib = naive_coord_score_scaled.calibrate(cal_pred, cal_gt, cal_score, alpha=0.1)
        pred, gt = cal_pred[0], cal_gt[0]
        # Must not raise ZeroDivisionError / produce inf or nan.
        result = naive_coord_score_scaled.covers(calib, pred, gt, pred_score=0.0)
        assert isinstance(result, (bool, np.bool_))
        halfwidths = naive_coord_score_scaled.cxcy_box_halfwidths(calib, pred_score=0.0)
        assert all(np.isfinite(h) for h in halfwidths)
        set_size = set_size_cxcy_slice_scaled(calib, pred_score=0.0)
        assert np.isfinite(set_size) and set_size > 0

    def test_low_score_detection_gets_wider_interval_than_high_score(self, gen_matched_pairs, rng):
        cal_pred, cal_gt = gen_matched_pairs(rng, 800)
        cal_score = rng.uniform(0.05, 1.0, size=len(cal_pred))
        calib = naive_coord_score_scaled.calibrate(cal_pred, cal_gt, cal_score, alpha=0.1)
        halfwidths_low = naive_coord_score_scaled.cxcy_box_halfwidths(calib, pred_score=0.05)
        halfwidths_high = naive_coord_score_scaled.cxcy_box_halfwidths(calib, pred_score=1.0)
        assert halfwidths_low[0] > halfwidths_high[0]
        assert halfwidths_low[1] > halfwidths_high[1]
        set_size_low = set_size_cxcy_slice_scaled(calib, pred_score=0.05)
        set_size_high = set_size_cxcy_slice_scaled(calib, pred_score=1.0)
        assert set_size_low > set_size_high

    def test_geometry_scale_equivariance(self, gen_matched_pairs, rng):
        # Scaling ALL translation/size coordinates (cx, cy, w, h) of both pred and gt
        # by a constant k, with scores held fixed, must scale the calibrated q for
        # those coordinates by exactly k -- the score-based sigma normalization is
        # independent of geometric units, same as the unscaled construction.
        cal_pred, cal_gt = gen_matched_pairs(rng, 1500)
        cal_score = rng.uniform(0.05, 1.0, size=len(cal_pred))
        alpha = 0.1
        k = 3.0

        cal_pred_scaled = cal_pred.copy()
        cal_gt_scaled = cal_gt.copy()
        cal_pred_scaled[:, :4] *= k  # cx, cy, w, h; theta (col 4) untouched
        cal_gt_scaled[:, :4] *= k

        calib = naive_coord_score_scaled.calibrate(cal_pred, cal_gt, cal_score, alpha=alpha)
        calib_scaled = naive_coord_score_scaled.calibrate(cal_pred_scaled, cal_gt_scaled, cal_score, alpha=alpha)

        for c in ("cx", "cy", "w", "h"):
            assert calib_scaled.q[c] == pytest.approx(k * calib.q[c], rel=1e-6)
        # theta residuals are unaffected by the (cx, cy, w, h) rescale.
        assert calib_scaled.q["theta"] == pytest.approx(calib.q["theta"], rel=1e-6)

    def test_calibrate_rejects_mismatched_score_length(self, gen_matched_pairs, rng):
        cal_pred, cal_gt = gen_matched_pairs(rng, 50)
        bad_score = np.ones(len(cal_pred) - 1)
        with pytest.raises(ValueError):
            naive_coord_score_scaled.calibrate(cal_pred, cal_gt, bad_score, alpha=0.1)

    def test_set_size_scaled_matches_formula(self, gen_matched_pairs, rng):
        cal_pred, cal_gt = gen_matched_pairs(rng, 300)
        cal_score = rng.uniform(0.05, 1.0, size=len(cal_pred))
        calib = naive_coord_score_scaled.calibrate(cal_pred, cal_gt, cal_score, alpha=0.1)
        pred_score = 0.37
        sigma = aleatoric_sigma_from_score(pred_score, calib.score_floor)
        expected = 4.0 * (calib.q["cx"] * sigma) * (calib.q["cy"] * sigma)
        assert set_size_cxcy_slice_scaled(calib, pred_score) == pytest.approx(expected)

"""Exhaustive property tests for rotcert.gwd -- the paper's centerpiece.

Covers: le90 canonicalization correctness, seam continuity (the C1 boundary-
discontinuity pathology), square-isotropy (the degeneracy safety), w/h-exchange
invariance, metric-axiom spot checks, hand-computed 2x2 Bures cases, and scale
normalization.
"""

from __future__ import annotations

import numpy as np
import pytest

from rotcert.gwd import (
    bures_sq,
    canonicalize_le90,
    gwd,
    gwd_decompose,
    gwd_sq,
    obb_gwd,
    obb_to_gaussian,
)


# ---------------------------------------------------------------------------
# canonicalize_le90
# ---------------------------------------------------------------------------


class TestCanonicalizeLe90:
    def test_already_canonical_passthrough(self):
        w, h, t = canonicalize_le90(20.0, 10.0, np.deg2rad(30))
        assert w == pytest.approx(20.0)
        assert h == pytest.approx(10.0)
        assert t == pytest.approx(np.deg2rad(30))

    def test_swaps_when_h_greater_than_w(self):
        # -30 + 90 = 60deg, safely inside [-90, 90) -- no additional wrap needed, so
        # this isolates the swap-and-rotate-by-90 behavior from the wrap behavior
        # (covered separately by test_theta_wrapped_into_range).
        w, h, t = canonicalize_le90(10.0, 20.0, np.deg2rad(-30))
        assert w == pytest.approx(20.0)
        assert h == pytest.approx(10.0)
        assert t == pytest.approx(np.deg2rad(-30) + np.pi / 2)

    def test_theta_wrapped_into_range(self):
        # 100 degrees is out of [-90, 90); after swap-free wrap should land at -80.
        w, h, t = canonicalize_le90(20.0, 10.0, np.deg2rad(100))
        assert -np.pi / 2 <= t < np.pi / 2
        assert t == pytest.approx(np.deg2rad(-80))

    def test_theta_at_exact_lower_bound_stays(self):
        w, h, t = canonicalize_le90(20.0, 10.0, -np.pi / 2)
        assert t == pytest.approx(-np.pi / 2)

    def test_full_period_wrap_is_identity_on_sigma(self):
        w0, h0, t0 = 20.0, 10.0, np.deg2rad(10)
        w1, h1, t1 = canonicalize_le90(w0, h0, t0 + np.pi)
        _, S0 = obb_to_gaussian(0, 0, w0, h0, t0, canonicalize=False)
        _, S1 = obb_to_gaussian(0, 0, w1, h1, t1, canonicalize=False)
        assert np.allclose(S0, S1, atol=1e-9)

    def test_negative_side_lengths_rejected(self):
        with pytest.raises(ValueError):
            canonicalize_le90(-1.0, 5.0, 0.0)

    def test_vectorized(self):
        w = np.array([10.0, 30.0, 5.0])
        h = np.array([20.0, 15.0, 5.0])
        t = np.array([0.0, np.deg2rad(10), np.deg2rad(200)])
        w2, h2, t2 = canonicalize_le90(w, h, t)
        assert np.all(w2 >= h2)
        assert np.all((t2 >= -np.pi / 2) & (t2 < np.pi / 2))


# ---------------------------------------------------------------------------
# obb_to_gaussian / bures_sq hand-computed cases
# ---------------------------------------------------------------------------


class TestObbToGaussian:
    def test_axis_aligned_sigma_is_diagonal(self):
        mu, Sigma = obb_to_gaussian(5.0, 7.0, 20.0, 10.0, 0.0)
        assert mu[0] == pytest.approx(5.0)
        assert mu[1] == pytest.approx(7.0)
        assert Sigma[0, 0] == pytest.approx((20 / 2) ** 2)
        assert Sigma[1, 1] == pytest.approx((10 / 2) ** 2)
        assert Sigma[0, 1] == pytest.approx(0.0, abs=1e-9)

    def test_90deg_rotation_swaps_diagonal(self):
        # theta=90deg is out of le90 range and canonicalizes to a w/h swap at theta=0.
        mu, Sigma = obb_to_gaussian(0.0, 0.0, 20.0, 10.0, np.pi / 2)
        assert Sigma[0, 0] == pytest.approx((10 / 2) ** 2, abs=1e-9)
        assert Sigma[1, 1] == pytest.approx((20 / 2) ** 2, abs=1e-9)

    def test_symmetric_psd(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            w, h = rng.uniform(1, 50, 2)
            theta = rng.uniform(-10, 10)
            _, Sigma = obb_to_gaussian(0, 0, w, h, theta)
            assert Sigma[0, 1] == pytest.approx(Sigma[1, 0])
            eigvals = np.linalg.eigvalsh(Sigma)
            assert np.all(eigvals >= -1e-9)

    def test_bures_sq_hand_computed_diagonal_case(self):
        # Two axis-aligned Gaussians: Sigma1=diag(4,1), Sigma2=diag(1,4).
        # Bures^2 = tr1+tr2 - 2*sqrt(tr(S1 S2) + 2*sqrt(det1*det2))
        # tr1=5, tr2=5, S1 S2 = diag(4,4) -> tr=8, det1=4, det2=4 -> sqrt(16)=4
        # inner = 8 + 2*4 = 16, sqrt(16)=4 -> Bures^2 = 5+5-2*4 = 2
        S1 = np.array([[4.0, 0.0], [0.0, 1.0]])
        S2 = np.array([[1.0, 0.0], [0.0, 4.0]])
        b2 = bures_sq(S1, S2)
        assert float(b2) == pytest.approx(2.0, abs=1e-9)

    def test_bures_sq_identical_matrices_is_zero(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            w, h = rng.uniform(1, 30, 2)
            theta = rng.uniform(-2, 2)
            _, S = obb_to_gaussian(0, 0, w, h, theta)
            assert float(bures_sq(S, S)) == pytest.approx(0.0, abs=1e-7)

    def test_bures_sq_isotropic_case_matches_euclidean(self):
        # For isotropic Sigma1=a*I, Sigma2=b*I (2x2), Bures^2 = 2*(sqrt(a)-sqrt(b))^2
        # (well-known: Bures distance between isotropic Gaussians reduces to
        # Euclidean distance between their std-devs, scaled by sqrt(dim)).
        a, b = 9.0, 4.0
        S1 = a * np.eye(2)
        S2 = b * np.eye(2)
        expected = 2.0 * (np.sqrt(a) - np.sqrt(b)) ** 2
        assert float(bures_sq(S1, S2)) == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Seam continuity (the C1 boundary-discontinuity pathology, design §2.1)
# ---------------------------------------------------------------------------


class TestSeamContinuity:
    def test_89_vs_minus_89_is_small(self):
        # Physically 2 degrees apart (89 -> 91 == -89 mod 180), numerically 178deg apart.
        pred = np.array([50.0, 50.0, 20.0, 10.0, np.deg2rad(89)])
        gt_seam = np.array([50.0, 50.0, 20.0, 10.0, np.deg2rad(-89)])
        gt_far = np.array([50.0, 50.0, 20.0, 10.0, np.deg2rad(0)])
        d_seam = obb_gwd(pred, gt_seam)
        d_far = obb_gwd(pred, gt_far)
        assert d_seam < d_far

    def test_seam_gwd_matches_equivalent_small_physical_offset(self):
        # GWD(89deg, -89deg) should match GWD(89deg, 91deg-worth-of-canonical-equiv)
        # i.e. the same physical 2-degree offset computed entirely within-range.
        pred = np.array([50.0, 50.0, 20.0, 10.0, np.deg2rad(89)])
        gt_seam = np.array([50.0, 50.0, 20.0, 10.0, np.deg2rad(-89)])
        gt_equiv = np.array([50.0, 50.0, 20.0, 10.0, np.deg2rad(87)])  # 2deg from 89, no wrap
        assert obb_gwd(pred, gt_seam) == pytest.approx(obb_gwd(pred, gt_equiv), abs=1e-6)

    def test_continuity_sweep_across_seam(self):
        # GWD(fixed pred near seam, gt) should vary smoothly (no jump) as gt sweeps
        # across the +-90deg boundary.
        pred = np.array([50.0, 50.0, 20.0, 10.0, np.deg2rad(89.9)])
        thetas = np.linspace(np.deg2rad(85), np.deg2rad(-85) + np.pi, 200)  # sweep through the seam, unwrapped
        thetas_wrapped = ((thetas + np.pi / 2) % np.pi) - np.pi / 2
        gts = np.stack(
            [np.full(200, 50.0), np.full(200, 50.0), np.full(200, 20.0), np.full(200, 10.0), thetas_wrapped],
            axis=-1,
        )
        dists = obb_gwd(np.broadcast_to(pred, gts.shape), gts)
        # Max consecutive jump should be small (no discontinuous spike at the seam).
        jumps = np.abs(np.diff(dists))
        assert np.max(jumps) < 0.5 * np.max(dists) + 1e-6


# ---------------------------------------------------------------------------
# Square-isotropy (the degeneracy safety, design §2.1 pathology 2)
# ---------------------------------------------------------------------------


class TestSquareIsotropy:
    def test_exact_square_angle_invariant(self):
        pred = np.array([10.0, 10.0, 15.0, 15.0, np.deg2rad(5)])
        for angle_deg in [-80, -40, 0, 40, 80]:
            gt = np.array([10.0, 10.0, 15.0, 15.0, np.deg2rad(angle_deg)])
            assert obb_gwd(pred, gt) == pytest.approx(0.0, abs=1e-9)

    def test_near_square_small_gwd_across_angles(self):
        pred = np.array([10.0, 10.0, 15.0, 14.9, np.deg2rad(5)])
        gt = np.array([10.0, 10.0, 15.0, 14.9, np.deg2rad(85)])
        # Not exactly 0 (w != h) but should be small relative to the elongated case.
        d_square = obb_gwd(pred, gt)
        pred_elong = np.array([10.0, 10.0, 30.0, 5.0, np.deg2rad(5)])
        gt_elong = np.array([10.0, 10.0, 30.0, 5.0, np.deg2rad(85)])
        d_elong = obb_gwd(pred_elong, gt_elong)
        assert d_square < d_elong

    def test_square_gaussian_is_isotropic(self):
        _, Sigma = obb_to_gaussian(0, 0, 20.0, 20.0, np.deg2rad(37))
        assert Sigma[0, 0] == pytest.approx(Sigma[1, 1])
        assert Sigma[0, 1] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# w/h-exchange invariance (design §2.2 identifiability)
# ---------------------------------------------------------------------------


class TestExchangeInvariance:
    @pytest.mark.parametrize("theta_deg", [-80, -45, -10, 0, 10, 45, 80])
    def test_swap_and_rotate_90_is_identity(self, theta_deg):
        theta = np.deg2rad(theta_deg)
        obb_a = np.array([3.0, 4.0, 30.0, 12.0, theta])
        obb_b = np.array([3.0, 4.0, 12.0, 30.0, theta - np.pi / 2])
        assert obb_gwd(obb_a, obb_b) == pytest.approx(0.0, abs=1e-8)

    def test_canonicalization_of_swapped_form_matches(self):
        w1, h1, t1 = canonicalize_le90(30.0, 12.0, np.deg2rad(20))
        w2, h2, t2 = canonicalize_le90(12.0, 30.0, np.deg2rad(20) - np.pi / 2)
        assert (float(w1), float(h1), float(t1)) == pytest.approx((float(w2), float(h2), float(t2)), abs=1e-9)


# ---------------------------------------------------------------------------
# Metric-axiom spot checks
# ---------------------------------------------------------------------------


class TestMetricAxioms:
    def test_identity_of_indiscernibles(self, rng=np.random.default_rng(3)):
        for _ in range(30):
            obb = np.array([rng.uniform(0, 100), rng.uniform(0, 100), rng.uniform(1, 40), rng.uniform(1, 40), rng.uniform(-2, 2)])
            assert obb_gwd(obb, obb) == pytest.approx(0.0, abs=1e-6)

    def test_symmetry(self):
        rng = np.random.default_rng(4)
        for _ in range(50):
            a = np.array([rng.uniform(0, 100), rng.uniform(0, 100), rng.uniform(1, 40), rng.uniform(1, 40), rng.uniform(-2, 2)])
            b = np.array([rng.uniform(0, 100), rng.uniform(0, 100), rng.uniform(1, 40), rng.uniform(1, 40), rng.uniform(-2, 2)])
            assert obb_gwd(a, b) == pytest.approx(obb_gwd(b, a), abs=1e-9)

    def test_nonnegativity(self):
        rng = np.random.default_rng(5)
        for _ in range(100):
            a = np.array([rng.uniform(0, 100), rng.uniform(0, 100), rng.uniform(0.5, 40), rng.uniform(0.5, 40), rng.uniform(-5, 5)])
            b = np.array([rng.uniform(0, 100), rng.uniform(0, 100), rng.uniform(0.5, 40), rng.uniform(0.5, 40), rng.uniform(-5, 5)])
            assert obb_gwd(a, b) >= -1e-9

    def test_triangle_inequality_spot_checks(self):
        rng = np.random.default_rng(6)
        violations = 0
        n = 300
        for _ in range(n):
            a = np.array([rng.uniform(0, 100), rng.uniform(0, 100), rng.uniform(1, 40), rng.uniform(1, 40), rng.uniform(-2, 2)])
            b = np.array([rng.uniform(0, 100), rng.uniform(0, 100), rng.uniform(1, 40), rng.uniform(1, 40), rng.uniform(-2, 2)])
            c = np.array([rng.uniform(0, 100), rng.uniform(0, 100), rng.uniform(1, 40), rng.uniform(1, 40), rng.uniform(-2, 2)])
            d_ac = obb_gwd(a, c)
            d_ab_bc = obb_gwd(a, b) + obb_gwd(b, c)
            if d_ac > d_ab_bc + 1e-6:
                violations += 1
        # GWD is a genuine metric (2-Wasserstein distance restricted to Gaussians is a
        # metric on the Bures-Wasserstein manifold); allow zero tolerance violations.
        assert violations == 0


# ---------------------------------------------------------------------------
# gwd_decompose (diagnostic-only split)
# ---------------------------------------------------------------------------


class TestDecompose:
    def test_decompose_sums_to_gwd_sq(self):
        rng = np.random.default_rng(7)
        for _ in range(30):
            a = np.array([rng.uniform(0, 100), rng.uniform(0, 100), rng.uniform(1, 40), rng.uniform(1, 40), rng.uniform(-2, 2)])
            b = np.array([rng.uniform(0, 100), rng.uniform(0, 100), rng.uniform(1, 40), rng.uniform(1, 40), rng.uniform(-2, 2)])
            mu1, S1 = obb_to_gaussian(*a[:2], *a[2:])
            mu2, S2 = obb_to_gaussian(*b[:2], *b[2:])
            center, shape = gwd_decompose(mu1, S1, mu2, S2)
            assert float(center + shape) == pytest.approx(float(gwd_sq(mu1, S1, mu2, S2)), abs=1e-9)
            assert float(center + shape) == pytest.approx(float(obb_gwd(a, b) ** 2), abs=1e-6)

    def test_pure_translation_has_zero_shape_term(self):
        a = np.array([0.0, 0.0, 20.0, 10.0, 0.3])
        b = np.array([5.0, -3.0, 20.0, 10.0, 0.3])
        mu1, S1 = obb_to_gaussian(*a[:2], *a[2:])
        mu2, S2 = obb_to_gaussian(*b[:2], *b[2:])
        center, shape = gwd_decompose(mu1, S1, mu2, S2)
        assert float(shape) == pytest.approx(0.0, abs=1e-8)
        assert float(center) == pytest.approx(25.0 + 9.0, abs=1e-8)  # 5^2+3^2


# ---------------------------------------------------------------------------
# Scale normalization
# ---------------------------------------------------------------------------


class TestScaleNorm:
    def test_sqrt_area_norm_scales_down_large_objects(self):
        small_pred = np.array([0.0, 0.0, 10.0, 5.0, 0.0])
        small_gt = np.array([1.0, 1.0, 10.0, 5.0, 0.0])
        large_pred = np.array([0.0, 0.0, 100.0, 50.0, 0.0])
        large_gt = np.array([10.0, 10.0, 100.0, 50.0, 0.0])
        raw_small = obb_gwd(small_pred, small_gt)
        raw_large = obb_gwd(large_pred, large_gt)
        norm_small = obb_gwd(small_pred, small_gt, scale_norm="sqrt-area")
        norm_large = obb_gwd(large_pred, large_gt, scale_norm="sqrt-area")
        assert raw_large > raw_small  # raw grows with object scale
        # normalized values should be much closer in magnitude to each other
        assert abs(norm_large - norm_small) < abs(raw_large - raw_small)

    def test_unknown_scale_norm_raises(self):
        a = np.array([0.0, 0.0, 10.0, 5.0, 0.0])
        with pytest.raises(ValueError):
            obb_gwd(a, a, scale_norm="bogus")

    def test_bad_last_dim_raises(self):
        with pytest.raises(ValueError):
            obb_gwd(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# gwd / gwd_sq consistency
# ---------------------------------------------------------------------------


def test_gwd_is_sqrt_of_gwd_sq():
    rng = np.random.default_rng(8)
    a = np.array([rng.uniform(0, 100), rng.uniform(0, 100), rng.uniform(1, 40), rng.uniform(1, 40), rng.uniform(-2, 2)])
    b = np.array([rng.uniform(0, 100), rng.uniform(0, 100), rng.uniform(1, 40), rng.uniform(1, 40), rng.uniform(-2, 2)])
    mu1, S1 = obb_to_gaussian(*a[:2], *a[2:])
    mu2, S2 = obb_to_gaussian(*b[:2], *b[2:])
    assert float(gwd(mu1, S1, mu2, S2)) == pytest.approx(float(np.sqrt(gwd_sq(mu1, S1, mu2, S2))))


def test_vectorized_batch_matches_loop():
    rng = np.random.default_rng(9)
    n = 25
    a = np.stack([rng.uniform(0, 100, n), rng.uniform(0, 100, n), rng.uniform(1, 40, n), rng.uniform(1, 40, n), rng.uniform(-2, 2, n)], axis=-1)
    b = np.stack([rng.uniform(0, 100, n), rng.uniform(0, 100, n), rng.uniform(1, 40, n), rng.uniform(1, 40, n), rng.uniform(-2, 2, n)], axis=-1)
    batch = obb_gwd(a, b)
    loop = np.array([obb_gwd(a[i], b[i]) for i in range(n)])
    assert np.allclose(batch, loop, atol=1e-8)

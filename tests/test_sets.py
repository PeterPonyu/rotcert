"""GWD membership, exact center containment and numerical shape diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from rotcert.gwd import canonicalize_le90, obb_gwd
from rotcert.sets import center_envelope, envelope, gwd_ball_membership, shape_envelope


class TestBallMembership:
    def test_self_is_member(self):
        obb = np.array([10.0, 10.0, 20.0, 10.0, 0.3])
        assert gwd_ball_membership(obb, obb, q_hat=0.01)

    def test_far_point_not_member_at_small_radius(self):
        pred = np.array([0.0, 0.0, 20.0, 10.0, 0.0])
        far = np.array([50.0, 50.0, 20.0, 10.0, 0.0])
        assert not gwd_ball_membership(pred, far, q_hat=1.0)

    def test_far_point_member_at_large_radius(self):
        pred = np.array([0.0, 0.0, 20.0, 10.0, 0.0])
        far = np.array([50.0, 50.0, 20.0, 10.0, 0.0])
        assert gwd_ball_membership(pred, far, q_hat=1000.0)

    def test_negative_q_hat_raises(self):
        obb = np.array([0.0, 0.0, 20.0, 10.0, 0.0])
        with pytest.raises(ValueError):
            gwd_ball_membership(obb, obb, q_hat=-1.0)


class TestCenterEnvelope:
    def test_exact_disk_radius(self):
        pred = np.array([10.0, 20.0, 20.0, 10.0, 0.0])
        env = center_envelope(pred, q_hat=3.0)
        assert env["cx"] == (7.0, 13.0)
        assert env["cy"] == (17.0, 23.0)

    def test_center_bound_is_attained_same_shape(self):
        # Point at (cx+q, cy) with the SAME shape should be exactly on the ball boundary.
        pred = np.array([10.0, 20.0, 20.0, 10.0, 0.3])
        q = 4.0
        boundary_pt = pred.copy()
        boundary_pt[0] += q
        d = obb_gwd(pred, boundary_pt)
        assert d == pytest.approx(q, abs=1e-6)


class TestShapeEnvelope:
    def test_non_square_gives_bounded_arc(self):
        pred = np.array([50.0, 50.0, 20.0, 10.0, np.deg2rad(30)])
        env = shape_envelope(pred, q_hat=3.0)
        assert not env["theta_full_arc"]
        assert env["w"][0] <= 20.0 <= env["w"][1]
        assert env["h"][0] <= 10.0 <= env["h"][1]

    def test_near_square_gives_full_arc(self):
        pred = np.array([50.0, 50.0, 15.0, 14.6, np.deg2rad(10)])
        env = shape_envelope(pred, q_hat=2.0)
        assert env["theta_full_arc"]
        assert env["theta"] == (-np.pi / 2, np.pi / 2)

    def test_tiny_q_hat_reports_unavailable_shape_bounds(self):
        pred = np.array([50.0, 50.0, 20.0, 10.0, np.deg2rad(30)])
        env = shape_envelope(pred, q_hat=1e-8, n_w=15, n_h=9, n_theta=31)
        assert env["degenerate"]
        assert env["w"] is None and env["h"] is None and env["theta"] is None
        assert env["status"] == "insufficient_grid_resolution"
        assert env["continuous_outer_bound"] is False

    def test_seam_adjacent_arc_wraps(self):
        pred = np.array([50.0, 50.0, 20.0, 10.0, np.deg2rad(89)])
        env = shape_envelope(pred, q_hat=3.0)
        assert not env["theta_full_arc"]
        lo, hi = env["theta"]
        # The predicted angle itself must lie within the reported arc (possibly via
        # its unwrapped +pi representative).
        pred_t = np.deg2rad(89)
        assert (lo - 1e-6 <= pred_t <= hi + 1e-6) or (lo - 1e-6 <= pred_t + np.pi <= hi + 1e-6)


class TestExactCenterContainment:
    def test_off_grid_ball_member_need_not_be_in_sampled_shape_extent(self):
        # Positive-radius ball has members even when the coarse grid misses them.
        pred = np.array([0., 0., 20., 10., 0.3])
        q = 1e-4
        candidate = pred.copy()
        candidate[2] += q
        assert gwd_ball_membership(pred, candidate, q)
        result = envelope(pred, q, n_w=3, n_h=2, n_theta=2)
        assert result["w"] is None
        assert result["cx"] == (-q, q)
        assert result["cy"] == (-q, q)

    def test_random_ball_members_obey_exact_center_bounds(self):
        pred = np.array([10., 20., 30., 5., 0.3]); q = 3.
        rng = np.random.default_rng(42)
        candidates = np.tile(pred, (200, 1))
        candidates[:, :2] += rng.normal(size=(200, 2))
        members = candidates[gwd_ball_membership(pred, candidates, q)]
        assert len(members) > 100
        bounds = center_envelope(pred, q)
        assert np.all((members[:, 0] >= bounds["cx"][0]) & (members[:, 0] <= bounds["cx"][1]))
        assert np.all((members[:, 1] >= bounds["cy"][0]) & (members[:, 1] <= bounds["cy"][1]))

    @pytest.mark.parametrize("radius", [-1., float("nan"), float("inf")])
    def test_invalid_shape_radius_is_rejected(self, radius):
        with pytest.raises(ValueError):
            shape_envelope(np.array([0., 0., 20., 10., 0.3]), radius)

    def test_nonempty_grid_is_explicitly_uncertified(self):
        result = shape_envelope(np.array([0., 0., 20., 10., 0.3]), 3.)
        assert result["status"] == "sampled_feasible_extrema"
        assert result["continuous_outer_bound"] is False

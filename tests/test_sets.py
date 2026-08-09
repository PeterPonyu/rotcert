"""Tests for rotcert.sets: GWD-ball membership + the conservative envelope, including
the ball-subseteq-envelope property (design's reader-warning requirement)."""

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

    def test_tiny_q_hat_degenerate_collapses_to_point(self):
        pred = np.array([50.0, 50.0, 20.0, 10.0, np.deg2rad(30)])
        env = shape_envelope(pred, q_hat=1e-8, n_w=15, n_h=9, n_theta=31)
        assert env["degenerate"]
        assert env["w"] == (pytest.approx(20.0), pytest.approx(20.0))

    def test_seam_adjacent_arc_wraps(self):
        pred = np.array([50.0, 50.0, 20.0, 10.0, np.deg2rad(89)])
        env = shape_envelope(pred, q_hat=3.0)
        assert not env["theta_full_arc"]
        lo, hi = env["theta"]
        # The predicted angle itself must lie within the reported arc (possibly via
        # its unwrapped +pi representative).
        pred_t = np.deg2rad(89)
        assert (lo - 1e-6 <= pred_t <= hi + 1e-6) or (lo - 1e-6 <= pred_t + np.pi <= hi + 1e-6)


class TestBallSubsetEnvelope:
    """The reader-warning property: EVERY point in the GWD-ball must fall within the
    reported per-parameter envelope (design §2.3)."""

    @pytest.mark.parametrize(
        "pred,q",
        [
            (np.array([50.0, 50.0, 20.0, 10.0, np.deg2rad(30)]), 3.0),
            (np.array([50.0, 50.0, 20.0, 10.0, np.deg2rad(89)]), 3.0),
            (np.array([50.0, 50.0, 15.0, 14.5, np.deg2rad(10)]), 2.0),
            (np.array([10.0, 10.0, 30.0, 5.0, np.deg2rad(-89.5)]), 2.5),
        ],
    )
    def test_rejection_sampled_ball_points_within_envelope(self, pred, q):
        rng = np.random.default_rng(42)
        env = envelope(pred, q)
        cx, cy = pred[0], pred[1]
        w_lo, w_hi = env["w"]
        h_lo, h_hi = env["h"]
        th_lo, th_hi = env["theta"]

        n_checked = 0
        for _ in range(3000):
            cand = pred + rng.normal(scale=[q * 1.5, q * 1.5, q * 1.5, q * 1.5, 0.3], size=5)
            cand[2] = abs(cand[2]) + 0.5
            cand[3] = abs(cand[3]) + 0.1
            if not gwd_ball_membership(pred, cand, q):
                continue
            n_checked += 1
            wc, hc, thc = canonicalize_le90(cand[2], cand[3], cand[4])
            wc, hc, thc = float(wc), float(hc), float(thc)

            assert (cx - q - 1e-6) <= cand[0] <= (cx + q + 1e-6)
            assert (cy - q - 1e-6) <= cand[1] <= (cy + q + 1e-6)
            assert (w_lo - 1e-6) <= wc <= (w_hi + 1e-6)
            assert (h_lo - 1e-6) <= hc <= (h_hi + 1e-6)
            if not env["theta_full_arc"]:
                ok_th = (th_lo - 1e-6) <= thc <= (th_hi + 1e-6)
                if not ok_th and th_hi > np.pi / 2:
                    ok_th = (th_lo - 1e-6) <= (thc + np.pi) <= (th_hi + 1e-6)
                assert ok_th
        assert n_checked > 20  # sanity: the sampler actually found ball members

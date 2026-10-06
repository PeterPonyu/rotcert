"""GWD-ball membership, exact center bounds and sampled shape extents.

Center bounds follow analytically from nonnegative Bures distance. Shape extrema
are from feasible grid points only, not conservative bounds on the continuous
GWD ball and not calibrated marginal intervals. Increasing resolution or padding
does not establish an outward error bound. Probability statements require the
separate calibrated-score assumptions. These diagnostics do not certify false
positives or membership of arbitrary deployment detections in the TP population.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from rotcert.gwd import bures_sq, canonicalize_le90, obb_gwd, obb_to_gaussian

__all__ = ["gwd_ball_membership", "center_envelope", "shape_envelope", "envelope"]


def gwd_ball_membership(
    pred_obb: np.ndarray, candidate_obb: np.ndarray, q_hat: float
) -> np.ndarray:
    """Boolean: is ``candidate_obb`` inside the GWD-ball ``S(pred_obb)`` of radius
    ``q_hat``? This is geometric set membership; a coverage interpretation requires valid calibration."""
    if q_hat < 0:
        raise ValueError("gwd_ball_membership: q_hat must be non-negative")
    return obb_gwd(pred_obb, candidate_obb) <= q_hat


def center_envelope(pred_obb: np.ndarray, q_hat: float) -> Dict[str, Any]:
    """Exact per-axis bound on ``(cx, cy)`` (see module docstring)."""
    pred_obb = np.asarray(pred_obb, dtype=float)
    cx, cy = float(pred_obb[0]), float(pred_obb[1])
    return {"cx": (cx - q_hat, cx + q_hat), "cy": (cy - q_hat, cy + q_hat)}


def _circular_arc_bounds(feasible_thetas: np.ndarray, all_thetas: np.ndarray):
    """Smallest-enclosing circular arc (period ``pi``) covering ``feasible_thetas``.

    Returns ``(lo, hi, full_arc)``. ``full_arc=True`` means every grid angle is
    feasible on the sampled grid (report the full ``[-pi/2, pi/2)`` arc, per the
    sampled angle range, without a continuous-set guarantee). Otherwise ``lo <= hi`` with
    ``hi`` possibly ``>= pi/2`` (a wrap through the seam is represented by letting the
    arc continue past ``pi/2``; callers wanting a display-range value should take
    ``hi - pi`` if ``hi >= pi/2``, but the raw ``(lo, hi)`` is the mathematically
    arc enclosing the sampled angles only).
    """
    if feasible_thetas.size == 0:
        return None
    if feasible_thetas.size >= all_thetas.size:
        return (-np.pi / 2.0, np.pi / 2.0, True)
    thetas = np.sort(np.unique(feasible_thetas))
    if thetas.size == 1:
        return (float(thetas[0]), float(thetas[0]), False)
    gaps = np.diff(thetas)
    wrap_gap = (thetas[0] + np.pi) - thetas[-1]
    all_gaps = np.append(gaps, wrap_gap)
    max_gap_idx = int(np.argmax(all_gaps))
    if max_gap_idx == len(thetas) - 1:
        # Largest gap is the wraparound itself: the arc does not cross the seam.
        return (float(thetas[0]), float(thetas[-1]), False)
    start_idx = max_gap_idx + 1
    lo = float(thetas[start_idx])
    hi = float(thetas[max_gap_idx] + np.pi)  # unwrap through the seam
    return (lo, hi, False)


def shape_envelope(
    pred_obb: np.ndarray,
    q_hat: float,
    w_pad_factor: float = 4.0,
    n_w: int = 81,
    n_h: int = 41,
    n_theta: int = 181,
    min_w_pad: float = 1e-3,
) -> Dict[str, Any]:
    """Sampled feasible shape extents, not an outer envelope.

    ``bounds_kind`` is always ``sampled_feasible_extrema`` and
    ``continuous_outer_bound`` is always false. If no grid point is feasible,
    shape bounds are null, ``degenerate`` is true, and ``status`` reports
    ``insufficient_grid_resolution``. The exact center bounds remain usable.
    All-angle grid feasibility is budget saturation at this grid resolution;
    it does not prove continuous angular coverage or identify near-square boxes.
    """
    if not np.isfinite(q_hat) or q_hat < 0:
        raise ValueError("shape diagnostics require a finite, non-negative radius")
    if any(type(n) is not int or n < 2 for n in (n_w, n_h, n_theta)):
        raise ValueError("grid dimensions must be integers of at least two")
    if not np.isfinite(w_pad_factor) or w_pad_factor <= 0 or not np.isfinite(min_w_pad) or min_w_pad <= 0:
        raise ValueError("grid padding must be finite and positive")
    pred_obb = np.asarray(pred_obb, dtype=float)
    w_p, h_p, theta_p = canonicalize_le90(pred_obb[2], pred_obb[3], pred_obb[4])
    w_p, h_p, theta_p = float(w_p), float(h_p), float(theta_p)
    _, Sigma_pred = obb_to_gaussian(0.0, 0.0, w_p, h_p, theta_p, canonicalize=False)

    pad = w_pad_factor * q_hat + min_w_pad
    w_lo_grid = max(1e-6, w_p - pad)
    w_hi_grid = w_p + pad
    w_grid = np.linspace(w_lo_grid, w_hi_grid, n_w)
    theta_grid = np.linspace(-np.pi / 2.0, np.pi / 2.0, n_theta, endpoint=False)
    h_frac = np.linspace(1e-3, 1.0, n_h)

    W = w_grid[:, None, None]
    H = W * h_frac[None, :, None]
    TH = theta_grid[None, None, :]
    W_b, H_b, TH_b = np.broadcast_arrays(W, H, TH)

    _, Sigma_grid = obb_to_gaussian(0.0, 0.0, W_b, H_b, TH_b, canonicalize=False)
    b2 = bures_sq(Sigma_grid, Sigma_pred)
    feasible = b2 <= (q_hat ** 2)

    if not feasible.any():
        return {
            "w": None,
            "h": None,
            "theta": None,
            "status": "insufficient_grid_resolution",
            "bounds_kind": "sampled_feasible_extrema",
            "continuous_outer_bound": False,
            "theta_full_arc": False,
            "touches_w_grid_edge": False,
            "degenerate": True,
        }

    w_feas = W_b[feasible]
    h_feas = H_b[feasible]
    theta_feas_grid_vals = np.unique(TH_b[feasible])

    w_lo, w_hi = float(w_feas.min()), float(w_feas.max())
    h_lo, h_hi = float(h_feas.min()), float(h_feas.max())
    arc = _circular_arc_bounds(theta_feas_grid_vals, theta_grid)
    theta_lo, theta_hi, full_arc = arc

    touches_edge = bool(
        np.isclose(w_lo, w_lo_grid, rtol=0, atol=1e-9)
        or np.isclose(w_hi, w_hi_grid, rtol=0, atol=1e-9)
    )

    return {
        "w": (w_lo, w_hi),
        "h": (h_lo, h_hi),
        "theta": (theta_lo, theta_hi),
        "theta_full_arc": bool(full_arc),
        "touches_w_grid_edge": touches_edge,
        "degenerate": False,
        "status": "sampled_feasible_extrema",
        "bounds_kind": "sampled_feasible_extrema",
        "continuous_outer_bound": False,
    }


def envelope(pred_obb: np.ndarray, q_hat: float, **shape_kwargs: Any) -> Dict[str, Any]:
    """Exact center bounds plus explicitly uncalibrated sampled shape extents."""
    out = dict(center_envelope(pred_obb, q_hat))
    out.update(shape_envelope(pred_obb, q_hat, **shape_kwargs))
    return out

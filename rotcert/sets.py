"""G1 coverage sets: the GWD-ball certificate and its conservative per-parameter
envelope (design §2.3).

**Reader warning, restated from the design doc (do not violate this in any consumer):**
the certificate is the GWD-BALL ``S(p) = {b : GWD(p, b) <= q_hat}``. The per-parameter
envelope computed here is a conservative BOUNDING BOX around that ball for
visualization only -- it over-covers, and its per-axis widths are NOT calibrated
marginal intervals. Never report an envelope width as if it carried the ``1 - alpha``
guarantee on its own; only the ball does.

Center envelope (exact, closed form)
--------------------------------------
``GWD^2 = ||mu_pred - mu||^2 + Bures^2(Sigma_pred, Sigma)`` is additive with
``Bures^2 >= 0``, so any point in the ball has ``||mu_pred - mu||^2 <= q_hat^2``,
hence ``|cx - cx_pred| <= q_hat`` and ``|cy - cy_pred| <= q_hat`` individually -- a
TIGHT bound (attained by any same-shape box shifted by exactly ``q_hat`` along one
axis, which has ``Bures^2 = 0``).

Shape envelope (w, h, theta): grid-search bound, not closed form
----------------------------------------------------------------------
Unlike the center, ``Bures^2(Sigma_pred, Sigma(w,h,theta))`` does not decompose per
shape parameter, so there is no simple closed-form per-axis bound. We instead grid-scan
the reachable ``(w, h, theta)`` neighborhood (canonical: ``h`` swept as a fraction of
``w`` in ``(0, w]``, so every grid point is already le90-canonical) and report, PER
AXIS, the min/max value attained by any grid point whose Bures^2 lies inside the
remaining budget ``q_hat^2`` (full budget, since the worst case for shape spends none
of it on center offset -- ``mu = mu_pred`` exactly). This is a genuine (not merely
per-coordinate-conditional) marginal bound: for THETA specifically, because it is
angular, the feasible set can WRAP through the le90 seam (``+-pi/2``) or, for a
near-square predicted box, cover the FULL arc (the square-stratum "angular vacuity"
the design calls out, §4.5) -- :func:`_circular_arc_bounds` detects both cases rather
than reporting a naive (and wrong) ``[min, max]`` over the raw grid values.
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
    ``q_hat``? THIS is the certificate; :func:`envelope` is reporting-only."""
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
    feasible (the near-square case -- report the full ``[-pi/2, pi/2)`` arc, per the
    design's square-stratum "angular vacuity" disclosure). Otherwise ``lo <= hi`` with
    ``hi`` possibly ``>= pi/2`` (a wrap through the seam is represented by letting the
    arc continue past ``pi/2``; callers wanting a display-range value should take
    ``hi - pi`` if ``hi >= pi/2``, but the raw ``(lo, hi)`` is the mathematically
    correct arc and is what the ball-in-envelope containment check must use).
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
    """Conservative grid-search envelope on ``(w, h, theta)`` (see module docstring).

    Parameters
    ----------
    pred_obb:
        ``(cx, cy, w, h, theta)`` -- only ``w, h, theta`` are used (canonicalized).
    q_hat:
        GWD-ball radius.
    w_pad_factor, min_w_pad:
        The ``w`` search window is
        ``[max(eps, w_pred - pad), w_pred + pad]``, ``pad = w_pad_factor * q_hat +
        min_w_pad``. Generous by construction (over-covers rather than clipping the
        true boundary) -- raise ``w_pad_factor`` if :func:`shape_envelope` reports a
        feasible region touching the grid edge (checked via ``touches_w_grid_edge``
        in the returned dict; callers should treat that as "re-run with a wider pad,"
        not as the true bound).
    n_w, n_h, n_theta:
        Grid resolution. ``h`` is swept as ``n_h`` fractions of each candidate ``w``
        in ``(0, w]`` so every grid point is le90-canonical by construction.

    Returns
    -------
    dict
        ``w`` (lo, hi), ``h`` (lo, hi), ``theta`` (lo, hi, possibly wrapped past
        ``pi/2`` -- see :func:`_circular_arc_bounds`), ``theta_full_arc`` (bool),
        ``touches_w_grid_edge`` (bool, a resolution/padding warning), ``degenerate``
        (bool: q_hat too small relative to grid resolution for any grid point to be
        feasible -- the envelope collapses to the predicted box exactly, a
        (correctly) conservative but not visually useful report; increase resolution
        for very small q_hat).
    """
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
            "w": (w_p, w_p),
            "h": (h_p, h_p),
            "theta": (theta_p, theta_p),
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
    }


def envelope(pred_obb: np.ndarray, q_hat: float, **shape_kwargs: Any) -> Dict[str, Any]:
    """Full conservative per-parameter envelope: :func:`center_envelope` +
    :func:`shape_envelope`, merged into one dict (``cx``, ``cy``, ``w``, ``h``,
    ``theta``, ``theta_full_arc``, ``touches_w_grid_edge``, ``degenerate``)."""
    out = dict(center_envelope(pred_obb, q_hat))
    out.update(shape_envelope(pred_obb, q_hat, **shape_kwargs))
    return out

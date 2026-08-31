"""Angle-aware Gaussian-Wasserstein-distance (GWD) nonconformity (design §2.2d, §2.3).

This module IS the paper's centerpiece (protocol specification §2). Every
other module in ``rotcert`` is plumbing around this one.

The le90 long-edge convention
------------------------------
An oriented box (OBB) is ``(cx, cy, w, h, theta)`` with ``w`` the LONG edge, ``h`` the
SHORT edge, ``theta`` the long-edge's angle in ``[-pi/2, pi/2)`` (radians). Because a
non-square rectangle is invariant under a 180-degree rotation, its orientation lives on
the quotient ``R / pi*Z`` (period ``pi``, not ``2*pi``); le90 picks one representative
per period. :func:`canonicalize_le90` maps ANY ``(w, h, theta)`` (any assignment of the
two side lengths, any angle) onto this canonical representative -- every public function
in this module canonicalizes its inputs before scoring (``canonicalize=True`` default),
so callers never need to pre-canonicalize.

The Gaussian representation and why it is boundary-continuous + square-safe
------------------------------------------------------------------------------
Represent an OBB as a 2-D Gaussian ``N(mu, Sigma)``, ``mu = (cx, cy)``,
``Sigma = R(theta) @ diag((w/2)^2, (h/2)^2) @ R(theta)^T``. ``Sigma`` is an EXACTLY
pi-periodic, smooth function of ``theta`` (``R(theta + pi) = -R(theta)``, and
``(-R) @ D @ (-R)^T = R @ D @ R^T``), so a prediction at ``89 deg`` and a ground truth
at ``-89 deg`` (178 degrees apart numerically, 2 degrees apart physically) map to Sigma
matrices that are close -- this is the "seam continuity" property tests in
``tests/test_gwd.py`` check directly. As ``w -> h``, ``Sigma`` tends to an ISOTROPIC
matrix (a multiple of the identity), so the (now unidentifiable) long-edge angle stops
mattering to ``Sigma`` at all -- the "square-safety" the coordinate-wise baselines lack.

The score: the squared 2-Wasserstein distance between the two Gaussians
--------------------------------------------------------------------------
::

    GWD^2(N(mu1,Sigma1), N(mu2,Sigma2))
        = ||mu1 - mu2||^2 + Bures^2(Sigma1, Sigma2)
    Bures^2(Sigma1, Sigma2) = Tr(Sigma1) + Tr(Sigma2)
        - 2 * Tr( (Sigma1^{1/2} Sigma2 Sigma1^{1/2})^{1/2} )

The general Bures term needs a matrix square root; for 2x2 SPD (or PSD) matrices it has
a CLOSED FORM avoiding eigendecomposition entirely (:func:`bures_sq`'s docstring derives
it): for any 2x2 PSD matrix ``A`` with eigenvalues ``l1, l2 >= 0``,
``Tr(sqrt(A)) = sqrt(l1) + sqrt(l2)``, and
``(sqrt(l1) + sqrt(l2))^2 = Tr(A) + 2*sqrt(det(A))``, so
``Tr(sqrt(A)) = sqrt(Tr(A) + 2*sqrt(det(A)))``. Applying this with
``A = Sigma1^{1/2} Sigma2 Sigma1^{1/2}`` and the cyclic-trace identity
``Tr(Sigma1^{1/2} Sigma2 Sigma1^{1/2}) = Tr(Sigma1 Sigma2)`` plus
``det(A) = det(Sigma1) * det(Sigma2)`` gives::

    Bures^2(Sigma1, Sigma2) = Tr(Sigma1) + Tr(Sigma2)
        - 2 * sqrt( Tr(Sigma1 @ Sigma2) + 2*sqrt(det(Sigma1) * det(Sigma2)) )

-- pure trace/determinant/matrix-product arithmetic, fully vectorized, no eigendecomp.

Decomposition (diagnostic-only -- never conformalized, design §2.2)
------------------------------------------------------------------------
``GWD^2 = center_term + shape_term`` where ``center_term = ||mu1-mu2||^2`` (localization)
and ``shape_term = Bures^2(Sigma1, Sigma2)`` (size + orientation, entangled). This split
is reported for diagnosis (:func:`gwd_decompose`) but ONLY THE SCALAR GWD carries the
split-conformal coverage guarantee; per-component thresholds are never certified.

Scale normalization
---------------------
``scale_norm="sqrt-area"`` divides the raw GWD by ``sqrt(w_pred * h_pred)`` (the
predicted box's own scale) so DOTA's ~3-order-of-magnitude object-size range does not
let large objects dominate the calibration quantile. Preregistered, ablatable
(design §2.2).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = [
    "canonicalize_le90",
    "obb_to_gaussian",
    "bures_sq",
    "gwd_sq",
    "gwd",
    "gwd_decompose",
    "obb_gwd",
]

_HALF_PI = np.pi / 2.0


def canonicalize_le90(
    w: np.ndarray, h: np.ndarray, theta: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonicalize ``(w, h, theta)`` to le90 long-edge form: ``w >= h`` and
    ``theta in [-pi/2, pi/2)``.

    Accepts ANY side-length assignment and ANY angle (any convention/period); swaps
    ``w``/``h`` and rotates ``theta`` by +90 degrees when ``h > w`` (long-edge relabeling
    is a physical no-op on the rectangle), then wraps ``theta`` into ``[-pi/2, pi/2)`` via
    the ``R / pi*Z`` quotient. On the measure-zero square set (``w == h``) the swap branch
    is simply not taken (``h > w`` is strict); ``theta`` is passed through unchanged and
    wrapped -- the orientation is genuinely unidentifiable there, which is exactly the
    degeneracy :func:`obb_to_gaussian` absorbs into an isotropic ``Sigma``.
    """
    w = np.asarray(w, dtype=float)
    h = np.asarray(h, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if np.any(w < 0) or np.any(h < 0):
        raise ValueError("canonicalize_le90: w and h must be non-negative")
    swap = h > w
    w2 = np.where(swap, h, w)
    h2 = np.where(swap, w, h)
    theta2 = np.where(swap, theta + _HALF_PI, theta)
    # Wrap into [-pi/2, pi/2) via the R/pi*Z quotient.
    theta2 = np.mod(theta2 + _HALF_PI, np.pi) - _HALF_PI
    return w2, h2, theta2


def obb_to_gaussian(
    cx: np.ndarray,
    cy: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    theta: np.ndarray,
    canonicalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map OBB parameters to ``(mu, Sigma)``.

    Parameters
    ----------
    cx, cy, w, h, theta:
        Broadcastable arrays (any common shape, including scalars).
    canonicalize:
        Apply :func:`canonicalize_le90` to ``(w, h, theta)`` first (default ``True``).
        Set ``False`` only when the caller has already canonicalized (e.g. inside a
        vectorized loop that canonicalized once upstream) -- ``Sigma`` is well-defined
        either way (it does not care about the long/short labeling), but
        :func:`canonicalize_le90`'s injectivity precondition (design §2.2) is what makes
        GWD a proper metric ON BOXES (not just on Gaussians), so callers scoring actual
        OBBs should leave this ``True``.

    Returns
    -------
    mu:
        Array of shape ``(*batch, 2)``.
    Sigma:
        Array of shape ``(*batch, 2, 2)``, symmetric PSD.
    """
    cx = np.asarray(cx, dtype=float)
    cy = np.asarray(cy, dtype=float)
    if canonicalize:
        w, h, theta = canonicalize_le90(w, h, theta)
    else:
        w = np.asarray(w, dtype=float)
        h = np.asarray(h, dtype=float)
        theta = np.asarray(theta, dtype=float)

    cx, cy, w, h, theta = np.broadcast_arrays(cx, cy, w, h, theta)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    a = (w / 2.0) ** 2
    b = (h / 2.0) ** 2

    s11 = a * cos_t ** 2 + b * sin_t ** 2
    s22 = a * sin_t ** 2 + b * cos_t ** 2
    s12 = (a - b) * sin_t * cos_t

    mu = np.stack([cx, cy], axis=-1)
    Sigma = np.stack(
        [
            np.stack([s11, s12], axis=-1),
            np.stack([s12, s22], axis=-1),
        ],
        axis=-2,
    )
    return mu, Sigma


def _trace2x2(A: np.ndarray) -> np.ndarray:
    return A[..., 0, 0] + A[..., 1, 1]


def _det2x2(A: np.ndarray) -> np.ndarray:
    return A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]


def bures_sq(Sigma1: np.ndarray, Sigma2: np.ndarray) -> np.ndarray:
    """Squared Bures distance between batches of 2x2 PSD matrices (closed form; see
    module docstring for the derivation). Clipped to ``>= 0`` (floating-point guard
    only -- the closed form is exact and non-negative for genuinely PSD inputs)."""
    Sigma1 = np.asarray(Sigma1, dtype=float)
    Sigma2 = np.asarray(Sigma2, dtype=float)
    tr1 = _trace2x2(Sigma1)
    tr2 = _trace2x2(Sigma2)
    prod = np.matmul(Sigma1, Sigma2)
    tr_prod = _trace2x2(prod)
    det1 = _det2x2(Sigma1)
    det2 = _det2x2(Sigma2)
    det_prod = np.clip(det1 * det2, 0.0, None)  # guard tiny negative fp noise
    inner = tr_prod + 2.0 * np.sqrt(det_prod)
    inner = np.clip(inner, 0.0, None)
    cross = np.sqrt(inner)
    b2 = tr1 + tr2 - 2.0 * cross
    return np.clip(b2, 0.0, None)


def gwd_sq(mu1: np.ndarray, Sigma1: np.ndarray, mu2: np.ndarray, Sigma2: np.ndarray) -> np.ndarray:
    """Squared GWD between batches of Gaussians ``N(mu1,Sigma1)``, ``N(mu2,Sigma2)``."""
    mu1 = np.asarray(mu1, dtype=float)
    mu2 = np.asarray(mu2, dtype=float)
    center = np.sum((mu1 - mu2) ** 2, axis=-1)
    return center + bures_sq(Sigma1, Sigma2)


def gwd(mu1: np.ndarray, Sigma1: np.ndarray, mu2: np.ndarray, Sigma2: np.ndarray) -> np.ndarray:
    """GWD (not squared) between batches of Gaussians."""
    return np.sqrt(gwd_sq(mu1, Sigma1, mu2, Sigma2))


def gwd_decompose(
    mu1: np.ndarray, Sigma1: np.ndarray, mu2: np.ndarray, Sigma2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """DIAGNOSTIC-ONLY split of squared GWD into ``(center_term, shape_term)``
    (design §2.2: "never conformalized into per-component certificates"). Returns
    ``(||mu1-mu2||^2, Bures^2(Sigma1,Sigma2))``; their sum is :func:`gwd_sq`'s output."""
    mu1 = np.asarray(mu1, dtype=float)
    mu2 = np.asarray(mu2, dtype=float)
    center = np.sum((mu1 - mu2) ** 2, axis=-1)
    shape = bures_sq(Sigma1, Sigma2)
    return center, shape


def obb_gwd(
    obb1: np.ndarray,
    obb2: np.ndarray,
    scale_norm: Optional[str] = None,
) -> np.ndarray:
    """GWD nonconformity directly between OBB parameter arrays.

    Parameters
    ----------
    obb1, obb2:
        Arrays of shape ``(..., 5)``: ``(cx, cy, w, h, theta)``, radians. ``obb1`` is
        conventionally the prediction, ``obb2`` the ground truth (order does not matter
        -- GWD is symmetric).
    scale_norm:
        ``None`` (raw GWD, pixel units) or ``"sqrt-area"`` (divide by
        ``sqrt(w1 * h1)``, i.e. the PREDICTION's own scale -- see module docstring).

    Returns
    -------
    ndarray of shape ``(...,)``.
    """
    obb1 = np.asarray(obb1, dtype=float)
    obb2 = np.asarray(obb2, dtype=float)
    if obb1.shape[-1] != 5 or obb2.shape[-1] != 5:
        raise ValueError("obb_gwd: last axis of obb1/obb2 must be size 5 (cx,cy,w,h,theta)")
    mu1, Sigma1 = obb_to_gaussian(obb1[..., 0], obb1[..., 1], obb1[..., 2], obb1[..., 3], obb1[..., 4])
    mu2, Sigma2 = obb_to_gaussian(obb2[..., 0], obb2[..., 1], obb2[..., 2], obb2[..., 3], obb2[..., 4])
    d = gwd(mu1, Sigma1, mu2, Sigma2)
    if scale_norm is None:
        return d
    if scale_norm == "sqrt-area":
        w1c, h1c, _ = canonicalize_le90(obb1[..., 2], obb1[..., 3], obb1[..., 4])
        scale = np.sqrt(np.clip(w1c * h1c, 1e-12, None))
        return d / scale
    raise ValueError(f"obb_gwd: unknown scale_norm {scale_norm!r}; use None or 'sqrt-area'")

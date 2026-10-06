"""Closed-form readouts of the GWD ball (revision spec §3 C2).

For a predicted box with long/short sides ``w >= h`` and GWD radius ``q``:

* the center offset is certified within ``q`` (inclusion argument, unchanged);
* the exact orientation projection over ALL boxes in the ball (any width/height) is
  ``Theta(q) = 0.5*arccos( sqrt(((a+b)/2 - q^2)^2 - ab) / ((a-b)/2) )`` with
  ``a = w^2/4, b = h^2/4``, valid for ``q < (w - h)/(2*sqrt(2))``; at or above that radius
  the ball contains near-square boxes of every angle and orientation is unconstrained.

Derivation: with ``z = (sqrt(a'), sqrt(b'))`` the squared Bures distance at relative
angle ``phi`` is ``a + b + |z|^2 - 2*sqrt(z^T P z)``,
``P = [[a c^2 + b s^2, sqrt(ab)], [sqrt(ab), a s^2 + b c^2]]``; minimizing over the long-edge
cone ``z1 >= z2 >= 0`` gives ``a + b - lambda_max(P)`` for ``phi <= pi/4`` and the isotropic
boundary value ``(sqrt(a) - sqrt(b))^2 / 2`` for ``phi >= pi/4``.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

__all__ = [
    "unconstrained_radius",
    "min_bures_at_angle",
    "orientation_half_extent",
    "orientation_half_extent_array",
    "center_slice_area",
]


def _long_short(w: float, h: float):
    w, h = float(w), float(h)
    if w < 0 or h < 0:
        raise ValueError("w and h must be non-negative")
    return (w, h) if w >= h else (h, w)


def unconstrained_radius(w: float, h: float) -> float:
    lw, sh = _long_short(w, h)
    return (lw - sh) / (2.0 * math.sqrt(2.0))


def min_bures_at_angle(w: float, h: float, phi: float) -> float:
    lw, sh = _long_short(w, h)
    phi = abs(float(phi)) % math.pi
    phi = min(phi, math.pi - phi)
    a, b = lw * lw / 4.0, sh * sh / 4.0
    if phi >= math.pi / 4:
        return unconstrained_radius(lw, sh)
    c2 = math.cos(2.0 * phi) ** 2
    d2 = (a + b) / 2.0 - math.sqrt(((a - b) / 2.0) ** 2 * c2 + a * b)
    return math.sqrt(max(d2, 0.0))


def orientation_half_extent(w: float, h: float, q: float) -> float:
    lw, sh = _long_short(w, h)
    q = float(q)
    if q < 0:
        raise ValueError("q must be non-negative")
    r = unconstrained_radius(lw, sh)
    if q >= r:
        return math.pi / 2.0
    a, b = lw * lw / 4.0, sh * sh / 4.0
    t = ((a + b) / 2.0 - q * q) ** 2 - a * b
    ratio = math.sqrt(max(t, 0.0)) / ((a - b) / 2.0)
    return 0.5 * math.acos(min(1.0, max(-1.0, ratio)))


def orientation_half_extent_array(w, h, q) -> np.ndarray:
    w, h, q = np.broadcast_arrays(np.asarray(w, float), np.asarray(h, float), np.asarray(q, float))
    return np.array([orientation_half_extent(a, b, c) for a, b, c in zip(w.ravel(), h.ravel(), q.ravel())]).reshape(w.shape)


def center_slice_area(covers_shift: Callable[[np.ndarray, np.ndarray], np.ndarray], n_dirs: int = 360,
                      r0: float = 1.0, iters: int = 40) -> float:
    """Area of {(dx,dy): shifted prediction is in the region}, assuming star-shape about 0.

    Per direction: find an outside radius by doubling, then bisect."""
    ang = np.linspace(0.0, 2.0 * math.pi, n_dirs, endpoint=False)
    ux, uy = np.cos(ang), np.sin(ang)
    lo = np.zeros(n_dirs)
    hi = np.full(n_dirs, float(r0))
    for _ in range(60):
        inside = covers_shift(hi * ux, hi * uy)
        if not inside.any():
            break
        hi = np.where(inside, hi * 2.0, hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        inside = covers_shift(mid * ux, mid * uy)
        lo = np.where(inside, mid, lo)
        hi = np.where(inside, hi, mid)
    r = 0.5 * (lo + hi)
    return float(0.5 * np.sum(r * r) * (2.0 * math.pi / n_dirs))

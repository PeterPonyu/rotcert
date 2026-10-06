import math
import numpy as np
import pytest
from rotcert import geometry as G
from rotcert.gwd import obb_gwd


def test_unconstrained_radius():
    assert G.unconstrained_radius(10.0, 2.0) == pytest.approx(8 / (2 * math.sqrt(2)))
    assert G.unconstrained_radius(2.0, 10.0) == pytest.approx(8 / (2 * math.sqrt(2)))
    assert G.unconstrained_radius(3.0, 3.0) == 0.0


def test_min_bures_endpoints():
    assert G.min_bures_at_angle(10, 2, 0.0) == pytest.approx(0.0, abs=1e-12)
    for phi in (math.pi / 4, 1.0, math.pi / 2):
        assert G.min_bures_at_angle(10, 2, phi) == pytest.approx(G.unconstrained_radius(10, 2))


def test_min_bures_matches_brute_force():
    # brute force over w' >= h' at fixed angle using the library GWD with zero center offset
    rng = np.random.default_rng(0)
    for _ in range(20):
        w = rng.uniform(5, 60)
        h = rng.uniform(1, w)
        phi = rng.uniform(0, math.pi / 4)
        ws = np.linspace(0.05, 1.6, 180) * w
        best = math.inf
        best_wp = best_hp = None
        for wp in ws:
            hs = np.linspace(0.01, 1.0, 120) * wp
            pred = np.tile([0, 0, w, h, 0.0], (hs.size, 1))
            cand = np.column_stack([np.zeros_like(hs), np.zeros_like(hs), np.full_like(hs, wp), hs, np.full_like(hs, phi)])
            vals = obb_gwd(pred, cand)
            i = int(np.argmin(vals))
            if float(vals[i]) < best:
                best, best_wp, best_hp = float(vals[i]), float(wp), float(hs[i])
        # Local zoom-refinement: the coarse 180x120 grid above spans wide relative ranges
        # of (w', h') at fixed absolute resolution, so it under-resolves the true minimum
        # whenever the optimal shape sits in a narrow well close to (w, h) -- which happens
        # for small phi (the well narrows as phi -> 0, since D(phi) -> 0 there too). Zoom
        # into a shrinking window around the coarse best, still using only the library
        # obb_gwd (never the closed form), so `best` tracks the true continuum minimum
        # regardless of where in the randomized (w, h, phi) domain a given draw lands.
        window_w = ws[1] - ws[0]
        window_h = window_w * (best_hp / max(best_wp, 1e-9))
        for _ in range(6):
            wps = np.clip(best_wp + np.linspace(-window_w, window_w, 21), 1e-6, None)
            hps = np.clip(best_hp + np.linspace(-window_h, window_h, 21), 1e-6, None)
            WP, HP = np.meshgrid(wps, hps, indexing="ij")
            HP = np.minimum(HP, WP)  # keep w' >= h' so phi still means the long-edge angle
            pred = np.tile([0, 0, w, h, 0.0], (WP.size, 1))
            cand = np.column_stack([np.zeros(WP.size), np.zeros(WP.size), WP.ravel(), HP.ravel(), np.full(WP.size, phi)])
            vals = obb_gwd(pred, cand)
            i = int(np.argmin(vals))
            if float(vals[i]) < best:
                best, best_wp, best_hp = float(vals[i]), float(WP.ravel()[i]), float(HP.ravel()[i])
            window_w /= 5.0
            window_h /= 5.0
        assert G.min_bures_at_angle(w, h, phi) <= best + 1e-9
        assert G.min_bures_at_angle(w, h, phi) >= best - 0.02 * max(1.0, best)


def test_half_extent_counterexample_2026_09_20():
    th = G.orientation_half_extent(10.0, 2.0, 2.38)
    assert math.degrees(th) == pytest.approx(31.72, abs=0.05)
    # the fixed-shape readout (29.45 deg) under-covers: a 30-deg shrunken box is inside
    assert obb_gwd(np.array([0, 0, 10, 2, 0.0]), np.array([0, 0, 8.871202, 1.774240, math.radians(30)])) < 2.38


def test_half_extent_monotone_and_jump():
    w, h = 20.0, 4.0
    r = G.unconstrained_radius(w, h)
    qs = np.linspace(0, r * 0.999, 50)
    th = [G.orientation_half_extent(w, h, q) for q in qs]
    assert all(b >= a - 1e-12 for a, b in zip(th, th[1:]))
    assert th[-1] < math.pi / 4 + 1e-6
    assert G.orientation_half_extent(w, h, r) == pytest.approx(math.pi / 2)


def test_half_extent_is_a_valid_projection():
    # every random box inside the ball has |dtheta| <= Theta(q) (or the ball is unconstrained)
    rng = np.random.default_rng(1)
    w, h, q = 30.0, 6.0, 5.0
    th = G.orientation_half_extent(w, h, q)
    n = 200000
    cand = np.column_stack([rng.normal(0, 2, n), rng.normal(0, 2, n),
                            rng.uniform(20, 40, n), rng.uniform(1, 12, n),
                            rng.uniform(-math.pi / 2, math.pi / 2, n)])
    pred = np.tile([0, 0, w, h, 0.0], (cand.shape[0], 1))
    inside = obb_gwd(pred, cand) <= q
    from rotcert.gwd import canonicalize_le90
    wc, hc, tc = canonicalize_le90(cand[:, 2], cand[:, 3], cand[:, 4])
    dth = np.abs(np.mod(tc + math.pi / 2, math.pi) - math.pi / 2)
    assert inside.sum() > 100
    assert np.all(dth[inside] <= th + 1e-9)


def test_center_slice_area_disk():
    area = G.center_slice_area(lambda dx, dy: np.hypot(dx, dy) <= 3.0)
    assert area == pytest.approx(math.pi * 9.0, rel=2e-3)

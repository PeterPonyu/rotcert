import math
import numpy as np
import pytest
from rotcert import scores_ext as X


def _boxes(n=50, seed=0):
    rng = np.random.default_rng(seed)
    p = np.column_stack([rng.uniform(0, 500, n), rng.uniform(0, 500, n), rng.uniform(10, 80, n),
                         rng.uniform(3, 40, n), rng.uniform(-math.pi / 2, math.pi / 2, n)])
    g = p + np.column_stack([rng.normal(0, 2, n), rng.normal(0, 2, n), rng.normal(0, 2, n),
                             rng.normal(0, 1, n), rng.normal(0, 0.05, n)])
    g[:, 2:4] = np.abs(g[:, 2:4]) + 1.0
    return p, g


@pytest.mark.parametrize("name", sorted(X.EXT_SCORES))
def test_zero_on_identical_and_nonnegative(name):
    p, g = _boxes()
    f = X.EXT_SCORES[name]
    assert np.allclose(f(p, p), 0.0, atol=1e-7)
    assert np.all(f(p, g) >= -1e-9)


NOT_SQUARE_SAFE = {"coord_max_additive", "coord_max_multiplicative"}


@pytest.mark.parametrize("name", sorted(X.EXT_SCORES))
def test_seam_continuity(name):
    f = X.EXT_SCORES[name]
    a = np.array([[0, 0, 40, 10, math.radians(-89.9)]])
    b = np.array([[0, 0, 40, 10, math.radians(89.9)]])
    assert f(a, b)[0] < 0.05 * max(1.0, float(f(a, np.array([[0, 0, 40, 10, 0.3]]))[0]))


@pytest.mark.parametrize("name", [
    pytest.param(n, marks=pytest.mark.xfail(strict=True, reason="angle residual is not square-safe by construction"))
    if n in NOT_SQUARE_SAFE else n
    for n in sorted(X.EXT_SCORES)
])
def test_square_safety(name):
    f = X.EXT_SCORES[name]
    s0 = np.array([[5, 5, 20, 20, 0.0]])
    s90 = np.array([[5, 5, 20, 20, math.pi / 2]])
    assert f(s0, s90)[0] == pytest.approx(0.0, abs=1e-6)


def test_expansion_margin_semantics():
    pred = np.array([[0, 0, 10, 4, 0.0]])
    inside = np.array([[0, 0, 8, 2, 0.0]])
    shifted = np.array([[3, 0, 10, 4, 0.0]])   # right edge pokes out by 3
    assert X.expansion_margin(pred, inside)[0] == 0.0
    assert X.expansion_margin(pred, shifted)[0] == pytest.approx(3.0)


def test_corner_max_translation():
    pred = np.array([[0, 0, 10, 4, 0.3]])
    gt = np.array([[3, 4, 10, 4, 0.3]])
    assert X.corner_max(pred, gt)[0] == pytest.approx(5.0)


def test_kld_known_translation():
    # equal covariances: KL = 0.5 * mahalanobis^2 under the GT covariance
    pred = np.array([[1, 0, 4, 2, 0.0]])
    gt = np.array([[0, 0, 4, 2, 0.0]])
    # Sigma = diag(4, 1); dx = 1 -> 0.5 * 1/4
    assert X.kld(pred, gt)[0] == pytest.approx(0.125)


def test_hellinger_bounds_and_bhattacharyya_relation():
    p, g = _boxes()
    hd = X.hellinger(p, g)
    bd = X.bhattacharyya(p, g)
    assert np.all((hd >= 0) & (hd <= 1))
    assert np.allclose(hd, np.sqrt(1 - np.exp(-bd)))


def test_maxrank_fit_score_range():
    rng = np.random.default_rng(0)
    design = np.abs(rng.normal(size=(500, 5)))
    mr = X.MaxRank.fit(design)
    s = mr.score(np.abs(rng.normal(size=(100, 5))))
    assert np.all((s >= 0) & (s <= 1))
    assert mr.score(np.zeros((1, 5)))[0] == 0.0


# --- Fix round 1: hand-computed known-value tests (guard against constant/sign regressions) ---


def test_bhattacharyya_hellinger_known_translation():
    # w=4,h=2,theta=0 canonicalizes unchanged for both -> Sigma_pred = Sigma_gt = diag(4, 1).
    # Equal covariances make the log-det term 0, so bhattacharyya = 0.125 * mahalanobis^2
    # under s = Sigma (since s = 0.5*(Sigma+Sigma) = Sigma). mu differs by (1, 0):
    # mahalanobis^2 = 1^2/4 + 0^2/1 = 0.25 -> bhattacharyya = 0.125 * 0.25 = 0.03125.
    pred = np.array([[1, 0, 4, 2, 0.0]])
    gt = np.array([[0, 0, 4, 2, 0.0]])
    expected_bhat = 0.125 * 0.25
    assert X.bhattacharyya(pred, gt)[0] == pytest.approx(expected_bhat)
    assert X.hellinger(pred, gt)[0] == pytest.approx(math.sqrt(1 - math.exp(-expected_bhat)))


def test_bhattacharyya_known_shape_difference():
    # Sigma_pred = diag(4, 1) (w=4,h=2); Sigma_gt = diag(1, 1) (w=h=2 -> isotropic, square).
    # Same center -> mahalanobis term is 0. s = 0.5*(diag(4,1)+diag(1,1)) = diag(2.5, 1):
    # bhattacharyya = 0.5*(ln(det s) - 0.5*(ln(det Sigma_pred) + ln(det Sigma_gt)))
    #              = 0.5*(ln(2.5) - 0.5*ln(4)) since det(s)=2.5, det(Sigma_pred)=4, det(Sigma_gt)=1.
    pred = np.array([[0, 0, 4, 2, 0.0]])
    gt = np.array([[0, 0, 2, 2, 0.0]])
    expected = 0.5 * (math.log(2.5) - 0.5 * math.log(4.0))
    assert X.bhattacharyya(pred, gt)[0] == pytest.approx(expected)


def test_coord_max_known_translation_and_angle():
    # pred theta=0 -> prediction frame == world frame, so the rotated offset is just (dx, dy)
    # = (1, 2); dtheta=0.1 rad is already within (-pi/2, pi/2], so wrapping is a no-op.
    pred = np.array([[0, 0, 10, 4, 0.0]])
    gt = np.array([[1, 2, 10, 4, 0.1]])
    expected_additive = max(1.0, 2.0, 0.0, 0.0, 0.5 * 10 * 0.1)  # = max(1, 2, 0, 0, 0.5) = 2.0
    expected_mult = max(1.0 / 10, 2.0 / 4, 0.0, 0.0, 0.5 * 0.1)  # = max(0.1, 0.5, 0, 0, 0.05) = 0.5
    assert X.coord_max_additive(pred, gt)[0] == pytest.approx(expected_additive)
    assert X.coord_max_multiplicative(pred, gt)[0] == pytest.approx(expected_mult)


def test_coord_max_additive_arc_term_dominates():
    # Only theta differs (0 -> 0.5 rad, no wrap needed); the arc-length term 0.5*pw*dtheta
    # = 0.5*10*0.5 = 2.5 dominates the (zero) translation/size terms.
    pred = np.array([[0, 0, 10, 4, 0.0]])
    gt = np.array([[0, 0, 10, 4, 0.5]])
    assert X.coord_max_additive(pred, gt)[0] == pytest.approx(2.5)


def test_coord_max_rotated_prediction_frame():
    # theta=pi/2 canonicalizes to -pi/2 (w=10 > h=4, no swap; wraps into [-pi/2,pi/2)).
    # At t=-pi/2: c=cos(-pi/2)=0, s=sin(-pi/2)=-1, so a world-frame shift of dx=1 rotates to
    # local (lx, ly) = (c*dx, -s*dx) = (0, 1) -- i.e. it lands entirely on the SHORT axis (h),
    # not the long axis (w), because the box's long edge now points along world y.
    pred = np.array([[0, 0, 10, 4, math.pi / 2]])
    gt = np.array([[1, 0, 10, 4, math.pi / 2]])
    assert X.coord_max_additive(pred, gt)[0] == pytest.approx(1.0)
    assert X.coord_max_multiplicative(pred, gt)[0] == pytest.approx(1.0 / 4)


def test_maxrank_interior_values():
    # Each of the 5 columns sorts to exactly [0, 1, 2, 3, 4] (n=5).
    design = np.tile(np.arange(5.0)[:, None], (1, 5))
    mr = X.MaxRank.fit(design)
    # searchsorted([0,1,2,3,4], 2.5, side="left") = 3 (0,1,2 are < 2.5) -> cdf = 3/5 = 0.6
    assert mr.score(np.array([[2.5] * 5]))[0] == pytest.approx(0.6)
    # column 0: searchsorted([0,1,2,3,4], 4.0, side="left") = 4 (0,1,2,3 are < 4.0) -> 4/5=0.8
    # columns 1-4: searchsorted(..., 0.0, side="left") = 0 -> cdf = 0
    assert mr.score(np.array([[4.0, 0, 0, 0, 0]]))[0] == pytest.approx(0.8)


def test_expansion_margin_rotated_along_long_axis():
    # theta=pi/2 canonicalizes to -pi/2 for both boxes, making w=10 the box's VERTICAL extent.
    # Shifting gt's center by 3px along that (vertical) world-y axis is a pure shift along the
    # prediction's own long axis, so the far corners poke out by exactly 3px.
    pred = np.array([[0, 0, 10, 4, math.pi / 2]])
    gt = np.array([[0, 3, 10, 4, math.pi / 2]])
    assert X.expansion_margin(pred, gt)[0] == pytest.approx(3.0)

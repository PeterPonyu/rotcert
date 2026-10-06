import math
import numpy as np
import pytest
from rotcert import g2
from rotcert.ltt import eb_pvalue


def test_fixed_grid_shape():
    assert len(g2.FIXED_LAMBDA_GRID) == 50
    assert g2.FIXED_LAMBDA_GRID[0] == 0.01 and g2.FIXED_LAMBDA_GRID[-1] == 0.99


def test_rejects_data_dependent_grid():
    with pytest.raises(g2.GridProvenanceError):
        g2.certify_risk(np.zeros((10, 2)), [0.1, 0.2], 0.2, 0.05, grid_source="quantiles_of_same_data")


def test_certifies_easy_case_and_selects_smallest_lambda():
    n = 3000
    rm = np.column_stack([np.full(n, 0.5), np.full(n, 0.02), np.full(n, 0.01)])
    out = g2.certify_risk(rm, [0.1, 0.5, 0.9], beta=0.2, delta=0.05)
    assert out["certified"] and out["lambda_star"] == 0.5
    assert out["level"] == pytest.approx(0.05 / 3)


def test_family_size_divides_level():
    rm = np.full((50, 2), 0.1)
    out = g2.certify_risk(rm, [0.1, 0.2], 0.2, 0.05, family_size=10)
    assert out["level"] == pytest.approx(0.05 / 20)


def test_eb_min_images_consistent_with_pvalue():
    r, v, beta, eta = 0.10, 0.005, 0.20, 0.001
    n = int(g2.eb_min_images(r, v, beta, eta))
    assert 50 < n < 10**5

    def sample(k):
        # exact mean r and exact sample variance v (ddof=1), no clipping needed
        half = k // 2
        d = math.sqrt(v * (k - 1) / (2 * half))
        y = np.array([r - d] * half + [r + d] * half + ([r] if k % 2 else []))
        assert y.min() >= 0 and abs(y.mean() - r) < 1e-12 and abs(y.var(ddof=1) - v) < 1e-12
        return y

    assert eb_pvalue(sample(n), beta) <= eta * (1 + 1e-9)
    assert eb_pvalue(sample(n - 1), beta) > eta * (1 - 1e-9)


def test_eb_necessary_matches_manuscript_eq13():
    # Eq. (13): n >= 1 + 7 ln(2G/delta) / (3 (beta - r)) with eta = delta/G
    G, delta, beta, r = 50, 0.05, 0.20, 0.17
    eta = delta / G
    assert g2.eb_necessary_images(r, beta, eta) == pytest.approx(1 + 7 * math.log(2 * G / delta) / (3 * (beta - r)))
    assert math.isinf(g2.eb_necessary_images(0.25, 0.20, eta))


# --- Fix round 1: controller-required coverage ---------------------------------------


def test_lambda_max_stops_at_last_rejected_not_last_grid_point():
    # Finding 1 (reviewer Minor 2): lambda_star alone hides how far the valid range
    # extends. Column 0.5 has mean risk 0.5 > beta and must NOT reject, so lambda_max
    # (largest REJECTED) must stop at 0.02, not run to the last grid point 0.5.
    n = 3000
    rm = np.column_stack([np.full(n, 0.01), np.full(n, 0.02), np.full(n, 0.5)])
    out = g2.certify_risk(rm, [0.01, 0.02, 0.5], beta=0.2, delta=0.05)
    assert out["certified"]
    assert out["lambda_star"] == 0.01 and out["realized_risk"] == pytest.approx(0.01)
    assert out["lambda_max"] == 0.02 and out["realized_risk_max"] == pytest.approx(0.02)
    assert out["n_rejected"] == 2


def test_rejects_non_increasing_or_non_finite_grid():
    # Finding 2 (reviewer Minor 3): certify_risk must not silently re-sort/permute the
    # grid against risk_matrix's columns (image_risk_matrix's columns are always
    # ascending; a caller-supplied out-of-order grid would otherwise mislabel a
    # column's risk under the wrong lambda -- a silent false certificate).
    with pytest.raises(ValueError):
        g2.certify_risk(np.zeros((5, 3)), [0.5, 0.1, 0.9], 0.2, 0.05)
    with pytest.raises(ValueError):
        g2.certify_risk(np.zeros((5, 3)), [0.1, math.inf, 0.9], 0.2, 0.05)


def test_bonferroni_uses_level_not_delta_family_size_flips_certified():
    # Finding 4a (reviewer Minor 1a): checking `p <= delta` instead of `p <= level`
    # passes every pre-existing test. Here p ~= 0.0105 is <= delta (0.05) but > the
    # family_size=10 level (0.005): family_size must flip `certified` on identical data.
    rm = np.full((50, 1), 0.1)
    small = g2.certify_risk(rm, [0.5], beta=0.35, delta=0.05, family_size=1)
    large = g2.certify_risk(rm, [0.5], beta=0.35, delta=0.05, family_size=10)
    p = small["trace"][0]["p_value"]
    assert large["level"] < p <= 0.05
    assert small["certified"] is True
    assert large["certified"] is False


def test_accepts_design_split_grid_source():
    # Finding 4c: "design_split" (grid built on scenes disjoint from the tested ones)
    # must be accepted, not just "fixed".
    n = 3000
    rm = np.column_stack([np.full(n, 0.01), np.full(n, 0.5)])
    out = g2.certify_risk(rm, [0.1, 0.5], beta=0.2, delta=0.05, grid_source="design_split")
    assert out["certified"] and out["grid_source"] == "design_split"


def test_hb_p_value_path():
    # Finding 4c: the "hb" p-value path is untested.
    n = 3000
    rm = np.column_stack([np.full(n, 0.01), np.full(n, 0.5)])
    out = g2.certify_risk(rm, [0.1, 0.5], beta=0.2, delta=0.05, p_value="hb")
    assert out["certified"] and out["lambda_star"] == 0.1 and out["p_value"] == "hb"


def test_eb_min_images_infeasible_paths():
    # Finding 4c: both the beta<=r short-circuit and the n_max-too-small path are
    # untested.
    assert math.isinf(g2.eb_min_images(0.5, 0.01, 0.2, 0.001))  # beta <= r
    assert math.isinf(g2.eb_min_images(0.19, 0.0, 0.2, 0.001, n_max=10))  # n_max too small

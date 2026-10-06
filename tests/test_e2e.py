import numpy as np
import pytest
from rotcert import e2e


def test_loss_matrix_semantics():
    recs = [(np.array([0.9, 0.2, np.nan]), np.array([1.0, 1.0, np.nan]))]
    lm = e2e.e2e_loss_matrix(recs, [0.5, 2.0], lam=0.5)
    # q=0.5: nothing localized -> loss 1; q=2.0: only the 0.9-conf GT counts -> 1 - 1/3
    assert lm.tolist() == [[1.0, pytest.approx(2 / 3)]]


def test_loss_monotone_in_q():
    rng = np.random.default_rng(0)
    recs = [(rng.uniform(size=5), rng.exponential(size=5)) for _ in range(30)]
    lm = e2e.e2e_loss_matrix(recs, e2e.FIXED_Q_GRID_PX, lam=0.3)
    assert np.all(np.diff(lm, axis=1) <= 1e-12)


def test_certify_fixed_sequence():
    n = 4000
    q = np.array([1.0, 2.0, 3.0])
    lm = np.column_stack([np.full(n, 0.9), np.full(n, 0.2), np.full(n, 0.05)])
    out = e2e.certify_e2e(lm, q, eps=0.3, delta=0.05)
    assert out["certified"] and out["q_star"] == 2.0


def test_composition_bound():
    assert e2e.composition_bound(0.1, 0.2) == pytest.approx(0.3)


# --- Fix round 1 --------------------------------------------------------------

def test_all_miss_scene_loss_is_one_at_q_infinity():
    # Minor 2: misses use +inf as a score sentinel; inf <= inf is True, so without
    # gating on `det` an all-miss scene falsely reads as "localized" at q = inf.
    recs = [(np.array([np.nan]), np.array([np.nan]))]
    lm = e2e.e2e_loss_matrix(recs, [np.inf], lam=0.5)
    assert lm.tolist() == [[1.0]]


def test_boundary_ties_are_inclusive():
    # conf == lam and score == q must both count (kills `>= -> >` and `<= -> <`).
    recs = [(np.array([0.5]), np.array([2.0]))]
    lm = e2e.e2e_loss_matrix(recs, [2.0], lam=0.5)
    assert lm.tolist() == [[0.0]]


def test_nan_confidence_alone_is_a_miss():
    # NaN confidence must never be treated as detected, even with a real score.
    recs = [(np.array([np.nan]), np.array([0.1]))]
    lm = e2e.e2e_loss_matrix(recs, [10.0], lam=0.5)
    assert lm.tolist() == [[1.0]]


def test_nan_score_alone_is_not_localized():
    # NaN score must never be treated as localized, even when detected.
    recs = [(np.array([0.9]), np.array([np.nan]))]
    lm = e2e.e2e_loss_matrix(recs, [10.0], lam=0.5)
    assert lm.tolist() == [[1.0]]


def test_empty_scene_is_dropped_not_zero_row():
    # An empty scene contributes no row at all -- it must not appear as a
    # spurious perfect-score (0.0 loss) row.
    recs = [(np.array([]), np.array([])), (np.array([0.9]), np.array([1.0]))]
    lm = e2e.e2e_loss_matrix(recs, [2.0], lam=0.5)
    assert lm.shape == (1, 1)
    assert lm.tolist() == [[0.0]]


def test_certify_nonmonotone_stops_at_first_nonrejection():
    # Non-monotone input (impossible from real e2e_loss_matrix output, but
    # certify_e2e must not assume it) exposes any scan that keeps going past
    # the first non-rejection, or that scans bottom-up.
    n = 4000
    q = np.array([1.0, 2.0, 3.0])
    lm = np.column_stack([np.full(n, 0.05), np.full(n, 0.9), np.full(n, 0.05)])
    out = e2e.certify_e2e(lm, q, eps=0.3, delta=0.05)
    assert out["certified"] and out["q_star"] == 3.0
    assert out["realized_loss"] == pytest.approx(0.05)
    trace = [(t["q"], t["rejected"]) for t in out["trace"]]
    assert trace == [(3.0, True), (2.0, False)]


def test_certify_unsorted_grid():
    # q_grid given out of order; lm columns must be permuted together with it,
    # and realized_loss must be read from the matching (not the raw) column.
    n = 4000
    q = np.array([3.0, 1.0, 2.0])
    lm = np.column_stack([np.full(n, 0.05), np.full(n, 0.9), np.full(n, 0.4)])
    out = e2e.certify_e2e(lm, q, eps=0.3, delta=0.05)
    assert out["certified"] and out["q_star"] == 3.0
    assert out["realized_loss"] == pytest.approx(0.05)


def test_certify_nothing_certifies():
    # Even the most permissive q fails outright: exactly one trace entry,
    # not a full scan of the grid.
    n = 4000
    q = np.array([1.0, 2.0, 3.0])
    lm = np.column_stack([np.full(n, 0.9), np.full(n, 0.9), np.full(n, 0.9)])
    out = e2e.certify_e2e(lm, q, eps=0.3, delta=0.05)
    assert out["certified"] is False
    assert out["q_star"] is None
    assert out["realized_loss"] is None
    assert len(out["trace"]) == 1


def test_certify_boundary_p_equals_delta_rejects(monkeypatch):
    # p == delta must reject (kills `p <= delta -> p < delta`).
    monkeypatch.setitem(e2e._PV, "eb", lambda y, a: 0.05)
    lm = np.zeros((10, 1))
    out = e2e.certify_e2e(lm, [1.0], eps=0.3, delta=0.05)
    assert out["certified"] and out["q_star"] == 1.0


def test_certify_default_p_value_is_eb(monkeypatch):
    # Default p_value must resolve to "eb", not "hb".
    monkeypatch.setitem(e2e._PV, "eb", lambda y, a: 0.0)
    monkeypatch.setitem(e2e._PV, "hb", lambda y, a: 1.0)
    lm = np.zeros((10, 2))
    out = e2e.certify_e2e(lm, [1.0, 2.0], eps=0.3, delta=0.05)
    assert out["certified"] is True
    assert out["q_star"] == 1.0

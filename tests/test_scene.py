import math
import numpy as np
import pytest
from rotcert._vendor.relmetrics.conformal import SplitConformal
from rotcert import scene as S


def _rand_groups(rng, k, max_m=6):
    return [rng.exponential(size=rng.integers(1, max_m + 1)) for _ in range(k)]


@pytest.mark.parametrize("alpha", [0.1, 1 / 3, 0.05])
def test_pooled_matches_relmetrics(alpha):
    rng = np.random.default_rng(0)
    g = _rand_groups(rng, 50)
    ref = SplitConformal(alpha=alpha, randomize=False).fit(np.concatenate(g)).threshold
    assert S.pooled_threshold(g, alpha) == ref


def test_hcp_equals_pooled_for_singleton_scenes():
    rng = np.random.default_rng(1)
    g = [np.array([v]) for v in rng.normal(size=40)]
    assert S.hcp_threshold(g, 0.1) == S.pooled_threshold(g, 0.1)


def test_hcp_floor_nine_scenes_at_alpha_point_one():
    g8 = [np.array([float(i)]) for i in range(8)]
    g9 = [np.array([float(i)]) for i in range(9)]
    assert math.isinf(S.hcp_threshold(g8, 0.1))
    assert S.hcp_threshold(g9, 0.1) == 8.0


@pytest.mark.parametrize("alpha,expected", [(0.5, 3.0), (0.4, 4.0), (0.6, 2.0)])
def test_hcp_weights_hand_computed(alpha, expected):
    # scene A: one object (weight 1/3), scene B: three objects (1/9 each), +inf mass 1/3
    g = [np.array([1.0]), np.array([2.0, 3.0, 4.0])]
    assert S.hcp_threshold(g, alpha) == expected


def test_hcp_ignores_empty_scenes():
    g = [np.array([1.0]), np.array([]), np.array([2.0, 3.0, 4.0])]
    assert S.hcp_threshold(g, 0.5) == 3.0


def test_crc_object_hand_computed():
    g = [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])]
    # budget = 0.8*5 - 0.2*3 = 3.4 -> 3 objects may exceed q -> q = 2nd smallest
    assert S.crc_object_threshold(g, 0.8, M=3) == 2.0
    # budget = 0.4*5 - 0.6*3 = 0.2 -> 0 objects may exceed q -> q = max
    assert S.crc_object_threshold(g, 0.4, M=3) == 5.0
    # budget = 0.8*5 - 0.2*10 = 2.0 -> 2 objects may exceed q -> q = 3rd smallest
    # (pins that CRC uses the supplied M, not the calibration maximum of 3)
    assert S.crc_object_threshold(g, 0.8, M=10) == 3.0


def test_crc_object_infeasible_and_oversize():
    g = [np.array([1.0, 2.0]), np.array([3.0])]
    assert math.isinf(S.crc_object_threshold(g, 0.1, M=100))
    with pytest.raises(S.SceneCalibrationError):
        S.crc_object_threshold(g, 0.1, M=1)


def test_scene_max_rank_and_floor():
    g9 = [np.array([float(i), -1.0]) for i in range(9)]
    assert S.scene_max_threshold(g9, 0.1) == 8.0
    assert math.isinf(S.scene_max_threshold(g9[:8], 0.1))


def test_coverages():
    g = [np.array([1.0, 3.0]), np.array([2.0])]
    assert S.coverage_scene_weighted(g, 2.0) == pytest.approx((0.5 + 1.0) / 2)
    assert S.coverage_object_weighted(g, 2.0) == pytest.approx(2 / 3)
    assert S.coverage_scene_simultaneous(g, 2.0) == pytest.approx(0.5)


def test_hcp_scene_weighted_validity_monte_carlo():
    # clustered data: scene random effect + heavy-tailed sizes; HCP must keep
    # scene-weighted coverage >= 1 - alpha (allow 3 Monte Carlo SE).
    rng = np.random.default_rng(7)
    alpha, reps, K = 0.1, 400, 25
    covs = []
    for _ in range(reps):
        def draw(k):
            out = []
            for _ in range(k):
                m = int(min(1 + rng.pareto(1.2) * 3, 200))
                out.append(np.exp(rng.normal(0, 1.0) + 0.3 * rng.normal(size=m)))
            return out
        q = S.hcp_threshold(draw(K), alpha)
        covs.append(S.coverage_scene_weighted(draw(300), q))
    covs = np.array(covs)
    assert covs.mean() >= 1 - alpha - 3 * covs.std(ddof=1) / np.sqrt(reps)


# --- Controller-ruling hardening (Task 1 review) ---


def test_as_groups_rejects_flat_pooled_array():
    # A flat pooled score array must never be silently treated as singleton scenes.
    with pytest.raises(S.SceneCalibrationError):
        S.hcp_threshold(np.array([1.0, 2.0, 3.0]), 0.1)


def test_pooled_threshold_casts_float32_alpha_like_relmetrics():
    rng = np.random.default_rng(11)
    g = [np.array([v]) for v in rng.exponential(size=99)]
    alpha32 = np.float32(0.29)
    ref = SplitConformal(alpha=float(alpha32), randomize=False).fit(np.concatenate(g)).threshold
    assert S.pooled_threshold(g, alpha32) == ref
    assert S.pooled_threshold(g, alpha32) == S.pooled_threshold(g, 0.29)


def test_pooled_threshold_rank_exceeds_n_returns_inf():
    g = [np.array([float(i)]) for i in range(5)]
    assert math.isinf(S.pooled_threshold(g, 0.1))


def test_empty_input_returns_inf_for_all_thresholds():
    assert math.isinf(S.pooled_threshold([], 0.1))
    assert math.isinf(S.hcp_threshold([], 0.1))
    assert math.isinf(S.scene_max_threshold([], 0.1))
    assert math.isinf(S.crc_object_threshold([], 0.1, M=5))


def test_pooled_crc_and_scene_max_ignore_empty_scenes():
    rng = np.random.default_rng(21)
    g = _rand_groups(rng, 10)
    g_plus_empty = g + [np.array([])]
    assert S.pooled_threshold(g, 0.2) == S.pooled_threshold(g_plus_empty, 0.2)

    gc = [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])]
    gc_plus_empty = gc + [np.array([])]
    assert S.crc_object_threshold(gc, 0.8, M=3) == S.crc_object_threshold(gc_plus_empty, 0.8, M=3)

    g9 = [np.array([float(i), -1.0]) for i in range(9)]
    g9_plus_empty = g9 + [np.array([])]
    # alpha=0.19 (not 0.1) is the discriminating value here: counting the empty
    # scene as a -inf "maximum" instead of ignoring it changes 8.0 -> 7.0 only
    # at this rank boundary, so this is the alpha that actually pins the ignore
    # behavior rather than passing vacuously.
    assert S.scene_max_threshold(g9, 0.19) == S.scene_max_threshold(g9_plus_empty, 0.19) == 8.0


def test_coverage_functions_ignore_empty_scenes():
    g = [np.array([1.0, 3.0]), np.array([2.0])]
    g_plus_empty = g + [np.array([])]
    assert S.coverage_scene_weighted(g, 2.0) == S.coverage_scene_weighted(g_plus_empty, 2.0)
    assert S.coverage_object_weighted(g, 2.0) == S.coverage_object_weighted(g_plus_empty, 2.0)
    assert S.coverage_scene_simultaneous(g, 2.0) == S.coverage_scene_simultaneous(g_plus_empty, 2.0)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1])
def test_invalid_alpha_raises_for_all_thresholds(alpha):
    g = [np.array([1.0]), np.array([2.0, 3.0])]
    with pytest.raises(S.SceneCalibrationError):
        S.pooled_threshold(g, alpha)
    with pytest.raises(S.SceneCalibrationError):
        S.hcp_threshold(g, alpha)
    with pytest.raises(S.SceneCalibrationError):
        S.scene_max_threshold(g, alpha)
    with pytest.raises(S.SceneCalibrationError):
        S.crc_object_threshold(g, alpha, M=5)


def test_non_finite_score_raises():
    with pytest.raises(S.SceneCalibrationError):
        S.pooled_threshold([np.array([1.0, math.inf])], 0.1)
    with pytest.raises(S.SceneCalibrationError):
        S.hcp_threshold([np.array([1.0, math.nan])], 0.1)
    with pytest.raises(S.SceneCalibrationError):
        S.scene_max_threshold([np.array([math.inf])], 0.1)
    with pytest.raises(S.SceneCalibrationError):
        S.crc_object_threshold([np.array([math.inf])], 0.1, M=5)


def test_crc_object_M_zero_raises():
    with pytest.raises(S.SceneCalibrationError):
        S.crc_object_threshold([np.array([1.0])], 0.1, M=0)


# --- Task 2: grouping, per-class thresholds, coverage evaluation ---


def test_grouped_orders_by_scene_code():
    s = np.array([5.0, 1.0, 2.0, 7.0])
    sc = np.array([3, 1, 1, 3])
    g = S.grouped(s, sc)
    assert [list(x) for x in g] == [[1.0, 2.0], [5.0, 7.0]]


def test_class_thresholds_and_evaluate_roundtrip():
    rng = np.random.default_rng(3)
    n = 600
    scene = rng.integers(0, 60, size=n)
    cls = rng.integers(0, 2, size=n)
    s = rng.exponential(size=n)
    th = S.class_thresholds(s, scene, cls, 0.1, "hcp")
    assert set(th) == {0, 1}
    assert all(v["status"] == "certified" for v in th.values())
    q = {c: v["q_hat"] for c, v in th.items()}
    ev = S.evaluate(s, scene, cls, q)
    assert ev["n_out_of_support"] == 0
    assert 0.0 < ev["overall"]["cov_scene"] <= 1.0
    assert set(ev["per_class"]) == {0, 1}


def test_evaluate_out_of_support_and_infinite_threshold():
    s = np.array([1.0, 2.0, 3.0])
    scene = np.array([0, 0, 1])
    cls = np.array([0, 1, 2])
    ev = S.evaluate(s, scene, cls, {0: 0.5, 1: math.inf})
    assert ev["n_out_of_support"] == 1
    assert ev["per_class"][0]["cov_object"] == 0.0
    assert ev["per_class"][1]["cov_object"] == 1.0


def test_class_thresholds_crc_requires_M():
    with pytest.raises(S.SceneCalibrationError):
        S.class_thresholds(np.array([1.0]), np.array([0]), np.array([0]), 0.1, "crc_object")


# --- Task 2 fix round 1: strengthen tests that pinned too little of evaluate()/class_thresholds() ---


def test_evaluate_hand_computed_all_fields():
    # Non-contiguous, unsorted scene codes; class 3 is eval-only (out of support)
    # and shares scene 5 with an in-support class-1 object, so it must not leak
    # into that scene's overall coverage.
    scores = np.array([1.0, 3.0, 2.0, 5.0, 1.0, 4.0, 9.0, 6.0])
    scenes = np.array([7, 7, 2, 2, 2, 5, 5, 7])
    classes = np.array([0, 0, 0, 1, 1, 1, 3, 0])
    ev = S.evaluate(scores, scenes, classes, {0: 2.0, 1: math.inf})
    assert ev["n_out_of_support"] == 1
    assert set(ev["per_class"]) == {0, 1}
    assert ev["per_class"][0] == pytest.approx(
        {"cov_scene": 2 / 3, "cov_object": 0.5, "cov_sim": 0.5, "n_scenes": 2, "n_objects": 4}
    )
    assert ev["per_class"][1] == pytest.approx(
        {"cov_scene": 1.0, "cov_object": 1.0, "cov_sim": 1.0, "n_scenes": 2, "n_objects": 3}
    )
    assert ev["overall"] == pytest.approx({"cov_scene": 7 / 9, "cov_object": 5 / 7, "cov_sim": 2 / 3})


def test_class_thresholds_is_mondrian_by_class():
    # Scale class-1 scores up so a pooled-across-classes threshold would visibly
    # differ from a per-class one; compare against an independent reconstruction
    # via grouped()+hcp_threshold() restricted to each class's own mask.
    rng = np.random.default_rng(3)
    scene = rng.integers(0, 60, size=600)
    cls = rng.integers(0, 2, size=600)
    s = rng.exponential(size=600)
    s[cls == 1] *= 10.0
    th = S.class_thresholds(s, scene, cls, 0.1, "hcp")
    for c in (0, 1):
        m = cls == c
        assert th[c]["q_hat"] == S.hcp_threshold(S.grouped(s[m], scene[m]), 0.1)
        assert th[c]["n_scenes"] == len(np.unique(scene[m]))
        assert th[c]["n_objects"] == int(m.sum())


def test_class_thresholds_support_floor_status():
    # Two singleton scenes at alpha=0.1: HCP cannot certify (see
    # test_hcp_floor_nine_scenes_at_alpha_point_one for the same floor at K=8).
    th = S.class_thresholds(np.array([1.0, 2.0]), np.array([0, 1]), np.array([0, 0]), 0.1, "hcp")
    assert math.isinf(th[0]["q_hat"])
    assert th[0]["status"] == "support_floor"


def test_conformal_rank_casts_float32_alpha():
    # conformal_rank() is public and callable directly, bypassing the four
    # threshold wrappers that already cast alpha -- it needs its own cast.
    n = 99
    alpha32 = np.float32(0.29)
    assert S.conformal_rank(n, alpha32) == S.conformal_rank(n, float(alpha32))


@pytest.mark.parametrize("method", ["pooled", "hcp", "scene_max", "crc_object"])
def test_float32_alpha_cast_every_threshold(method):
    rng = np.random.default_rng(11)
    g = [np.array([v]) for v in rng.exponential(size=99)]
    kwargs = {"M": 1} if method == "crc_object" else {}
    fn = S.THRESHOLDS[method]
    assert fn(g, np.float32(0.29), **kwargs) == fn(g, float(np.float32(0.29)), **kwargs)


def test_class_thresholds_validates_method_and_alpha_before_looping_on_empty_input():
    # With no class codes the per-class loop never runs, so a bad method/alpha
    # must not be able to hide behind an empty input and return {} silently.
    with pytest.raises(S.SceneCalibrationError):
        S.class_thresholds(np.array([]), np.array([]), np.array([]), 0.1, "nope")
    with pytest.raises(S.SceneCalibrationError):
        S.class_thresholds(np.array([]), np.array([]), np.array([]), 7.0, "hcp")


def test_class_thresholds_length_mismatch_raises():
    with pytest.raises(S.SceneCalibrationError):
        S.class_thresholds(np.array([1.0, 2.0]), np.array([0, 0, 1]), np.array([0, 0]), 0.1, "hcp")


def test_evaluate_length_mismatch_raises():
    with pytest.raises(S.SceneCalibrationError):
        S.evaluate(np.array([1.0, 2.0]), np.array([0, 0, 1]), np.array([0, 0]), {0: 1.0})


def test_evaluate_non_finite_score_raises():
    with pytest.raises(S.SceneCalibrationError):
        S.evaluate(np.array([1.0, math.nan]), np.array([0, 0]), np.array([0, 0]), {0: 1.0})


def test_non_integer_class_codes_rejected_by_class_thresholds_and_evaluate():
    # int(1.2) == int(1.7) == 1 would otherwise silently collide two distinct
    # classes onto the same output/support key.
    with pytest.raises(S.SceneCalibrationError):
        S.class_thresholds(np.array([1.0, 2.0]), np.array([0, 1]), np.array([1.2, 1.7]), 0.4, "pooled")
    with pytest.raises(S.SceneCalibrationError):
        S.evaluate(np.array([1.0, 2.0]), np.array([0, 1]), np.array([1.2, 1.7]), {1: 1.0})


def test_grouped_rejects_non_1d_input():
    with pytest.raises(S.SceneCalibrationError):
        S.grouped(np.array([[1.0, 2.0]]), np.array([[0, 0]]))


# --- Task 16: scene-adaptive CRC for object-weighted coverage (CRC-D) ---


def test_crc_adaptive_matches_crc_formula_without_size_check():
    g = [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])]
    # same budget arithmetic as CRC with M replaced by d_test
    assert S.crc_object_adaptive_threshold(g, 0.8, 3) == 2.0
    assert S.crc_object_adaptive_threshold(g, 0.8, 10) == 3.0
    # a calibration scene larger than d_test is allowed (no size check): budget = 4 - 0.2 = 3.8 -> q = 2nd smallest
    assert S.crc_object_adaptive_threshold(g, 0.8, 1) == 2.0
    # d=0: budget = 0.8*5 - 0.2*0 = 4.0 -> 4 objects may exceed q -> q = smallest.
    # An "M = max(d_test, largest calibration scene)" mutant would instead use
    # M=3 here (budget 3.4 -> q = 2.0), so this pins the correct value against it.
    assert S.crc_object_adaptive_threshold(g, 0.8, 0) == 1.0
    # vectorized form agrees with the scalar form for every d
    d = np.array([0, 1, 3, 10, 100])
    vec = S.crc_object_adaptive_thresholds(g, 0.8, d)
    assert vec.tolist() == [S.crc_object_adaptive_threshold(g, 0.8, int(x)) for x in d]
    assert math.isinf(S.crc_object_adaptive_threshold(g, 0.1, 100))
    with pytest.raises(S.SceneCalibrationError):
        S.crc_object_adaptive_threshold(g, 0.1, -1)


def test_evaluate_q_matches_evaluate():
    s = np.array([1, 3, 2, 5, 1, 4, 9, 6], dtype=float)
    sc = np.array([7, 7, 2, 2, 2, 5, 5, 7])
    cl = np.array([0, 0, 0, 1, 1, 1, 3, 0])
    q = {0: 2.0, 1: math.inf}
    qobj = np.array([q.get(int(c), math.nan) for c in cl])
    assert S.evaluate_q(s, sc, cl, qobj) == S.evaluate(s, sc, cl, q)


def test_crc_adaptive_object_weighted_validity_monte_carlo():
    # crowded scenes are harder; D = m + Poisson(0.3 m) false positives
    rng = np.random.default_rng(11)
    alpha, reps, K = 0.1, 300, 120
    covs = []
    for _ in range(reps):
        def draw(k):
            out = []
            for _ in range(k):
                m = int(min(1 + np.floor(rng.pareto(1.3) * 10), 400))
                out.append(np.exp(rng.normal(0.3 * math.log(m), 0.6) + 0.5 * rng.normal(size=m)))
            return out
        cal = draw(K)
        test = draw(200)
        unc = tot = 0
        for g in test:
            d = g.size + int(rng.poisson(0.3 * g.size))
            q = S.crc_object_adaptive_threshold(cal, alpha, d)
            unc += int((g > q).sum())
            tot += g.size
        covs.append(1 - unc / tot)
    covs = np.array(covs)
    assert covs.mean() >= 1 - alpha - 3 * covs.std(ddof=1) / np.sqrt(reps)


# --- Task 16 fix round 1: review findings ---


def test_crc_adaptive_threshold_rejects_non_finite_d():
    g = [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])]
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(S.SceneCalibrationError):
            S.crc_object_adaptive_threshold(g, 0.8, bad)


def test_crc_adaptive_thresholds_rejects_non_finite_d():
    g = [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])]
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(S.SceneCalibrationError):
            S.crc_object_adaptive_thresholds(g, 0.8, np.array([1.0, bad]))


def test_crc_adaptive_thresholds_empty_calibration_returns_inf_for_every_d():
    d = np.array([0, 1, 5, 100])
    out = S.crc_object_adaptive_thresholds([], 0.5, d)
    assert out.shape == d.shape
    assert np.all(np.isinf(out))
    # calibration scenes that are all present-but-empty are equivalent to no
    # calibration data at all
    out2 = S.crc_object_adaptive_thresholds([np.array([]), np.array([])], 0.5, d)
    assert np.all(np.isinf(out2))


def test_evaluate_q_rejects_q_obj_shape_mismatch():
    with pytest.raises(S.SceneCalibrationError):
        S.evaluate_q(
            np.array([1.0, 2.0, 3.0]), np.array([0, 0, 1]), np.array([0, 0, 1]), np.array([1.0, 2.0])
        )


def test_evaluate_rejects_nan_class_codes_before_building_q_obj():
    # A NaN class code must raise SceneCalibrationError (via _check_integer_codes,
    # run before q_obj is built), not crash inside the q_obj list comprehension
    # with a bare ValueError from int(nan).
    with pytest.raises(S.SceneCalibrationError):
        S.evaluate(np.array([1.0, 2.0]), np.array([0, 0]), np.array([0.0, math.nan]), {0: 1.0})


def test_evaluate_rejects_2d_class_codes_before_building_q_obj():
    # A 2-D class_codes array must raise SceneCalibrationError (via _check_aligned,
    # run before q_obj is built), not crash inside the q_obj list comprehension
    # with a bare TypeError from int(row) on a multi-element row.
    scores = np.array([1.0, 2.0])
    scene = np.array([0, 0])
    cls = np.array([[0, 1], [0, 1]])
    with pytest.raises(S.SceneCalibrationError):
        S.evaluate(scores, scene, cls, {0: 1.0})


def test_evaluate_rejects_nan_valued_q_by_class():
    # A NaN threshold supplied in q_by_class is an invalid input, not "out of
    # support" -- out-of-support is reserved for classes absent from the dict.
    with pytest.raises(S.SceneCalibrationError):
        S.evaluate(np.array([1.0, 2.0]), np.array([0, 0]), np.array([0, 1]), {0: 1.0, 1: math.nan})


def test_evaluate_q_still_treats_nan_threshold_as_out_of_support():
    # evaluate_q's own per-object API is unchanged: NaN in q_obj means out of
    # support, even though evaluate() itself now forbids NaN q_by_class values.
    s = np.array([1.0, 2.0, 3.0])
    sc = np.array([0, 0, 1])
    cl = np.array([0, 0, 1])
    ev = S.evaluate_q(s, sc, cl, np.array([1.0, 1.0, math.nan]))
    assert ev["n_out_of_support"] == 1
    assert set(ev["per_class"]) == {0}

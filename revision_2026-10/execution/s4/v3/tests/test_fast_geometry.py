"""Bounded geometry-only regression; never imports/fits the model pipeline."""
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from decimal import Decimal, localcontext, Inexact, Rounded
from fractions import Fraction

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fast_geometry as fast

OLD_ROOT = ROOT.parent / "v2"
RECORDS_ROOT = Path(os.environ.get("ROTCERT_S4_RECORDS_ROOT", ROOT.parent / "records"))
SAVED_RUN = RECORDS_ROOT / "cpu-runtime-binding-v1/code/execution/s4/v2/runs/real-input-pilot-v2-budgeted"
ORACLE_SHA256 = "ad5b9207dfd4329d3aa014ce74901308adefbb2a5332e597869023942b201c3f"
CONDITIONS = ("identity", "brightness_070", "contrast_070", "gamma_150",
              "saturation_035", "blue_cast_115")

# Independently import the immutable OLD oracle, including its own module/cache.
spec = importlib.util.spec_from_file_location(
    "_fast_geometry_frozen_contracts", OLD_ROOT / "vendor/rotcert_contracts/__init__.py",
    submodule_search_locations=[str(OLD_ROOT / "vendor/rotcert_contracts")])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
spec = importlib.util.spec_from_file_location(
    "_fast_geometry_frozen_contracts.geometry", OLD_ROOT / "vendor/rotcert_contracts/geometry.py")
ORACLE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ORACLE
spec.loader.exec_module(ORACLE)


def original_arrays(w, h, q):
    w, h, q = np.broadcast_arrays(np.asarray(w, float), np.asarray(h, float), np.asarray(q, float))
    rows = [ORACLE.angle_projection(float(a), float(b), float(c))
            for a, b, c in zip(w.flat, h.flat, q.flat)]
    return tuple(np.asarray([getattr(row, attr) for row in rows], dtype=dtype).reshape(w.shape)
                 for attr, dtype in (("half_width_lower", float),
                                     ("half_width_upper", float), ("full", bool)))


def adversarial_samples():
    tiny = np.finfo(float).smallest_subnormal
    maximum = np.finfo(float).max
    pairs = [(10., 2.), (1., 1.), (1., np.nextafter(1., 0.)),
             (np.nextafter(1., 2.), 1.), (123.456, 123.455),
             (tiny, tiny), (tiny * 2, tiny), (tiny * 16, tiny),
             (np.finfo(float).tiny, tiny), (maximum, maximum / 2),
             (maximum, np.nextafter(maximum, 0.)), (maximum, tiny),
             (1e-300, 2e-300), (1e300, 2e300), (1., tiny)]
    rows = []
    for w, h in pairs:
        critical = ORACLE.critical_radius(w, h)
        qs = [0., -0., tiny, critical, math.inf]
        if critical > 0:
            qs += [np.nextafter(critical, 0.), np.nextafter(critical, math.inf),
                   critical * .125, critical * .75]
        for q in qs:
            rows += [(w, h, q), (h, w, q)]
    # Logarithmically distributed sides and near-boundary q, including scales
    # where binary64 squaring over/underflows but extended precision does not.
    rng = np.random.default_rng(20260929)
    for i in range(256):
        w = float(np.ldexp(rng.uniform(1., 2.), int(rng.integers(-1073, 1023))))
        if i % 4 == 0:
            h = float(np.nextafter(w, 0.))
        else:
            h = float(np.ldexp(rng.uniform(1., 2.), int(rng.integers(-1073, 1023))))
        if h == 0:
            h = tiny
        critical = ORACLE.critical_radius(w, h)
        q = float(critical * rng.choice([.001, .01, .1, .5, .9]))
        rows.append((w, h, q))
    # q around the atan-complement boundary numerator/denominator == 1.
    w, h = 10., 2.
    with localcontext() as ctx:
        ctx.prec = 100
        a, b, c = Decimal(26), Decimal(8), Decimal(18)
        q2 = ((a + b + c) - ((a + b + c)**2 - 8*b*c).sqrt()) / 4
        center = float(q2.sqrt())
    for q in [np.nextafter(center, 0.), center, np.nextafter(center, math.inf)]:
        rows.append((w, h, q))
    # Pell approximants put q^2 on BOTH sides of the exact critical branch
    # within binary80's interval uncertainty; this is a real Fraction fallback.
    p, r = 1, 1
    for _ in range(40):
        p, r = p + 2*r, p + r
        if 2*p + 1 >= 2**53:
            break
        if p >= 10**9:
            rows.append((float(2*p+1), 1., float(r)))
    return np.asarray(rows, dtype=float).T


def persisted_samples(per_group=16, *, conditions=CONDITIONS,
                      taus=(.5, .6, .7), checkpoints=("base", "draw-0000", "draw-0001")):
    """Use saved per-class q and matching predicted sides, with source indices."""
    blocks, records = [], []
    for model in ("dior-orcnn-s0", "dior-s2anet-s0"):
        for tau in taus:
            saved = []
            for checkpoint in checkpoints:
                path = SAVED_RUN / model / f"iou{tau:.1f}" / f"{checkpoint}.npz"
                with np.load(path, allow_pickle=False) as data:
                    saved.append((checkpoint, data["q"].copy(), path))
            for condition in conditions:
                ci = CONDITIONS.index(condition)
                path = SAVED_RUN / "prepared" / f"{model}__{condition}__iou{tau:.1f}.npz"
                with np.load(path, allow_pickle=False) as archive:
                    data = {key: archive[key] for key in ("tp_pred", "tp_class", "det_box", "det_class")}
                    for checkpoint, thresholds, threshold_path in saved:
                        for box_key, class_key in (("tp_pred", "tp_class"), ("det_box", "det_class")):
                            boxes, classes = data[box_key], data[class_key]
                            # Half uniform, half strictly informative to exercise
                            # the costly branch without fitting or filtering data.
                            for method in (0, 1):
                                q = thresholds[ci, method, classes]
                                candidates = np.flatnonzero(q < np.abs(boxes[:, 2] - boxes[:, 3]) / 3.)
                                uniform = np.linspace(0, len(boxes)-1, per_group//2, dtype=int)
                                informative = (candidates[np.linspace(0, len(candidates)-1,
                                               per_group - len(uniform), dtype=int)]
                                               if len(candidates) else uniform)
                                ix = np.unique(np.r_[uniform, informative])
                                block = np.column_stack((boxes[ix, 2:4], q[ix]))
                                blocks.append(block)
                                records.append(dict(model=model, iou=tau, condition=condition,
                                                    checkpoint=checkpoint, method=method, box_key=box_key,
                                                    row_indices=ix.tolist(), prepared=str(path),
                                                    thresholds=str(threshold_path), count=len(ix)))
    return np.concatenate(blocks).T, records


class FastGeometryTests(unittest.TestCase):
    def assert_bitwise(self, got, want):
        for i in range(3):
            self.assertEqual(got[i].shape, want[i].shape)
            self.assertEqual(got[i].dtype, want[i].dtype)
            np.testing.assert_array_equal(got[i].view(np.uint64) if i < 2 else got[i],
                                          want[i].view(np.uint64) if i < 2 else want[i])

    def test_original_oracle_identity(self):
        for root in (ROOT, OLD_ROOT):
            digest = hashlib.sha256((root / "vendor/rotcert_contracts/geometry.py").read_bytes()).hexdigest()
            self.assertEqual(digest, ORACLE_SHA256)

    def test_broadcast_scalar_empty_strided_and_input_immutability(self):
        w = np.arange(2., 26., 2.).reshape(3, 4)[:, ::2]
        h = np.array([[1.], [2.], [3.]])
        q = np.array([0., .01])
        copies = [a.copy() for a in (w, h, q)]
        self.assert_bitwise(fast.project_arrays(w, h, q), original_arrays(w, h, q))
        for before, after in zip(copies, (w, h, q)):
            np.testing.assert_array_equal(before, after)
        self.assert_bitwise(fast.project_arrays(10., 2., .1), original_arrays(10., 2., .1))
        self.assert_bitwise(fast.project_arrays(np.empty((0, 1)), 2., np.empty((1, 3))),
                            original_arrays(np.empty((0, 1)), 2., np.empty((1, 3))))
        with self.assertRaises(ValueError):
            fast.project_arrays(np.ones(2), np.ones(3), 0.)

    def test_invalid_sides_and_q(self):
        for bad in (0., -0., -1., -math.inf, math.inf, math.nan):
            for args in ((bad, 1., math.inf), (1., bad, 0.)):
                with self.subTest(args=args), self.assertRaises(ValueError):
                    fast.project_arrays(*args)
        for bad in (-1., -np.finfo(float).smallest_subnormal, -math.inf, math.nan):
            with self.subTest(q=bad), self.assertRaises(ValueError):
                fast.project_arrays(1., 1., bad)
        for bad in (True, np.array([False]), 1+2j):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                fast.project_arrays(bad, 1., 0.)

    def test_full_square_infinity_and_signed_zero_exact(self):
        w = np.array([1., 2., 1., 2., 2.])
        h = np.array([1., 1., 1., 1., 1.])
        q = np.array([0., math.inf, -0., 0., -0.])
        with patch.object(fast._oracle, "angle_projection", wraps=fast._oracle.angle_projection) as scalar:
            got = fast.project_arrays(w, h, q)
        self.assert_bitwise(got, original_arrays(w, h, q))
        self.assertFalse(np.signbit(got[0]).any())
        if fast._LONGDOUBLE_OK:
            self.assertEqual(scalar.call_count, 0)

    def test_adversarial_extremes_critical_near_square_and_zero(self):
        inputs = adversarial_samples()
        with np.errstate(all="raise"):
            got = fast.project_arrays(*inputs)
        self.assert_bitwise(got, original_arrays(*inputs))

    def test_unsupported_platform_uses_scalar_for_every_item(self):
        inputs = np.array([(10., 2., .03), (1., 1., 0.), (2., 1., math.inf), (2., 1., 0.)]).T
        with patch.object(fast, "_LONGDOUBLE_OK", False), patch.object(
                fast._oracle, "angle_projection", wraps=fast._oracle.angle_projection) as scalar:
            got = fast.project_arrays(*inputs)
        self.assertEqual(scalar.call_count, inputs.shape[1])
        self.assert_bitwise(got, original_arrays(*inputs))
        with patch.object(fast, "_GET_ROUNDING", return_value=1024):
            self.assertFalse(fast._platform_supported())
        with patch.object(fast.platform, "machine", return_value="aarch64"):
            self.assertFalse(fast._platform_supported())
        with localcontext() as ctx:
            ctx.Emin = -999
            self.assertFalse(fast._platform_supported())

    def test_ambiguous_angle_must_fall_back(self):
        w = np.array([10., 20., 40.]); h = w / 5; q = w / 100
        nan = np.full(3, np.nan, dtype=np.longdouble)
        with patch.object(fast, "_informative_bounds", return_value=(nan, nan)), \
                patch.object(fast, "_decimal_refine", return_value=None), patch.object(
                fast._oracle, "angle_projection", wraps=fast._oracle.angle_projection) as scalar:
            got = fast.project_arrays(w, h, q)
        self.assertEqual(scalar.call_count, 3)
        self.assert_bitwise(got, original_arrays(w, h, q))

    def test_unique_pair_rejects_float_endpoint_and_crossing(self):
        a = np.longdouble(1.)
        b = np.longdouble(np.nextafter(1., math.inf))
        x = np.array([a, np.nextafter(a, b), a, a], dtype=np.longdouble)
        y = np.array([a, np.nextafter(b, a), b, np.nextafter(b, np.longdouble(math.inf))])
        lo, hi, accepted = fast._unique_float_pair((x, y))
        np.testing.assert_array_equal(accepted, [False, True, False, False])
        self.assertEqual(lo[1], 1.); self.assertEqual(hi[1], float(b))

    def test_tick_bounded_chunks_and_fallbacks(self):
        w = np.full(23, 10.); h = np.full(23, 2.); q = np.full(23, .1)
        with patch.object(fast, "_CHUNK_SIZE", 7), patch.object(fast, "_FALLBACK_TICK", 3), \
                patch.object(fast, "_LONGDOUBLE_OK", False), patch.object(fast, "tick") as tick:
            got = fast.project_arrays(w, h, q)
        self.assertGreaterEqual(tick.call_count, 1 + 4 + 10)
        self.assertTrue(all(call.args == () and call.kwargs == {} for call in tick.call_args_list))
        self.assert_bitwise(got, original_arrays(w, h, q))
        with patch.object(fast, "tick", side_effect=RuntimeError("budget stop")):
            with self.assertRaisesRegex(RuntimeError, "budget stop"):
                fast.project_arrays(w, h, q)

    def test_interval_operations_enclose_decimal_endpoints(self):
        if not fast._LONGDOUBLE_OK:
            # Capability failure is tested via the full scalar path above.
            self.assertFalse(fast._platform_supported())
            return
        for exp in (-1000, -40, 0, 40, 1000):
            a = np.ldexp(np.longdouble(1.125), exp)
            b = np.ldexp(np.longdouble(.625), exp)
            da = ORACLE._Interval.exact(Fraction(*a.as_integer_ratio()))
            db = ORACLE._Interval.exact(Fraction(*b.as_integer_ratio()))
            for got, ref in ((fast._add((a, a), (b, b)), da + db),
                             (fast._sub((a, a), (b, b)), da - db),
                             (fast._mul((a, a), (b, b)), da * db),
                             (fast._div((a, a), (b, b)), da / db),
                             (fast._sqrt((a, a)), da.sqrt())):
                self.assertLessEqual(Fraction(*got[0].as_integer_ratio()), Fraction(ref.lo))
                self.assertGreaterEqual(Fraction(*got[1].as_integer_ratio()), Fraction(ref.hi))

    def test_short_series_encloses_original_40_term_interval(self):
        _, reciprocal = fast._constants()
        for value in (np.longdouble(0), np.longdouble(".0492"), np.longdouble(".125"),
                      np.ldexp(np.longdouble(1), -2100), np.longdouble(".001")):
            ref = ORACLE._Interval.exact(Fraction(*value.as_integer_ratio()))
            seed = (fast._decimal_endpoint(ref.lo, False), fast._decimal_endpoint(ref.hi, True))
            with np.errstate(all="ignore"):
                low, high = fast._atan_small(seed, reciprocal)
            ref = ORACLE._atan_small(ref, terms=40)
            self.assertLessEqual(Fraction(*low.as_integer_ratio()), Fraction(ref.lo))
            self.assertGreaterEqual(Fraction(*high.as_integer_ratio()), Fraction(ref.hi))
        low, high = fast._constants()[0]
        ref = ORACLE._pi() / ORACLE._Interval.exact(2)
        self.assertLessEqual(Fraction(*low.as_integer_ratio()), Fraction(ref.lo))
        self.assertGreaterEqual(Fraction(*high.as_integer_ratio()), Fraction(ref.hi))

    def test_full_arrays_adversarial_broadcast_and_empty(self):
        inputs = adversarial_samples()
        with patch.object(fast._oracle, "angle_projection", side_effect=AssertionError("angle work forbidden")):
            got = fast.full_arrays(*inputs)
            scalar = fast.full_arrays(1., 2., .03)
            empty = fast.full_arrays(np.empty((0, 1)), 2., np.empty((1, 3)))
        np.testing.assert_array_equal(got, original_arrays(*inputs)[2])
        self.assertEqual(scalar.shape, ())
        self.assertEqual(scalar.dtype, np.dtype(bool))
        self.assertEqual(empty.shape, (0, 3))
        w = np.array([[1.], [2.]])
        q = np.array([0., .1, math.inf])
        np.testing.assert_array_equal(fast.full_arrays(w, 1., q), original_arrays(w, 1., q)[2])

    def test_full_arrays_fraction_fallback_and_validation(self):
        w = np.array([10., 1., 2., 2., 2.]); h = np.ones(5)
        q = np.array([.1, 0., math.inf, 0., .1])
        with patch.object(fast, "_LONGDOUBLE_OK", False), patch.object(
                fast, "_fraction_full", wraps=fast._fraction_full) as fraction, patch.object(
                fast._oracle, "angle_projection", side_effect=AssertionError("angle work forbidden")):
            got = fast.full_arrays(w, h, q)
        self.assertEqual(fraction.call_count, 2)
        np.testing.assert_array_equal(got, original_arrays(w, h, q)[2])
        if fast._LONGDOUBLE_OK:
            # Deliberately inconclusive interval: equality remains inclusive
            # only in the exact Fraction predicate, not through an epsilon.
            with patch.object(fast, "_mul", return_value=(np.longdouble(0), np.longdouble(1000))), \
                    patch.object(fast, "_fraction_full", wraps=fast._fraction_full) as fraction:
                got = fast.full_arrays(10., 2., .1)
            self.assertEqual(fraction.call_count, 1)
            self.assertFalse(got)
        for args in ((0., 1., math.inf), (-1., 2., 0.), (math.inf, 1., 0.),
                     (1., math.nan, 0.), (1., 1., -1.), (1., 1., math.nan)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                fast.full_arrays(*args)

    def test_nondefault_decimal_traps_retain_original_exceptions(self):
        fast.project_arrays(10., 2., .1)  # Warm all default-context constants.
        for signal in (Inexact, Rounded):
            for q in (.1, math.inf):
                with self.subTest(signal=signal, q=q), localcontext() as ctx:
                    ctx.traps[signal] = True
                    self.assertFalse(fast._platform_supported())
                    for fn in (ORACLE.angle_projection, fast.project_arrays, fast.full_arrays):
                        with self.assertRaises(signal):
                            fn(10., 2., q)

    def test_decimal_refinement_is_bitwise_and_retains_hard_fallback(self):
        samples = adversarial_samples()
        expected = original_arrays(*samples)
        accepted = 0
        for i, (w, h, q) in enumerate(samples.T):
            if not np.isfinite(q):
                continue
            result = fast._decimal_refine(w, h, q)
            if result is not None:
                self.assert_bitwise(tuple(np.asarray(x) for x in result),
                                    tuple(x[i] for x in expected))
                accepted += 1
        self.assertGreater(accepted, 400)
        # Force first-tier ambiguity; the second tier must accept ordinary
        # inputs without evaluating the original 40-term interval oracle.
        nan = np.full(3, np.nan, dtype=np.longdouble)
        w = np.array([10., 20., 40.]); h = w/5; q = w/100
        with patch.object(fast, "_informative_bounds", return_value=(nan, nan)), patch.object(
                fast._oracle, "angle_projection", side_effect=AssertionError("unnecessary oracle call")):
            got = fast.project_arrays(w, h, q)
        self.assert_bitwise(got, original_arrays(w, h, q))

    def test_pell_critical_ambiguity_is_resolved_exactly(self):
        samples = np.array([(3710155683., 1., 1311738121.),
                            (8957108167., 1., 3166815962.)]).T
        expected = original_arrays(*samples)
        np.testing.assert_array_equal(expected[2], [True, False])
        with patch.object(fast, "_fraction_full", wraps=fast._fraction_full) as fraction:
            flags = fast.full_arrays(*samples)
        self.assertEqual(fraction.call_count, 2)
        np.testing.assert_array_equal(flags, expected[2])
        with patch.object(fast._oracle, "angle_projection", wraps=fast._oracle.angle_projection) as scalar:
            result = fast.project_arrays(*samples)
        self.assertLessEqual(scalar.call_count, 2)
        self.assert_bitwise(result, expected)

    def test_short_series_interval_input_encloses_original(self):
        _, reciprocal = fast._constants()
        for value in (np.longdouble(".0492"), np.longdouble(".001"),
                      np.ldexp(np.longdouble(1), -2100)):
            with localcontext() as ctx:
                ctx.prec = 80
                center = Decimal(Fraction(*value.as_integer_ratio()).numerator) / Decimal(Fraction(*value.as_integer_ratio()).denominator)
                dec = ORACLE._Interval(ctx.next_minus(center), ctx.next_plus(center))
            seed = (fast._decimal_endpoint(dec.lo, False), fast._decimal_endpoint(dec.hi, True))
            low, high = fast._atan_small(seed, reciprocal)
            ref = ORACLE._atan_small(dec, terms=40)
            self.assertLessEqual(Fraction(*low.as_integer_ratio()), Fraction(ref.lo))
            self.assertGreaterEqual(Fraction(*high.as_integer_ratio()), Fraction(ref.hi))

    def test_saved_orcnn_s2anet_prepared_sides_and_persisted_q(self):
        samples, records = persisted_samples(per_group=6, conditions=("identity", "gamma_150"),
                                             taus=(.5,), checkpoints=("base", "draw-0000"))
        self.assertEqual({row["model"] for row in records}, {"dior-orcnn-s0", "dior-s2anet-s0"})
        expected = original_arrays(*samples)
        self.assert_bitwise(fast.project_arrays(*samples), expected)
        with patch.object(fast._oracle, "angle_projection", side_effect=AssertionError("angle work forbidden")):
            np.testing.assert_array_equal(fast.full_arrays(*samples), expected[2])


if __name__ == "__main__":
    unittest.main(verbosity=2)

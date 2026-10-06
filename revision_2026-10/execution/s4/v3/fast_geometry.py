"""Bitwise-preserving array projection for the immutable Decimal geometry oracle.

Only Linux x86-64 binary80 in round-to-nearest and the default Decimal context
is accelerated. Binary80 ambiguity uses a certified 50-digit refinement before
the immutable scalar fallback. Each outward operation encloses Decimal rounding;
see evidence/fast-geometry-proof.md. An unresolved branch or float64 rounding
cell uses the original scalar implementation. No libm atan is used.
"""
from __future__ import annotations

import ctypes
from decimal import (Decimal, getcontext, localcontext, ROUND_HALF_EVEN,
                     ROUND_FLOOR, ROUND_CEILING, InvalidOperation, DivisionByZero, Overflow)
from fractions import Fraction
from functools import lru_cache
import math
import platform
import sys

import numpy as np

from resources import tick
from vendor.rotcert_contracts import geometry as _oracle

_LD = np.longdouble
_ZERO, _ONE = _LD(0), _LD(1)
_INF = _LD(np.inf)
_CHUNK_SIZE = 4096
_FALLBACK_TICK = 32
_DEFAULT_ENABLED_TRAPS = frozenset((InvalidOperation, DivisionByZero, Overflow))


def _load_rounding_probe():
    try:
        probe = ctypes.CDLL(None).fegetround
        probe.argtypes = []
        probe.restype = ctypes.c_int
        return probe
    except (AttributeError, OSError):
        return None


_GET_ROUNDING = _load_rounding_probe()


def _decimal_context_supported():
    ctx = getcontext()
    return (ctx.prec == 28 and ctx.rounding == ROUND_HALF_EVEN and ctx.Emin == -999999
            and ctx.Emax == 999999 and ctx.clamp == 0
            and frozenset(signal for signal, enabled in ctx.traps.items() if enabled)
            == _DEFAULT_ENABLED_TRAPS)


def _platform_supported():
    """Fail closed outside the ABI/arithmetic assumptions of the proof."""
    info = np.finfo(_LD)
    if (sys.platform != "linux" or platform.machine().lower() not in ("x86_64", "amd64")
            or info.nmant != 63 or info.minexp != -16382 or info.maxexp != 16384
            or np.finfo(np.float64).nmant != 52 or _oracle.PRECISION != 80
            or _GET_ROUNDING is None or _GET_ROUNDING() != 0):
        return False
    if not _decimal_context_supported():
        return False
    # Detect reduced x87 precision and loss of binary64 subnormals on conversion.
    if _ONE + np.ldexp(_ONE, -63) != np.nextafter(_ONE, _INF):
        return False
    probes = (np.finfo(np.float64).smallest_subnormal, np.finfo(np.float64).tiny,
              np.nextafter(1.0, 2.0), np.finfo(np.float64).max)
    if any(_LD(x).as_integer_ratio() != float(x).as_integer_ratio() for x in probes):
        return False
    root = np.sqrt(_LD(2))
    a = Fraction(*np.nextafter(root, -_INF).as_integer_ratio())
    b = Fraction(*np.nextafter(root, _INF).as_integer_ratio())
    return a * a < 2 < b * b


_LONGDOUBLE_OK = _platform_supported()


def _down(x):
    return np.nextafter(x, -_INF)


def _up(x):
    return np.nextafter(x, _INF)


def _add(a, b):
    return _down(a[0] + b[0]), _up(a[1] + b[1])


def _sub(a, b):
    return _down(a[0] - b[1]), _up(a[1] - b[0])


def _mul(a, b):
    """Nonnegative intervals only; zero lower endpoints stay nonnegative."""
    return np.maximum(_ZERO, _down(a[0] * b[0])), _up(a[1] * b[1])


def _div(a, b):
    # Mirror Decimal reciprocal THEN multiplication, including both roundings.
    reciprocal = (_down(_ONE / b[1]), _up(_ONE / b[0]))
    return _mul(a, reciprocal)


def _sqrt(a):
    return np.maximum(_ZERO, _down(np.sqrt(a[0]))), _up(np.sqrt(a[1]))


def _constant(x):
    x = _LD(x)
    return x, x


def _decimal_endpoint(value, upward):
    """Verify conversion by exact ratios; do not trust decimal-string rounding."""
    exact = Fraction(value)
    result = _LD(str(value))
    direction = _INF if upward else -_INF
    while ((Fraction(*result.as_integer_ratio()) < exact) if upward
           else (Fraction(*result.as_integer_ratio()) > exact)):
        result = np.nextafter(result, direction)
    return result


@lru_cache(maxsize=1)
def _constants():
    # The immutable oracle proves its pi enclosure using Machin's identity.
    pi_half = _oracle._pi() / _oracle._Interval.exact(2)
    pi_half = (_decimal_endpoint(pi_half.lo, False),
               _decimal_endpoint(pi_half.hi, True))
    # Integers are exact in both formats. Widened reciprocals contain the
    # oracle's 80-digit reciprocals used in every Taylor-term division.
    reciprocals = tuple((_down(_ONE / _LD(n)), _up(_ONE / _LD(n)))
                        for n in range(1, 34, 2))
    return pi_half, reciprocals


def _atan_small(x, reciprocals):
    """16 terms, enclosing BOTH the exact atan and the oracle's 40-term interval.

    The absolute geometric tail contains terms 16..39 and the oracle remainder.
    A separately proved 2**-200*x.hi covers the omitted Decimal additions; it
    never changes a branch. This avoids 24 gratuitous binary80 widenings.
    """
    power, squared, total = x, _mul(x, x), _constant(0)
    for k in range(16):
        term = _mul(power, reciprocals[k])
        total = _add(total, term) if k % 2 == 0 else _sub(total, term)
        power = _mul(power, squared)
    # One more binary80 ulp dominates every Decimal multiplication's relative
    # rounding. Bound all subsequent term magnitudes by a geometric series.
    ratio = _up(squared[1])
    tail = _div(_mul(power, reciprocals[16]),
                _sub(_constant(1), (_ZERO, ratio)))[1]
    decimal_roundoff = _up(np.ldexp(x[1], -200))
    bound = _up(tail + decimal_roundoff)
    return _down(total[0] - bound), _up(total[1] + bound)


def _float_down(x):
    result = np.asarray(x, dtype=np.float64)
    return np.where(result.astype(_LD) > x, np.nextafter(result, -np.inf), result)


def _float_up(x):
    result = np.asarray(x, dtype=np.float64)
    return np.where(result.astype(_LD) < x, np.nextafter(result, np.inf), result)


def _unique_float_pair(interval):
    lo, hi = interval
    lower, upper = _float_down(lo), _float_up(hi)
    accepted = (np.isfinite(lo) & np.isfinite(hi) & (lo >= 0) & (lo <= hi)
                & (lower == _float_down(hi)) & (upper == _float_up(lo))
                & (np.nextafter(lower, np.inf) == upper))
    return lower, upper, accepted


def _informative_bounds(w, h, q2, r2):
    """Enclose the oracle Decimal interval, returning NaN for unresolved work."""
    lo = np.full(w.shape, np.nan, dtype=_LD)
    hi = lo.copy()
    numerator = _mul(q2, _sub(_mul(_add(_mul((w, w), (w, w)),
                                           _mul((h, h), (h, h))), _constant(.25)), q2))
    side_sum = _add((w, w), (h, h))
    denominator = _mul(_sub(r2, q2),
                       _sub(_mul(_mul(side_sum, side_sum), _constant(.125)), q2))
    complement = numerator[0] > denominator[1]
    ordinary = numerator[1] <= denominator[0]
    valid = ((complement | ordinary) & (numerator[0] > 0) & (denominator[0] > 0)
             & np.isfinite(numerator[1]) & np.isfinite(denominator[1]))
    ix = np.flatnonzero(valid)
    if not ix.size:
        return lo, hi
    comp = complement[ix]
    n = tuple(np.where(comp, denominator[j][ix], numerator[j][ix]) for j in (0, 1))
    d = tuple(np.where(comp, numerator[j][ix], denominator[j][ix]) for j in (0, 1))
    t = _div(n, d)
    # The rational formula above is exact in the oracle. This extra outward
    # step includes _Interval.exact(rational_t)'s Decimal conversion as well.
    seed = (np.maximum(_ZERO, _down(t[0])), _up(t[1]))
    x = _sqrt(seed)
    for _ in range(4):
        x = _div(x, _add(_constant(1), _sqrt(_add(_constant(1), _mul(x, x)))))
    # 1/8 is an exactly represented strict subset of the oracle's [0, .21].
    valid = (x[0] >= 0) & (x[1] <= _LD(.125)) & np.isfinite(x[1])
    ix, comp = ix[valid], comp[valid]
    if not ix.size:
        return lo, hi
    x = x[0][valid], x[1][valid]
    pi_half, reciprocals = _constants()
    angle = _mul(_constant(16), _atan_small(x, reciprocals))
    complemented = _sub(pi_half, angle)
    angle = tuple(np.where(comp, complemented[j], angle[j]) for j in (0, 1))
    # A negative enclosure cannot be passed to nonnegative multiplication.
    valid = angle[0] >= 0
    ix = ix[valid]
    result = _mul((angle[0][valid], angle[1][valid]), _constant(.5))
    lo[ix], hi[ix] = result
    return lo, hi


@lru_cache(maxsize=1)
def _refinement_constants():
    with localcontext() as ctx:
        ctx.prec = 160
        ctx.rounding = ROUND_HALF_EVEN
        relative_guard = Decimal(2) ** -120  # Exact terminating decimal.
        pi = _oracle._pi()
        ctx.prec = 50
        pi_quarter = (pi.lo + pi.hi) / 8
    return relative_guard, pi_quarter, tuple(Decimal(2*k+1) for k in range(16))


def _decimal_refine(w, h, q):
    """Second tier for binary80 ambiguity, preserving the original float pair.

    An exact Fraction t is evaluated at 50 Decimal digits with four reductions
    and 16 terms. The proved relative envelope 2**-120 contains truncation,
    all 50-digit rounding AND the original 80-digit interval (proof note).
    This envelope changes no exact branch; inconclusive float pairs return None.
    """
    a, b, radius = Fraction(float(w)), Fraction(float(h)), Fraction(float(q))
    q2, r2 = radius*radius, (a-b)**2 / 8
    if q2 >= r2:
        return math.pi/2, math.pi/2, True
    if not radius:
        return 0., 0., False
    n = q2 * ((a*a+b*b)/4-q2)
    d = (r2-q2) * ((a+b)**2/8-q2)
    complement = n > d
    t = d/n if complement else n/d
    guard, pi_quarter, divisors = _refinement_constants()
    with localcontext() as ctx:
        ctx.prec = 50
        ctx.rounding = ROUND_HALF_EVEN
        x = (Decimal(t.numerator) / Decimal(t.denominator)).sqrt()
        for _ in range(4):
            x = x / (1 + (1 + x*x).sqrt())
        if not (0 < x <= Decimal(".0625")):
            return None
        power, squared, total = x, x*x, Decimal(0)
        for k, divisor in enumerate(divisors):
            term = power / divisor
            total = total + term if k % 2 == 0 else total - term
            power *= squared
        angle = 8 * total
        if complement:
            angle = pi_quarter - angle
        if angle <= 0:
            return None
        ctx.rounding = ROUND_CEILING
        error = angle * guard
        high = angle + error
        ctx.rounding = ROUND_FLOOR
        low = angle - error
    lower, upper = _oracle._float_down(low), _oracle.float_up(high)
    if (lower == _oracle._float_down(high) and upper == _oracle.float_up(low)
            and math.nextafter(lower, math.inf) == upper):
        return lower, upper, False
    return None


def _scalar_fill(w, h, q, indices, lower, upper, full, *, refine=False):
    for count, i in enumerate(indices):
        if count % _FALLBACK_TICK == 0:
            tick()
        if refine:
            refined = _decimal_refine(w[i], h[i], q[i])
            if refined is not None:
                lower[i], upper[i], full[i] = refined
                continue
        result = _oracle.angle_projection(float(w[i]), float(h[i]), float(q[i]))
        lower[i], upper[i], full[i] = (result.half_width_lower,
                                     result.half_width_upper, result.full)


def _project_chunk(w, h, q, accelerated):
    lower, upper = np.empty(w.size), np.empty(w.size)
    full = np.zeros(w.size, dtype=bool)
    if not accelerated:
        _scalar_fill(w, h, q, range(w.size), lower, upper, full)
        return lower, upper, full

    done = (w == h) | np.isposinf(q)
    full[done] = True
    lower[done] = upper[done] = math.pi / 2
    zero = (q == 0) & ~done
    lower[zero] = upper[zero] = 0.
    done |= zero
    ix = np.flatnonzero(~done)
    if ix.size:
        a = np.maximum(w[ix], h[ix]).astype(_LD)
        b = np.minimum(w[ix], h[ix]).astype(_LD)
        radius = q[ix].astype(_LD)
        q2 = _mul((radius, radius), (radius, radius))
        diff = _sub((a, a), (b, b))
        diff = np.maximum(_ZERO, diff[0]), diff[1]
        r2 = _mul(_mul(diff, diff), _constant(.125))
        entire = q2[0] >= r2[1]
        fi = ix[entire]
        full[fi] = done[fi] = True
        lower[fi] = upper[fi] = math.pi / 2
        informative = q2[1] < r2[0]
        ii = ix[informative]
        if ii.size:
            interval = _informative_bounds(a[informative], b[informative],
                                          tuple(v[informative] for v in q2),
                                          tuple(v[informative] for v in r2))
            low, high, accepted = _unique_float_pair(interval)
            ai = ii[accepted]
            lower[ai], upper[ai], done[ai] = low[accepted], high[accepted], True
    _scalar_fill(w, h, q, np.flatnonzero(~done), lower, upper, full, refine=True)
    return lower, upper, full


def _float64_input(value):
    array = np.asarray(value)
    if array.dtype.kind in "bc":
        raise ValueError("boolean and complex geometry inputs are not supported")
    return np.asarray(array, dtype=np.float64)


def _validate_chunk(w, h, q):
    if not (np.isfinite(w).all() and np.isfinite(h).all()
            and (w > 0).all() and (h > 0).all()):
        raise ValueError("box side lengths must be finite and strictly positive")
    if np.isnan(q).any() or (q < 0).any():
        raise ValueError("q must be nonnegative and not NaN")


def _fraction_full(w, h, q):
    """The original exact full predicate, with no transcendental work."""
    w, h, q = Fraction(float(w)), Fraction(float(h)), Fraction(float(q))
    return 8 * q * q >= (w - h) ** 2


def full_arrays(w, h, q):
    """Broadcast binary64 inputs and return only the oracle's exact full flags.

    No angle evaluation is needed in the supported default Decimal context.
    Directed intervals decide ordinary inputs; Fraction resolves an ambiguous
    critical-radius comparison (all unresolved items on unsupported platforms).
    Nondefault Decimal contexts use the original scalar oracle, including its
    exceptions before special branches such as +infinity.
    """
    w, h, q = np.broadcast_arrays(*map(_float64_input, (w, h, q)))
    tick()
    full = np.empty(w.shape, dtype=bool)
    context_ok = _decimal_context_supported()
    accelerated = _LONGDOUBLE_OK and _platform_supported()
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        for start in range(0, w.size, _CHUNK_SIZE):
            tick()
            stop = min(start + _CHUNK_SIZE, w.size)
            a, b, radius = (x.flat[start:stop] for x in (w, h, q))
            _validate_chunk(a, b, radius)
            if not context_ok:
                result = np.empty(a.size, dtype=bool)
                for count in range(a.size):
                    if count % _FALLBACK_TICK == 0:
                        tick()
                    result[count] = _oracle.angle_projection(
                        float(a[count]), float(b[count]), float(radius[count])).full
                full.flat[start:stop] = result
                continue
            result = (a == b) | np.isposinf(radius)
            done = result | (radius == 0)
            ix = np.flatnonzero(~done)
            if accelerated and ix.size:
                al = np.maximum(a[ix], b[ix]).astype(_LD)
                bl = np.minimum(a[ix], b[ix]).astype(_LD)
                ql = radius[ix].astype(_LD)
                diff = _sub((al, al), (bl, bl))
                diff = np.maximum(_ZERO, diff[0]), diff[1]
                r2 = _mul(_mul(diff, diff), _constant(.125))
                q2 = _mul((ql, ql), (ql, ql))
                entire = q2[0] >= r2[1]
                informative = q2[1] < r2[0]
                result[ix[entire]] = True
                done[ix[entire | informative]] = True
            for count, i in enumerate(np.flatnonzero(~done)):
                if count % _FALLBACK_TICK == 0:
                    tick()
                result[i] = _fraction_full(a[i], b[i], radius[i])
            full.flat[start:stop] = result
    return full


def project_arrays(w, h, q):
    """Broadcast binary64 inputs; return float64 lower/upper and bool full arrays.

    Sides must be finite and strictly positive; q must be nonnegative and may
    be +infinity. Square, zero, and full cases retain the original scalar
    semantics (including math.pi/2). Scalars produce zero-dimensional arrays;
    empty broadcasts stay empty. Work and scalar fallbacks tick in bounded
    chunks. Unsupported platforms or Decimal contexts use the immutable oracle
    for every item. Binary80 ambiguities first get a certified Decimal refinement.
    """
    w, h, q = np.broadcast_arrays(*map(_float64_input, (w, h, q)))
    count = w.size
    tick()
    lower, upper = np.empty(w.shape), np.empty(w.shape)
    full = np.empty(w.shape, dtype=bool)
    # Recheck per call: callers can change the floating-point environment.
    accelerated = _LONGDOUBLE_OK and _platform_supported()
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        for start in range(0, count, _CHUNK_SIZE):
            tick()
            stop = min(start + _CHUNK_SIZE, count)
            a, b, radius = (x.flat[start:stop] for x in (w, h, q))
            _validate_chunk(a, b, radius)
            lo, hi, entire = _project_chunk(a, b, radius, accelerated)
            lower.flat[start:stop], upper.flat[start:stop], full.flat[start:stop] = lo, hi, entire
    return lower, upper, full

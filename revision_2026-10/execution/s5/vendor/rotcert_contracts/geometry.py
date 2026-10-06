"""GWD angle projection with set-valued isotropic orientation.

The mathematical projection is exact. Finite numerical angle bounds enclose its
half-extent; full=True means the entire RP1, even at square q=0. Rational branch
comparisons, outward Decimal arithmetic and an alternating atan series avoid
cancellation at near-squares and overflow at extreme scales. This is a scalar
correctness implementation, not a claimed high-throughput vectorized kernel.
"""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction
from functools import lru_cache
import math

import numpy as np

from .exact import rational
from .bounds import float_up

PRECISION = 80


def _op(a, b, op, up):
    with localcontext() as ctx:
        ctx.prec = PRECISION
        ctx.rounding = ROUND_CEILING if up else ROUND_FLOOR
        return getattr(ctx, op)(a, b)


@dataclass(frozen=True)
class _Interval:
    lo: Decimal
    hi: Decimal

    @classmethod
    def exact(cls, x):
        f = rational(x)
        return cls(_op(Decimal(f.numerator), Decimal(f.denominator), "divide", False),
                   _op(Decimal(f.numerator), Decimal(f.denominator), "divide", True))

    def __add__(self, other):
        return _Interval(_op(self.lo, other.lo, "add", False), _op(self.hi, other.hi, "add", True))

    def __sub__(self, other):
        return _Interval(_op(self.lo, other.hi, "subtract", False), _op(self.hi, other.lo, "subtract", True))

    def __mul__(self, other):
        pairs = [(a, b) for a in (self.lo, self.hi) for b in (other.lo, other.hi)]
        return _Interval(min(_op(a, b, "multiply", False) for a, b in pairs),
                         max(_op(a, b, "multiply", True) for a, b in pairs))

    def __truediv__(self, other):
        if other.lo <= 0:
            raise ArithmeticError("interval division requires a positive divisor")
        reciprocal = _Interval(_op(Decimal(1), other.hi, "divide", False),
                               _op(Decimal(1), other.lo, "divide", True))
        return self * reciprocal

    def sqrt(self):
        if self.lo < 0:
            raise ArithmeticError("sqrt of negative interval")
        with localcontext() as ctx:
            ctx.prec = PRECISION
            lo = Decimal(0) if self.lo == 0 else ctx.next_minus(ctx.sqrt(self.lo))
            hi = Decimal(0) if self.hi == 0 else ctx.next_plus(ctx.sqrt(self.hi))
        return _Interval(lo, hi)


_ZERO, _ONE = _Interval.exact(0), _Interval.exact(1)


def _atan_small(x: _Interval, terms=80):
    """Alternating Taylor polynomial plus rigorously bounded next term."""
    if x.lo < 0 or x.hi > Decimal("0.21"):
        raise ArithmeticError("atan series argument must lie in [0,.21]")
    power, squared, total = x, x*x, _ZERO
    for k in range(terms):
        term = power / _Interval.exact(2*k+1)
        total = total + term if k % 2 == 0 else total - term
        power = power * squared
    remainder = power / _Interval.exact(2*terms+1)
    # The next term is positive when terms is even, negative otherwise.
    return total + _Interval(Decimal(0), remainder.hi) if terms % 2 == 0 else total - _Interval(Decimal(0), remainder.hi)


@lru_cache(maxsize=1)
def _pi():
    # Machin identity; no unverified hard-coded transcendental digits.
    return (_Interval.exact(16)*_atan_small(_Interval.exact(Fraction(1, 5)))
            - _Interval.exact(4)*_atan_small(_Interval.exact(Fraction(1, 239))))


def _atan_sqrt(t_squared: Fraction):
    complement = t_squared > 1
    x = _Interval.exact(1/t_squared if complement else t_squared).sqrt()
    # atan(x)=2 atan(x/(1+sqrt(1+x^2))), applied four times. x<=1 initially.
    for _ in range(4):
        x = x / (_ONE + (_ONE+x*x).sqrt())
    angle = _Interval.exact(16) * _atan_small(x, terms=40)
    return _pi() / _Interval.exact(2) - angle if complement else angle


def _sides(w, h):
    w, h = rational(w), rational(h)
    if w <= 0 or h <= 0:
        raise ValueError("box side lengths must be finite and strictly positive")
    return (w, h) if w >= h else (h, w)


def _float_down(d):
    result = float(d)
    return math.nextafter(result, -math.inf) if Decimal.from_float(result) > d else result


def critical_radius(w, h) -> float:
    """Smallest outward float bound for (long-short)/(2 sqrt(2)).

    Passing the returned float to angle_projection is always in the full-angle
    branch. Branch classification itself uses exact q^2 >= (w-h)^2/8.
    """
    w, h = _sides(w, h)
    if w == h:
        return 0.
    return float_up(_Interval.exact((w-h)**2 / 8).sqrt().hi)


@dataclass(frozen=True)
class AngleProjection:
    half_width_lower: float
    half_width_upper: float
    full: bool
    square_prediction: bool
    q_star_upper: float
    convention: str = "Orient(isotropic)=RP1; exact set-valued projection"

    @property
    def interval_length_upper(self):
        return math.pi if self.full else 2*self.half_width_upper

    @property
    def informative(self):
        return not self.full


def angle_projection(w, h, q) -> AngleProjection:
    w, h = _sides(w, h)
    square = w == h
    radius = critical_radius(w, h)
    if isinstance(q, (float, np.floating)) and math.isinf(float(q)):
        if q < 0:
            raise ValueError("q cannot be negative")
        return AngleProjection(math.pi/2, math.pi/2, True, square, radius)
    q = rational(q)
    if q < 0:
        raise ValueError("q cannot be negative")
    r2 = (w-h)**2 / 8
    if q*q >= r2:
        return AngleProjection(math.pi/2, math.pi/2, True, square, radius)
    if q == 0:
        return AngleProjection(0., 0., False, False, radius)
    numerator = q*q * ((w*w+h*h)/4 - q*q)
    denominator = (r2-q*q) * ((w+h)**2/8-q*q)
    angle = _atan_sqrt(numerator/denominator) / _Interval.exact(2)
    return AngleProjection(max(0., _float_down(angle.lo)), float_up(angle.hi), False, False, radius)


def orientation_half_extent(w, h, q):
    return angle_projection(w, h, q).half_width_upper


def orientation_half_extent_array(w, h, q):
    w, h, q = np.broadcast_arrays(w, h, q)
    return np.asarray([orientation_half_extent(float(a), float(b), float(c))
                       for a, b, c in zip(w.flat, h.flat, q.flat)]).reshape(w.shape)


def canonical_box(w, h, theta):
    _sides(w, h)
    theta = float(theta)
    if not math.isfinite(theta):
        raise ValueError("angle must be finite")
    w, h = float(w), float(h)
    if h > w:
        w, h, theta = h, w, theta + math.pi/2
    return w, h, math.remainder(theta, math.pi)


def rp1_distance(theta_a, theta_b):
    if not math.isfinite(theta_a) or not math.isfinite(theta_b):
        raise ValueError("angles must be finite")
    return abs(math.remainder(math.remainder(theta_a, math.pi) - math.remainder(theta_b, math.pi), math.pi))


def min_bures_at_angle(w, h, phi):
    """Stable diagnostic minimum (float, no rigorous transcendental enclosure).

    Used for QA/readouts; certified angle enclosure uses angle_projection.
    """
    w, h = _sides(w, h)
    phi = rp1_distance(float(phi), 0.)
    if w == h or phi == 0:
        return 0.
    if phi >= math.pi/4:
        return critical_radius(w, h)
    t = float(h/w)
    # Rationalize the subtraction c-sqrt(e^2*cos^2(2phi)+ab).
    e = (1-t)*(1+t)/8
    c = (1+t*t)/8
    inner = math.hypot(e*math.cos(2*phi), t/4)
    normalized = e*abs(math.sin(2*phi)) / math.sqrt(c+inner)
    return float(w) * normalized


def extremizer_semiaxes(w, h, phi):
    """Construct extremizing candidate semiaxes; diagnostic float arithmetic."""
    w, h = _sides(w, h)
    phi = rp1_distance(float(phi), 0.)
    if w == h or phi >= math.pi/4:
        side = float((w+h)/4)
        return side, side
    ratio = float(h/w)
    a, b = .25, ratio*ratio/4
    psi = .5 * math.atan2(ratio/2, (a-b)*math.cos(2*phi))
    u = np.array([math.cos(psi), math.sin(psi)])
    c, s = math.cos(phi), math.sin(phi)
    P = np.array([[a*c*c+b*s*s, ratio/4], [ratio/4, a*s*s+b*c*c]])
    r = math.sqrt(float(u @ P @ u))
    return float(w)*r*u[0], float(w)*r*u[1]

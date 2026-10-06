"""Conservative empirical Bernstein and exact binomial tolerance ranks."""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal, localcontext, ROUND_CEILING
from fractions import Fraction
from functools import lru_cache
import math

from .exact import integer, probability, rational

PRECISION = 80


def decimal_up(value: Fraction) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = PRECISION
        ctx.rounding = ROUND_CEILING
        return Decimal(value.numerator) / Decimal(value.denominator)


def float_up(value: Decimal) -> float:
    result = float(value)
    if Decimal.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    return result


@dataclass(frozen=True)
class EBBound:
    upper: float
    mean_exact: Fraction | None
    variance_exact: Fraction | None
    n: int
    eta: Fraction
    supported: bool

    def passes(self, risk_limit) -> bool:
        limit = probability(risk_limit, "risk_limit")
        return self.supported and Fraction.from_float(self.upper) <= limit

    def record(self):
        return {"upper": self.upper, "mean": None if self.mean_exact is None else float(self.mean_exact),
                "variance_ddof1": None if self.variance_exact is None else float(self.variance_exact),
                "n": self.n, "eta_exact": str(self.eta), "supported": self.supported,
                "arithmetic": "exact moments; 80-digit upward bound; float rounded upward"}


def empirical_bernstein(losses, eta) -> EBBound:
    """U=min(1,mean+sqrt(2*v*ln(2/eta)/n)+7ln(2/eta)/(3(n-1))).

    Moments are exact for supplied rational/binary values. Decimal basic
    operations round upward. Decimal ln/sqrt are correctly rounded nearest,
    so next_plus encloses them from above. Validation never accepts tolerance-
    enlarged [0,1], and n<2 always has supported=False, U=1.
    """
    eta = probability(eta, "eta")
    y = tuple(rational(v) for v in losses)
    if any(v < 0 or v > 1 for v in y):
        raise ValueError("losses must lie exactly in [0,1]")
    n = len(y)
    if n < 2:
        return EBBound(1.0, y[0] if n else None, None, n, eta, False)
    total = sum(y, Fraction(0))
    mean = total / n
    variance = (sum((v*v for v in y), Fraction(0)) - total*total/n) / (n-1)
    with localcontext() as ctx:
        ctx.prec = PRECISION
        ctx.rounding = ROUND_CEILING
        logarithm = ctx.next_plus(ctx.ln(decimal_up(2 / eta)))
        radicand = Decimal(2) * decimal_up(variance) * logarithm / Decimal(n)
        sqrt_upper = Decimal(0) if radicand == 0 else ctx.next_plus(ctx.sqrt(radicand))
        upper = (decimal_up(mean) + sqrt_upper
                 + Decimal(7)*logarithm / Decimal(3*(n-1)))
        upper = min(Decimal(1), upper)
    return EBBound(min(1.0, float_up(upper)), mean, variance, n, eta, True)


@dataclass(frozen=True)
class ToleranceRank:
    n: int
    rank: int | None
    tail: Fraction | None
    alpha: Fraction
    delta: Fraction

    def record(self):
        return {"n": self.n, "rank": self.rank, "alpha_exact": str(self.alpha),
                "delta_exact": str(self.delta),
                "tail_upper": None if self.tail is None else float_up(decimal_up(self.tail)),
                "comparison": "exact integer binomial tail <= exact rational delta"}


def tolerance_rank(n, alpha, delta) -> ToleranceRank:
    return _tolerance_rank(integer(n), probability(alpha, "alpha"), probability(delta, "delta"))


@lru_cache(maxsize=128)
def _tolerance_rank(n: int, alpha: Fraction, delta: Fraction) -> ToleranceRank:
    if n == 0:
        return ToleranceRank(n, None, None, alpha, delta)
    # All terms have denominator b**n. Walk down from j=n using exact integer
    # recurrences, stopping at the first inadmissible tail. No scipy tail rounding.
    b = alpha.denominator
    a = b - alpha.numerator
    c = alpha.numerator
    denominator = b**n
    term = a**n
    tail = term
    if tail * delta.denominator > delta.numerator * denominator:
        return ToleranceRank(n, None, None, alpha, delta)
    rank = n
    accepted_tail = tail
    for j in range(n, 1, -1):
        term, remainder = divmod(term*j*c, (n-j+1)*a)
        if remainder:
            raise ArithmeticError("nonintegral binomial recurrence")
        tail += term
        if tail * delta.denominator > delta.numerator * denominator:
            break
        rank, accepted_tail = j-1, tail
    return ToleranceRank(n, rank, Fraction(accepted_tail, denominator), alpha, delta)

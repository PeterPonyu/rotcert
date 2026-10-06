"""Frozen S1 paired source bootstrap primitives; no shared project imports.

Integer multiplicities represent distinct bootstrap copies, but a source and all
its copies always have one role. Empty evaluation classes remain NaN.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
import math
import numpy as np

CAL, DESIGN, EVAL = 1, 2, 3
METHODS = ('mondrian_pooled', 'mondrian_hcp')


def roles_from_codes(counts, seed):
    ids = np.flatnonzero(counts)
    if len(ids) < 3:
        return np.zeros(len(counts), dtype=np.uint8)
    perm = np.random.default_rng(seed).permutation(len(ids))
    m = len(ids)
    nc, nd = max(1, round(.4*m)), max(1, round(.2*m))
    if nc+nd >= m:
        nc, nd = max(1, m-2), 1
    ids = ids[perm]
    role = np.zeros(len(counts), dtype=np.uint8)
    role[ids[:nc]], role[ids[nc:nc+nd]], role[ids[nc+nd:]] = CAL, DESIGN, EVAL
    return role


def original_roles(names, universe, seed):
    order = np.asarray(sorted(universe, key=lambda i: str(names[i])))
    roles = roles_from_codes(np.ones(len(order), dtype=np.int64), seed)
    out = np.zeros(len(names), dtype=np.uint8)
    out[order] = roles
    return out


def pooled_sorted(scores, integer_weights):
    n = int(integer_weights.sum())
    if not n:
        return math.inf
    # Exact alpha=1/10, not ceil of a rounded binary product.
    rank = n+1-(n+1)//10
    if rank > n:
        return math.inf
    return float(scores[np.searchsorted(np.cumsum(integer_weights, dtype=np.int64), rank)])


def _exact_mass(scores, source, scene_sizes, scene_weights, q):
    covered = np.bincount(source[scores <= q], minlength=len(scene_sizes))
    ids = np.flatnonzero((scene_weights > 0) & (covered > 0))
    return sum((Fraction(int(scene_weights[i])*int(covered[i]), int(scene_sizes[i]))
                for i in ids), Fraction(0))


def hcp_sorted(scores, source, scene_sizes, scene_weights):
    """Exact-decimal alpha with roundoff bracketing and rational tie fallback.

    The fallback decides mathematical equality; the error bound never relaxes
    the required coverage mass. Sorted scores include zero-weight rows.
    """
    k = int(scene_weights[scene_sizes > 0].sum())
    if k < 9:
        return math.inf
    active = scene_weights[source] > 0
    if not active.any():
        return math.inf
    s, sc = scores[active], source[active]
    w = scene_weights[sc].astype(np.longdouble)/scene_sizes[sc]
    cum = np.cumsum(w, dtype=np.longdouble)
    target = np.longdouble(9*(k+1))/10
    idx = min(int(np.searchsorted(cum, target)), len(s)-1)
    q = float(s[idx])
    # Evaluate at the ends of tied score blocks, where the CDF is defined.
    left = int(np.searchsorted(s, q, side='left'))
    right = int(np.searchsorted(s, q, side='right'))-1
    error = np.longdouble(8)*(len(s)+4)*np.finfo(np.longdouble).eps*max(k, 1)
    before = cum[left-1] if left else np.longdouble(0)
    after = cum[right]
    if before < target-error and after > target+error:
        return q
    exact_target = Fraction(9*(k+1), 10)
    # Exact rational verification only at numerically ambiguous boundaries.
    while _exact_mass(s, sc, scene_sizes, scene_weights, q) < exact_target:
        nxt = int(np.searchsorted(s, q, side='right'))
        if nxt == len(s):
            return math.inf
        q = float(s[nxt])
    while True:
        prv = int(np.searchsorted(s, q, side='left'))-1
        if prv < 0:
            break
        pq = float(s[prv])
        if _exact_mass(s, sc, scene_sizes, scene_weights, pq) < exact_target:
            break
        q = pq
    return q


@dataclass
class ClassData:
    scores: np.ndarray
    sources: np.ndarray
    sizes: np.ndarray
    scales: np.ndarray
    center_errors: np.ndarray
    angle_errors: np.ndarray
    widths: np.ndarray
    heights: np.ndarray
    square_gt: np.ndarray


def prepare(scores, scenes, classes, pred, gt, n_sources, n_classes):
    if not np.all(np.isfinite(scores)):
        raise ValueError('finite scores required')
    result = []
    for k in range(n_classes):
        idx = np.flatnonzero(classes == k)
        idx = idx[np.argsort(scores[idx], kind='stable')]
        sc, p, g = scenes[idx], pred[idx], gt[idx]
        if np.any(p[:, 2:4] <= 0) or np.any(g[:, 2:4] <= 0):
            raise ValueError('positive side lengths required for geometry diagnostics')
        pw, ph = np.maximum(p[:, 2], p[:, 3]), np.minimum(p[:, 2], p[:, 3])
        pa = p[:, 4]+(p[:, 3] > p[:, 2])*np.pi/2
        ga = g[:, 4]+(g[:, 3] > g[:, 2])*np.pi/2
        ae = np.abs((ga-pa+np.pi/2) % np.pi-np.pi/2)
        result.append(ClassData(scores[idx], sc, np.bincount(sc, minlength=n_sources),
                     np.sqrt(pw*ph), np.linalg.norm(p[:, :2]-g[:, :2], axis=1),
                     ae, pw, ph, g[:, 2] == g[:, 3]))
    return result


def angle_halfwidth(w, h, q):
    """Set-valued square convention from frozen theory; no fixed-shape shortcut."""
    out = np.full(len(w), np.pi/2)
    if math.isinf(q):
        return out
    gap = (w-h)/(2*np.sqrt(2))
    mask = (w > h) & (q < gap)
    if q == 0:
        out[mask] = 0
        return out
    # Stable identity sin^2(2 Delta) = q²(T-q²)/D²,
    # T=(w²+h²)/4; D=(w²-h²)/8, before critical isotropy.
    wm, hm = w[mask], h[mask]
    t = (wm*wm+hm*hm)/4
    d = (wm-hm)*(wm+hm)/8
    v = (q/d)*np.sqrt(np.maximum(t-q*q, 0))
    out[mask] = .5*np.arcsin(np.clip(v, 0, 1))
    return out


# These are scene means within the fixed class TP-positive population.
METRICS = ('coverage', 'infinite_fraction', 'normalized_radius',
           'angle_halfwidth', 'informative_angle_fraction', 'joint_center_angle')


def one_inner(data, counts, roles, geometry=False):
    nk = len(data)
    q = np.full((2, nk), np.inf)
    metrics = np.full((2, nk, len(METRICS) if geometry else 1), np.nan)
    support = np.zeros((3, nk, 3), dtype=np.int64)  # unique/weighted sources, weighted objects
    for k, d in enumerate(data):
        for rr in (CAL, DESIGN, EVAL):
            weight = counts*(roles == rr)
            present = (weight > 0) & (d.sizes > 0)
            support[rr-1, k] = [present.sum(), weight[present].sum(), np.dot(weight, d.sizes)]
        cw = counts*(roles == CAL)
        ew = counts*(roles == EVAL)
        if len(d.scores):
            q[0, k] = pooled_sorted(d.scores, cw[d.sources])
            q[1, k] = hcp_sorted(d.scores, d.sources, d.sizes, cw)
        present = (ew > 0) & (d.sizes > 0)
        if not present.any():
            continue
        mask = ew[d.sources] > 0
        sc = d.sources[mask]
        objweights = ew[sc]/d.sizes[sc]
        denominator = float(ew[present].sum())
        for method in range(2):
            qq = q[method, k]
            metrics[method, k, 0] = np.dot(objweights, d.scores[mask] <= qq)/denominator
            if geometry:
                half = angle_halfwidth(d.widths[mask], d.heights[mask], qq)
                angle_covered = np.where(d.square_gt[mask], half == np.pi/2, d.angle_errors[mask] <= half)
                joint = (d.center_errors[mask] <= qq) & angle_covered
                values = (math.isinf(qq), qq/d.scales[mask], half, half < np.pi/2, joint)
                for j, v in enumerate(values, 1):
                    metrics[method, k, j] = float(v) if np.isscalar(v) else np.dot(objweights, v)/denominator
    return q, metrics, support


def strict_mean(x, axis=0):
    # np.mean propagates NA/inf as intended. No nanmean or valid-draw filtering.
    return np.mean(x, axis=axis)

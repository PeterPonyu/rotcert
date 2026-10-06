"""Owned standalone S2 numerics; no rotcert/worktree imports.

All thresholds use exact integer counts at alpha=1/10. Score equality is covered.
Method ordering, condition ordering, RNG streams and metrics are frozen in protocol.json.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'
import math
import numpy as np
from scipy.special import ndtr, gammaln
from scipy.stats import poisson

METHODS = ('pooled', 'hcp', 'scene_max', 'crc_m', 'crc_d_m', 'crc_d_m_plus_poisson5')
METRICS = (
    'scene_risk', 'object_expectation_estimator', 'test_object_ratio',
    'scene_any_failure', 'finite_scene_rate', 'infinite_scene_rate',
    'finite_object_rate', 'infinite_object_rate', 'q_finite_mean',
    'q_finite_median', 'q_finite_max', 'finite_subset_object_risk',
    'population_scene_risk_lower', 'population_scene_risk_upper',
    'population_object_risk_lower', 'population_object_risk_upper',
    'population_finite_scene_rate_lower', 'population_finite_scene_rate_upper',
    'accounting_residual', 'population_object_minus_scene',
)
MS = np.array([1, 8, 64], dtype=np.int64)
PS = np.array([.70, .25, .05])
MEAN_LOG_M = float(PS @ np.log(MS))
SD_LOG_M = float(np.sqrt(PS @ (np.log(MS) - MEAN_LOG_M) ** 2))
ZMS = (np.log(MS) - MEAN_LOG_M) / SD_LOG_M
K_POIS = np.arange(65, dtype=np.int64)
POIS_P = np.exp(-5 + K_POIS * np.log(5.) - gammaln(K_POIS + 1.))
POIS_TAIL = float(poisson.sf(64, 5))


def conditions():
    result = []
    for ncal in (50, 200, 1000):
        for regime in ('constant8', 'heterogeneous'):
            for gamma in ((0.,) if regime == 'constant8' else (-.75, 0., .75)):
                for rho in (0., .5, .9):
                    result.append(dict(condition_id=len(result), ncal=ncal,
                                       regime=regime, gamma=gamma, rho=rho,
                                       M=8 if regime == 'constant8' else 64))
    return result


def population(condition):
    if condition['regime'] == 'constant8':
        return np.array([8], dtype=np.int64), np.array([1.]), np.array([0.])
    return MS, PS, ZMS


def rng(condition_id, replication_id, stream):
    # Independent fixed-purpose streams; adding a metric never advances data RNG.
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [20260928, 2, int(condition_id), int(replication_id), int(stream)])))


def generate_scenes(condition, count, random):
    if condition['regime'] == 'constant8':
        m = np.full(count, 8, dtype=np.int64)
        z = np.zeros(count)
    else:
        indices = random.choice(3, size=count, p=PS)
        m = MS[indices]
        z = ZMS[indices]
    starts = np.r_[0, np.cumsum(m)[:-1]]
    u = random.standard_normal(count)
    e = random.standard_normal(int(m.sum()))
    log_s = (np.repeat(condition['gamma'] * z + math.sqrt(condition['rho']) * u, m)
             + math.sqrt(1 - condition['rho']) * e)
    scores = np.exp(log_s)
    if not np.all(np.isfinite(scores)) or np.any(scores <= 0):
        raise FloatingPointError('S2 assumes finite strictly positive generated scores')
    return m, starts, scores


def cp_rank(size):
    if size < 0:
        raise ValueError('size must be nonnegative')
    return (9 * (int(size) + 1) + 9) // 10


def ranked_threshold(sorted_scores, rank):
    if rank < 1:
        return 0.
    if rank > len(sorted_scores):
        return math.inf
    return float(sorted_scores[rank - 1])


def crc_thresholds(sorted_scores, d):
    """Smallest nonnegative q with 10*N_exceed <= Ncal - 9*D, no slack."""
    d = np.asarray(d)
    if not np.issubdtype(d.dtype, np.integer) or np.any(d < 0):
        raise ValueError('D must be exact nonnegative integers')
    if np.any(d > (np.iinfo(np.int64).max // 16)):
        raise OverflowError('D outside safe int64 domain')
    n = len(sorted_scores)
    budget = (n - 9 * d) // 10
    required = n - budget
    q = np.full(d.shape, math.inf)
    zero = required <= 0
    q[zero] = 0.
    valid = (required >= 1) & (required <= n)
    q[valid] = np.asarray(sorted_scores)[required[valid] - 1]
    return q


def calibrate(m, starts, scores, M):
    if not (len(m) == len(starts) and int(m.sum()) == len(scores)):
        raise ValueError('inconsistent calibration representation')
    if np.any(m < 1) or np.any(64 % m != 0):
        raise ValueError('integer-weight HCP requires positive divisors of64')
    if M not in (8, 64) or np.any(m > M):
        raise ValueError('M must be valid frozen population bound')
    if not np.all(np.isfinite(scores)) or np.any(scores < 0):
        raise ValueError('finite nonnegative scores required')
    order = np.argsort(scores, kind='stable')
    sorted_scores = scores[order]
    weights = np.repeat(64 // m, m)[order]
    prefix = np.cumsum(weights, dtype=np.int64)
    needed = (9 * (len(m) + 1) * 64 + 9) // 10
    where = int(np.searchsorted(prefix, needed, side='left'))
    hcp = float(sorted_scores[where]) if where < len(scores) else math.inf
    maxima = np.sort(np.maximum.reduceat(scores, starts))
    scalar = np.array([ranked_threshold(sorted_scores, cp_rank(len(scores))),
                       hcp, ranked_threshold(maxima, cp_rank(len(m))),
                       float(crc_thresholds(sorted_scores, np.array(M)))])
    return sorted_scores, scalar


def marginal_risks(condition, sorted_scores, scalar):
    """Conditional-on-calibration population means, marginalizing new scene/FP.

    For FP, sum Poisson0..64 and bracket omitted probability analytically.
    Bracket concerns tail truncation, not a rigorous enclosure of float rounding.
    """
    ms, ps, zs = population(condition)
    em = float(ps @ ms)
    out = np.zeros((6, 6))  # scene lo/hi, object lo/hi, finite scene lo/hi
    means = condition['gamma'] * zs
    for a in range(6):
        d = ms[:, None] + K_POIS[None, :] if a == 5 else ms
        if a < 4:
            q = np.full(len(ms), scalar[a])
        else:
            q = crc_thresholds(sorted_scores, d)
        with np.errstate(divide='ignore'):
            tails = ndtr(means[:, None] - np.log(q)) if a == 5 else ndtr(means - np.log(q))
        if a == 5:
            tail = tails @ POIS_P
            finite = np.isfinite(q) @ POIS_P
            rest = POIS_TAIL
        else:
            tail = tails
            finite = np.isfinite(q).astype(float)
            rest = 0.
        sr = float(ps @ tail)
        ob = float((ps * ms) @ tail / em)
        fin = float(ps @ finite)
        out[a] = sr, min(1., sr + rest), ob, min(1., ob + rest), fin, min(1., fin + rest)
    return out


def replicate(condition, replication_id, neval=2000):
    cid = condition['condition_id']
    cm, cs, cal = generate_scenes(condition, condition['ncal'], rng(cid, replication_id, 0))
    em, es, scores = generate_scenes(condition, neval, rng(cid, replication_id, 1))
    d_fp = em + rng(cid, replication_id, 2).poisson(5, neval)
    sorted_scores, scalars = calibrate(cm, cs, cal, condition['M'])
    q = np.empty((6, neval))
    q[:4] = scalars[:, None]
    q[4] = crc_thresholds(sorted_scores, em)
    q[5] = crc_thresholds(sorted_scores, d_fp)
    ps_ms, ps_p, _ = population(condition)
    expected_m = float(ps_p @ ps_ms)
    population_values = marginal_risks(condition, sorted_scores, scalars)
    result = np.empty((6, len(METRICS)))
    weights = 64 // em
    total_m = int(em.sum())
    uniform = bool(np.all(em == em[0]))
    for a in range(6):
        failed = scores > np.repeat(q[a], em)
        n_fail = np.add.reduceat(failed.astype(np.int64), es)
        sum_fail = int(n_fail.sum())
        weighted_fail = int(weights @ n_fail)
        scene_risk = weighted_fail / (64 * neval)
        test_ratio = sum_fail / total_m
        covered = em - n_fail
        weighted_covered = int(weights @ covered)
        total_covered = int(covered.sum())
        if uniform and total_covered * (64 * neval) != weighted_covered * total_m:
            raise AssertionError('exact constant-m identity broken')
        scene_cov = weighted_covered / (64 * neval)
        object_cov = total_covered / total_m
        cov = float(np.mean((em - em.mean()) * (covered / em - scene_cov)) / em.mean())
        residual = (object_cov - scene_cov) - cov
        if abs(residual) > 8e-15:
            raise AssertionError('covariance accounting diagnostic failed')
        fin = np.isfinite(q[a])
        finite_q = q[a, fin]
        finite_objects = int(em[fin].sum())
        finite_fail = int(n_fail[fin].sum())
        pp = population_values[a]
        result[a] = (scene_risk, sum_fail / (neval * expected_m), test_ratio,
                     float(np.mean(n_fail > 0)), float(fin.mean()), float((~fin).mean()),
                     finite_objects / total_m, 1 - finite_objects / total_m,
                     float(finite_q.mean()) if len(finite_q) else np.nan,
                     float(np.median(finite_q)) if len(finite_q) else np.nan,
                     float(finite_q.max()) if len(finite_q) else np.nan,
                     finite_fail / finite_objects if finite_objects else np.nan,
                     *pp, residual, pp[2] - pp[0])
    # Aux audit values: no scores or source arrays need to persist to resume;
    # every replication regenerates exactly from its fixed independent seed.
    aux = np.array([int(cm.sum()), total_m, expected_m, int(d_fp.max())], dtype=float)
    return result, aux


def describe(values):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    vals = values[valid]
    n = len(vals)
    if n == 0:
        return dict(mean=None, sd=None, mcse=None, q025=None, median=None, q975=None,
                    minimum=None, maximum=None, n=0, missing=int(len(values)))
    sd = float(vals.std(ddof=1)) if n > 1 else None
    q = np.quantile(vals, [.025, .5, .975], method='linear')
    return dict(mean=float(vals.mean()), sd=sd, mcse=(sd / math.sqrt(n) if n > 1 else None),
                q025=float(q[0]), median=float(q[1]), q975=float(q[2]),
                minimum=float(vals.min()), maximum=float(vals.max()), n=n,
                missing=int(len(values) - n))


def summarize(condition, values, aux):
    out = dict(condition=condition, replications=len(values), methods={})
    for a, method in enumerate(METHODS):
        rec = {name: describe(values[:, a, i]) for i, name in enumerate(METRICS)}
        target = ('scene_risk' if method == 'hcp' else 'scene_any_failure'
                  if method == 'scene_max' else 'object_expectation_estimator'
                  if method.startswith('crc') else None)
        rec['theorem_target_empirical_metric'] = target
        rec['simulation_not_PAC'] = True
        rec['q_summary_scope'] = 'within-eval finite q summary then across reps; missing explicit; infinity separately'
        rec['finite_subset_warning'] = 'descriptive selected subset; unconditional guarantee does not transfer'
        rec['exact_constant_m_identity'] = condition['regime'] == 'constant8'
        if method.startswith('crc'):
            popidx = METRICS.index('population_object_risk_lower')
        elif method == 'hcp':
            popidx = METRICS.index('population_scene_risk_lower')
        else:
            popidx = None
        rec['calibration_fraction_population_risk_above_alpha'] = (
            float(np.mean(values[:, a, popidx] > .1)) if popidx is not None else None)
        out['methods'][method] = rec
    scene = METRICS.index('population_scene_risk_lower')
    obj = METRICS.index('population_object_risk_lower')
    out['paired_contrasts'] = {
        'pooled_minus_hcp_population_scene_risk': describe(values[:, 0, scene] - values[:, 1, scene]),
        'pooled_minus_hcp_population_object_risk': describe(values[:, 0, obj] - values[:, 1, obj]),
        'crc_d_m_minus_crc_m_population_object_risk': describe(values[:, 4, obj] - values[:, 3, obj]),
        'crc_d_poisson_minus_crc_d_m_population_object_risk': describe(values[:, 5, obj] - values[:, 4, obj]),
    }
    out['mean_eval_object_count'] = float(aux[:, 1].mean())
    out['population_E_m'] = float(aux[0, 2])
    return out

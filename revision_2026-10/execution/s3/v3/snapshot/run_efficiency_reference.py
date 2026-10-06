#!/usr/bin/env python3
"""Design-tuned, held-out efficiency for the frozen S1--S14 score roster.

Scalar scores use Mondrian HCP. Legacy vector scores retain their literal
per-coordinate object-pooled Bonferroni calibrators. S14 additionally reports
the original uncorrected quantile and object-pooled finite-sample correction.
Center-slice areas use the prescribed 90-ray quadrature, with exact ray radii
for unchanged-shape translations. Infinity is never silently dropped.
"""
import argparse
import hashlib
import io
import marshal
import sys
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import common as C
from rotcert import scene as S
from rotcert import scores_ext as X
from rotcert.gwd import canonicalize_le90, obb_gwd
from rotcert.matching import rotated_iou

ALPHA_GRID = np.round(np.arange(0.02, 0.3001, 0.0025), 4)
# Bind provenance to the actual executing module and immutable startup bytes.
_LOADED_CODE = sys._getframe().f_code
_RUNNER_BYTES = Path(__file__).read_bytes()
if compile(_RUNNER_BYTES, _LOADED_CODE.co_filename, 'exec', dont_inherit=True,
           optimize=sys.flags.optimize) != _LOADED_CODE:
    raise RuntimeError('runner source changed while it was being loaded')
RUNNER_SHA256 = hashlib.sha256(_RUNNER_BYTES).hexdigest()
RUNNER_CODE_SHA256 = hashlib.sha256(marshal.dumps(_LOADED_CODE)).hexdigest()

N_SUB, N_DIRS = 300, 90
VECTOR_COORDS = {
    'naive-coord': ('cx', 'cy', 'w', 'h', 'theta'),
    'wrapped-coord': ('cx', 'cy', 'w', 'h', 'theta'),
    'doubled': ('cx', 'cy', 'w', 'h', 'cos2t', 'sin2t'),
    'hull': ('cx', 'cy', 'half_w', 'half_h'),
}
SCORE_IDS = {
    'gwd': 'S1', 'gwd_normalized': 'S2', 'naive-coord': 'S3',
    'wrapped-coord': 'S4', 'doubled': 'S5', 'coord_max_additive': 'S6',
    'max_rank': 'S7', 'coord_max_multiplicative': 'S8', 'hull': 'S9',
    'iou': 'S10', 'kld': 'S11', 'hellinger': 'S12', 'bhattacharyya': 'S12',
    'corner_max': 'S13', 'expansion_margin': 'S14',
    'expansion_margin_uncorrected': 'S14', 'expansion_margin_scp': 'S14',
}


def iou_score(pred, gt):
    return np.array([1.0 - rotated_iou(p, g) for p, g in zip(pred, gt)])


def iou_shift_score(p0, dx, dy):
    w, h, t = (float(v) for v in canonicalize_le90(p0[2], p0[3], p0[4]))
    c, s = np.cos(t), np.sin(t)
    ov = np.clip(w - np.abs(c * dx + s * dy), 0, None) * np.clip(h - np.abs(-s * dx + c * dy), 0, None)
    return 1.0 - ov / (2.0 * w * h - ov)


SCALAR = {'gwd': obb_gwd, 'gwd_normalized': lambda p, g: obb_gwd(p, g) / np.sqrt(p[:, 2] * p[:, 3]),
          'iou': iou_score, **X.EXT_SCORES}


def coordinate_arrays(pred, gt):
    """Vectorize the exact residual definitions in rotcert.scores.SCORES."""
    pw, ph, pt = canonicalize_le90(pred[:, 2], pred[:, 3], pred[:, 4])
    gw, gh, gt_ = canonicalize_le90(gt[:, 2], gt[:, 3], gt[:, 4])
    xy = np.abs(pred[:, :2] - gt[:, :2])
    wh = np.column_stack([np.abs(pw - gw), np.abs(ph - gh)])
    dt = np.abs(pt - gt_)
    wrapped = np.minimum(np.mod(dt, math.pi), math.pi - np.mod(dt, math.pi))
    hp = np.column_stack([np.abs(np.cos(pt)) * pw + np.abs(np.sin(pt)) * ph,
                          np.abs(np.sin(pt)) * pw + np.abs(np.cos(pt)) * ph]) / 2
    hg = np.column_stack([np.abs(np.cos(gt_)) * gw + np.abs(np.sin(gt_)) * gh,
                          np.abs(np.sin(gt_)) * gw + np.abs(np.cos(gt_)) * gh]) / 2
    return {'naive-coord': np.column_stack([xy, wh, dt]),
            'wrapped-coord': np.column_stack([xy, wh, wrapped]),
            'doubled': np.column_stack([xy, wh, np.abs(np.cos(2 * pt) - np.cos(2 * gt_)),
                                       np.abs(np.sin(2 * pt) - np.sin(2 * gt_))]),
            'hull': np.column_stack([xy, np.abs(hp - hg)])}


def score_arrays(pred, gt):
    result = {name: f(pred, gt)[:, None] for name, f in SCALAR.items()}
    result.update(coordinate_arrays(pred, gt))
    result['expansion_margin_uncorrected'] = result['expansion_margin']
    result['expansion_margin_scp'] = result['expansion_margin']
    return result


def calibrator_name(name):
    if name in VECTOR_COORDS:
        return 'mondrian_object_pooled_bonferroni'
    if name == 'expansion_margin_uncorrected':
        return 'mondrian_object_quantile_linear_uncorrected'
    if name == 'expansion_margin_scp':
        return 'mondrian_object_pooled_finite_sample'
    return 'mondrian_hcp'


def quantile_grid(values, scenes, alphas, mode='hcp'):
    """All alpha thresholds for one class; same ranks/tolerance as the core."""
    values = np.asarray(values, float)
    if values.ndim == 1:
        values = values[:, None]
    alphas = np.asarray(alphas, float)
    if np.any((alphas <= 0) | (alphas >= 1)) or not np.isfinite(alphas).all():
        raise ValueError('alphas must be finite and strictly between zero and one')
    if not np.isfinite(values).all():
        raise ValueError('nonconformity scores must be finite')
    n, d = values.shape
    if n == 0:
        return np.full((len(alphas), d), math.inf)
    sorted_values = np.sort(values, axis=0, kind='stable')
    if mode in ('bonferroni', 'scp'):
        alpha_coord = alphas / d if mode == 'bonferroni' else alphas
        ranks = (n + 1) - np.floor((n + 1) * alpha_coord).astype(int)
        return np.vstack([sorted_values, np.full((1, d), math.inf)])[ranks - 1]
    if mode == 'uncorrected':
        return np.quantile(values, 1.0 - alphas, axis=0, method='linear')
    if mode != 'hcp' or d != 1:
        raise ValueError('HCP requires one scalar score')
    _, inverse, counts = np.unique(scenes, return_inverse=True, return_counts=True)
    order = np.argsort(values[:, 0], kind='stable')
    cumulative = np.cumsum(1.0 / counts[inverse][order])
    target = (1.0 - alphas) * (len(counts) + 1) - 1e-9
    indexes = np.searchsorted(cumulative, target, side='left')
    return np.append(values[order, 0], math.inf)[indexes, None]


def threshold_grids(name, values, scenes, classes, cal_mask, alphas):
    mode = 'bonferroni' if name in VECTOR_COORDS else 'hcp'
    if name == 'expansion_margin_uncorrected':
        mode = 'uncorrected'
    elif name == 'expansion_margin_scp':
        mode = 'scp'
    return {int(k): quantile_grid(values[cal_mask & (classes == k)],
                                 scenes[cal_mask & (classes == k)], alphas, mode)
            for k in np.unique(classes[cal_mask])}


def coverage_grid(values, scenes, classes, grids):
    """Scene-weighted joint coverage at every candidate alpha, with support counts."""
    n_alpha = next(iter(grids.values())).shape[0] if grids else len(ALPHA_GRID)
    support = np.isin(classes, list(grids))
    if not support.any():
        return np.full(n_alpha, math.nan), int(len(classes))
    _, inverse, counts = np.unique(scenes[support], return_inverse=True, return_counts=True)
    weights = np.zeros(len(classes))
    weights[support] = 1.0 / (counts[inverse] * len(counts))
    coverage = np.zeros(n_alpha)
    for k, q in grids.items():
        use = np.flatnonzero(classes == k)
        # Bound peak memory for large cells without changing the statistic.
        for start in range(0, len(use), 4096):
            part = use[start:start + 4096]
            covered = np.all(values[part, None, :] <= q[None, :, :], axis=2)
            coverage += weights[part] @ covered
    return coverage, int((~support).sum())


def _rectangle_radii(qx, qy, ax, ay):
    with np.errstate(divide='ignore', invalid='ignore'):
        rx = np.divide(qx[:, None], ax, out=np.full_like(ax, math.inf), where=ax > 1e-15)
        ry = np.divide(qy[:, None], ay, out=np.full_like(ay, math.inf), where=ay > 1e-15)
    return np.minimum(rx, ry)


def center_areas(name, pred, q, rank_model=None, n_dirs=N_DIRS):
    """Prescribed polar quadrature; exact translation radii, including unbounded sets."""
    q = np.asarray(q, float)
    if q.ndim == 1:
        q = q[:, None]
    if len(pred) != len(q):
        raise ValueError('thresholds and predictions must align')
    w, h, t = canonicalize_le90(pred[:, 2], pred[:, 3], pred[:, 4])
    angle = np.linspace(0, 2 * math.pi, n_dirs, endpoint=False)
    ax = np.abs(np.cos(angle[None, :] - t[:, None]))
    ay = np.abs(np.sin(angle[None, :] - t[:, None]))
    scalar = q[:, 0]
    unbounded = np.isposinf(scalar)
    zero = np.zeros(len(q), dtype=bool)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        if name in VECTOR_COORDS:
            ux = np.broadcast_to(np.abs(np.cos(angle)), ax.shape)
            uy = np.broadcast_to(np.abs(np.sin(angle)), ay.shape)
            radii = _rectangle_radii(q[:, 0], q[:, 1], ux, uy)
            unbounded = np.isposinf(q[:, :2]).any(axis=1)
            zero = (q[:, :2] == 0).any(axis=1)
        elif name == 'max_rank':
            if rank_model is None or not len(rank_model.sorted_design):
                raise ValueError('max_rank needs a nonempty design-only rank fit')
            n = len(rank_model.sorted_design)
            indexes = np.floor(np.nan_to_num(scalar, nan=0, posinf=1) * n + 1e-9).astype(int)
            indexes = np.clip(indexes, 0, n - 1)
            half = rank_model.sorted_design[indexes, :2]
            radii = _rectangle_radii(half[:, 0], half[:, 1], ax, ay)
            unbounded = scalar >= 1
            zero = (half == 0).any(axis=1) & ~unbounded
        elif name in ('gwd', 'gwd_normalized', 'corner_max'):
            radius = scalar * np.sqrt(w * h) if name == 'gwd_normalized' else scalar
            radii = np.broadcast_to(radius[:, None], ax.shape)
        elif name in ('coord_max_additive', 'expansion_margin',
                      'expansion_margin_uncorrected', 'expansion_margin_scp'):
            radii = _rectangle_radii(scalar, scalar, ax, ay)
        elif name == 'coord_max_multiplicative':
            radii = _rectangle_radii(scalar * np.maximum(w, X.MIN_SIDE),
                                     scalar * np.maximum(h, X.MIN_SIDE), ax, ay)
        elif name in ('kld', 'bhattacharyya', 'hellinger'):
            factor = 2.0 if name == 'kld' else 0.5
            target = scalar.copy()
            if name == 'hellinger':
                unbounded = scalar >= 1
                target = -np.log1p(-np.minimum(scalar, 1) ** 2)
            a = factor * (ax ** 2 / np.maximum(w[:, None], X.MIN_SIDE) ** 2
                          + ay ** 2 / np.maximum(h[:, None], X.MIN_SIDE) ** 2)
            radii = np.sqrt(target[:, None] / a)
        elif name == 'iou':
            unbounded = scalar >= 1
            z = np.minimum(scalar, 1)
            constant = z / (2 - z)
            aa, bb = ax / w[:, None], ay / h[:, None]
            discriminant = np.maximum((aa + bb) ** 2 - 4 * aa * bb * constant[:, None], 0)
            radii = 2 * constant[:, None] / (aa + bb + np.sqrt(discriminant))
        else:
            raise ValueError('unknown score ' + name)
        areas = math.pi * np.mean(radii ** 2, axis=1)
    areas = np.array(areas, copy=True)
    areas[unbounded] = math.inf
    areas[zero] = 0.0
    areas[np.isnan(q).any(axis=1)] = math.nan
    return areas


def area_summary(areas):
    """Unconditional on finite area; unsupported entries are counted separately."""
    areas = np.asarray(areas, float)
    use = areas[~np.isnan(areas)]
    ordered = np.sort(use)
    n = len(ordered)
    median = (ordered[n // 2] if n % 2 else
              (ordered[n // 2 - 1] + ordered[n // 2]) / 2) if n else math.nan
    return {'mean': float(np.mean(use)) if n else math.nan, 'median': float(median),
            'n_sample': int(len(areas)), 'n_supported': n,
            'n_finite': int(np.isfinite(areas).sum()), 'n_infinite': int(np.isposinf(areas).sum()),
            'n_out_of_support': int(np.isnan(areas).sum())}


def area_ratio(numerator, denominator):
    if math.isnan(numerator) or math.isnan(denominator):
        return math.nan, 'out_of_support'
    if math.isinf(numerator) and math.isinf(denominator):
        return math.nan, 'both_unbounded'
    if denominator == 0:
        return (math.nan, 'both_zero') if numerator == 0 else (math.inf, 'zero_reference')
    if math.isinf(denominator):
        return 0.0, 'finite_over_unbounded'
    if math.isinf(numerator):
        return math.inf, 'unbounded_numerator'
    return numerator / denominator, 'finite'


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return None if math.isnan(value) else ('Infinity' if value == math.inf else
               '-Infinity' if value == -math.inf else float(value))
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(json_safe(obj), indent=1, allow_nan=False))
    tmp.replace(path)


def assert_unchanged_source(path, expected_hash):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_hash:
        raise RuntimeError('input changed during execution: ' + str(path))


def run_cell(cell_id, R, out_dir=None):
    if R < 1:
        raise ValueError('R must be positive')
    assert_unchanged_source(__file__, RUNNER_SHA256)
    meta = C.cell(cell_id)
    cache_path = C.CACHE / (cell_id + '.npz')
    cache_bytes = cache_path.read_bytes()
    cache_sha256 = hashlib.sha256(cache_bytes).hexdigest()
    with np.load(io.BytesIO(cache_bytes), allow_pickle=False) as archive:
        c = {key: archive[key] for key in archive.files}
    del cache_bytes
    pred, gt, sc, cl = c['tp_pred'], c['tp_gt'], c['tp_scene'], c['tp_cls']
    scores = score_arrays(pred, gt)
    residuals = X.wrapped_coord_residuals(pred, gt)
    fields = ('alpha_prime_by_split', 'cov_design_scene', 'design_coverage_gap', 'cov_eval_scene',
              'q_hat', 'center_area_mean', 'center_area_median', 'area_counts',
              'n_eval_out_of_support', 'n_design_out_of_support')
    res = {name: {field: [] for field in fields} for name in SCORE_IDS}
    samples = []
    for r, cal, des, ev in C.iter_splits(c['scene_names'], C.tp_universe(c), R):
        cal_m, des_m, ev_m = (C.mask_for(sc, x) for x in (cal, des, ev))
        if not des_m.any():
            raise ValueError('empty design split: cannot tune or fit max-rank')
        rank = X.MaxRank.fit(residuals[des_m])
        scores['max_rank'] = rank.score(residuals)[:, None]
        ev_idx = np.flatnonzero(ev_m)
        sub = np.random.default_rng(r).choice(ev_idx, size=min(N_SUB, ev_idx.size), replace=False)
        samples.append({'r': r, 'cache_row_indexes': sub.tolist()})
        for name in SCORE_IDS:
            values = scores[name]
            grids = threshold_grids(name, values, sc, cl, cal_m, ALPHA_GRID)
            design_curve, n_des_missing = coverage_grid(values[des_m], sc[des_m], cl[des_m], grids)
            if not np.isfinite(design_curve).any():
                raise ValueError('design split has no classes in calibration support')
            best = int(np.nanargmin(np.abs(design_curve - 0.90)))
            qd = {k: v[best:best + 1] for k, v in grids.items()}
            eval_curve, n_ev_missing = coverage_grid(values[ev_m], sc[ev_m], cl[ev_m], qd)
            dim = values.shape[1]
            q = np.array([qd.get(int(k), np.full((1, dim), math.nan))[0] for k in cl[sub]])
            areas = center_areas(name, pred[sub], q, rank_model=rank)
            summary = area_summary(areas)
            record = res[name]
            q_record = {k: (dict(zip(VECTOR_COORDS[name], v[0].tolist())) if name in VECTOR_COORDS
                             else float(v[0, 0])) for k, v in qd.items()}
            entries = {'alpha_prime_by_split': float(ALPHA_GRID[best]),
                       'cov_design_scene': float(design_curve[best]),
                       'design_coverage_gap': float(abs(design_curve[best] - .90)),
                       'cov_eval_scene': float(eval_curve[0]), 'q_hat': q_record,
                       'center_area_mean': summary['mean'], 'center_area_median': summary['median'],
                       'area_counts': {k: v for k, v in summary.items() if k not in ('mean', 'median')},
                       'n_eval_out_of_support': n_ev_missing, 'n_design_out_of_support': n_des_missing}
            for field, value in entries.items():
                record[field].append(value)
        print(f'{cell_id}: split {r + 1}/{R}', flush=True)
    ratios, ratio_status = {}, {}
    for name, record in res.items():
        record['alpha_prime'] = record['alpha_prime_by_split']
        paired = [area_ratio(a, b) for a, b in zip(record['center_area_mean'], res['gwd']['center_area_mean'])]
        ratios[name] = [p[0] for p in paired]
        ratio_status[name] = [p[1] for p in paired]
    assert_unchanged_source(__file__, RUNNER_SHA256)
    assert_unchanged_source(cache_path, cache_sha256)
    out = {'schema_version': 2, 'cell': meta, 'R': R, 'alpha_grid': ALPHA_GRID.tolist(),
           'score_ids': SCORE_IDS, 'calibrators': {n: calibrator_name(n) for n in SCORE_IDS},
           'per_score': res, 'area_ratio_vs_gwd': ratios, 'area_ratio_status': ratio_status,
           'area_samples': samples, 'n_dirs': N_DIRS,
           'runner_sha256': RUNNER_SHA256,
           'cache_sha256': cache_sha256,
           'executed_code_sha256': RUNNER_CODE_SHA256, 'python_version': sys.version,
           'max_rank_fit': 'design only; never calibration or evaluation',
           'nonfinite_encoding': 'Infinity/-Infinity strings; undefined=JSON null with counts/status',
           'note': 'Design-tuned alpha; held-out coverage; center slice only, not union footprint. '
                   'All sampled supported TPs retained, including infinite areas. '
                   'S14 uncorrected/SCP are class-Mondrian method-level comparisons, '
                   'not the original EAV-DETR pose-Mondrian system protocol.'}
    write_json(out, Path(out_dir or C.HERE / 'out') / f'eff_{cell_id}.json')
    return cell_id


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument('--cell', choices=[x['cell_id'] for x in C.cells()])
    which.add_argument('--all', action='store_true')
    ap.add_argument('--R', type=int, default=50)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out-dir', type=Path)
    a = ap.parse_args()
    if a.R < 1 or a.workers < 1:
        ap.error('R and workers must be positive')
    ids = [x['cell_id'] for x in C.cells()] if a.all else [a.cell]
    with ProcessPoolExecutor(max_workers=min(a.workers, len(ids))) as ex:
        for cid in ex.map(run_cell, ids, [a.R] * len(ids), [a.out_dir] * len(ids)):
            print('done', cid, flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""System-level comparison on CODrone for the RotCert revision (2026-10-04): EAV-DETR trained with the authors'
configuration, its detections on the official validation (calibration) and test (evaluation) tiles, and four ways of
building localization regions from the same detections at alpha = 0.10.

  native-pose    the EAV calibration as described: class-agnostic Hungarian matching of detections with score > 0.5 at
                 IoU > 0.5, expansion margin of the predicted polygon needed to contain the ground-truth polygon (the
                 authors' geometry_utils), one uncorrected linear quantile per flight condition (altitude x camera angle
                 parsed from the file name), object-level coverage on the test tiles;
  native-global  the same with one global quantile, which is what the released calibrate.py computes because the dataset
                 target carries no file name and every tile falls into the 'unknown' condition;
  margin-hcp     the authors' margin score with this paper's protocol: class-exact greedy matching at confidence >= 0.05
                 and rotated IoU >= 0.5, class-wise hierarchical (source-uniform) calibration with the frame as source;
  gwd-hcp        the Gaussian Wasserstein score with the same protocol (this paper's method).

The native variants are also reported with the finite-sample corrected quantile. Coverage is reported object-level (as
the EAV evaluation does) and frame-uniform, with a fixed twelve-class dictionary. A complete-dictionary macro is NA
when any evaluation class is absent; the supported-class macro is a separately labelled descriptive quantity. The
95% percentile intervals use 1,000 resamples of test frames with thresholds fixed, and are unavailable if any draw
misses a required class. They are approximate, not a coverage guarantee. Detection-side readouts (recall, false-positive share) and the overlap of
locations and recordings between the official validation and test splits are recorded. Descriptive: the official splits
share locations, so no source-scene certificate is claimed.

Usage: analyze_eav.py --exports DIR --eav-repo DIR --s1 DIR --out FILE
"""
import argparse
import collections
import json
import math
import os
import re
import sys
import time
from multiprocessing import Pool

import numpy as np
from export_contract import CLASSES, load_shards, sha256, strict_json

ALPHA = 0.10
TILE = 1024.0
B = 1000


def load_split(exports, split, allow_partial=False):
    return load_shards(exports, split, allow_partial)


def frame_of(tile):
    return tile.split('__1__')[0]


def condition_of(name):
    m = re.search(r'(\d+)m_(\d+)c', name)
    return f'{m.group(1)}m_{m.group(2)}deg' if m else 'unknown'


def recording_of(frame):
    return re.split(r'_frame_', frame)[0]


def location_of(frame):
    toks = [t for t in recording_of(frame).replace(' ', '').split('_') if t]
    keep = [t for t in toks if t not in ('day', 'night') and not re.fullmatch(r'\d+m|\d+c|\d+', t)]
    return '_'.join(keep)


def to_px(b):
    p = b.astype(np.float64).copy()
    p[:, :4] *= TILE
    return p


# ---------------------------------------------------------------- matching
def native_match(split, iou_fn):
    """Class-agnostic Hungarian matching as in the authors' calibrate.py: score > 0.5, cost 1 - IoU, keep IoU > 0.5."""
    from scipy.optimize import linear_sum_assignment
    import torch
    score = split['pred_prob'].max(1)
    hi = score > 0.5
    pairs = []
    gt_by_tile = collections.defaultdict(list)
    for j, t in enumerate(split['gt_tile']):
        gt_by_tile[int(t)].append(j)
    pr_by_tile = collections.defaultdict(list)
    for i in np.flatnonzero(hi):
        pr_by_tile[int(split['pred_tile'][i])].append(i)
    for t, pi in pr_by_tile.items():
        gj = gt_by_tile.get(t, [])
        if not gj:
            continue
        pb = torch.as_tensor(split['pred_box'][pi]).float().cuda()
        gb = torch.as_tensor(split['gt_box'][gj]).float().cuda()
        ious = iou_fn(pb.unsqueeze(1).expand(-1, len(gj), -1), gb.unsqueeze(0).expand(len(pi), -1, -1))
        cost = (1.0 - ious).cpu().numpy()
        if not np.isfinite(cost).all():
            raise ValueError(f'non-finite native Hungarian cost at tile {split["names"][t]}')
        r, c = linear_sum_assignment(cost)
        for a, b in zip(r, c):
            if 1.0 - cost[a, b] > 0.5:
                pairs.append((pi[a], gj[b]))
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


def greedy_match(split, conf=0.05, thr=0.5):
    """Class-exact one-to-one matching in descending confidence, rotated IoU >= thr (this paper's protocol)."""
    import torch
    from mmcv.ops import box_iou_rotated
    prob = split['pred_prob']
    cls = prob.argmax(1)
    score = prob.max(1)
    keep = score >= conf
    gt_by_tile = collections.defaultdict(list)
    for j, t in enumerate(split['gt_tile']):
        gt_by_tile[int(t)].append(j)
    pr_by_tile = collections.defaultdict(list)
    for i in np.flatnonzero(keep):
        pr_by_tile[int(split['pred_tile'][i])].append(i)
    pairs, fp = [], np.zeros(len(CLASSES), np.int64)
    for t in sorted(set(pr_by_tile) | set(gt_by_tile)):
        pi = sorted(pr_by_tile.get(t, []), key=lambda i: -score[i])
        gj = gt_by_tile.get(t, [])
        if not pi:
            continue
        if gj:
            ious = box_iou_rotated(torch.as_tensor(to_px(split['pred_box'][pi])).float(),
                                   torch.as_tensor(to_px(split['gt_box'][gj])).float()).numpy()
            if not np.isfinite(ious).all():
                raise ValueError(f'non-finite protocol IoU at tile {split["names"][t]}')
        used = set()
        for a, i in enumerate(pi):
            best, bj = thr, -1
            for b, j in enumerate(gj):
                if j in used or split['gt_label'][j] != cls[i]:
                    continue
                if ious[a, b] >= best:
                    best, bj = ious[a, b], b
            if bj >= 0:
                used.add(gj[bj]); pairs.append((i, gj[bj]))
            else:
                fp[cls[i]] += 1
    return np.array(pairs, dtype=np.int64).reshape(-1, 2), fp


# ---------------------------------------------------------------- scores
_GEO = None


def _margin(args):
    p, g = args
    return float(_GEO.calculate_expansion_margin(_GEO.convert_to_corners(p), _GEO.convert_to_corners(g)))


def _init_geo(repo):
    global _GEO
    sys.path.insert(0, os.path.join(repo, 'tools'))
    import geometry_utils as geo
    _GEO = geo


def margins(split, pairs, repo, procs):
    if not len(pairs):
        return np.zeros(0)
    # Match the released calibrator's float32 pixel boxes before its geometry helpers.
    P = to_px(split['pred_box'][pairs[:, 0]]).astype(np.float32)
    G = to_px(split['gt_box'][pairs[:, 1]]).astype(np.float32)
    with Pool(procs, initializer=_init_geo, initargs=(repo,)) as pool:
        result = np.array(pool.map(_margin, list(zip(P, G)), chunksize=256))
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError('native geometry returned a non-finite or negative expansion score')
    return result


# ---------------------------------------------------------------- calibration
def q_linear(x, a=ALPHA):
    return float(np.quantile(x, 1 - a)) if len(x) else math.inf


def q_conformal(x, a=ALPHA):
    n = len(x)
    r = n + 1 - (n + 1) // 10 if a == ALPHA else math.ceil((n + 1) * (1 - a))
    return math.inf if r > n else float(np.sort(x)[r - 1])


def hcp_by_class(core, cls, frame, score, n_frames):
    out = {}
    for k in range(len(CLASSES)):
        m = cls == k
        if not m.any():
            out[k] = math.inf
            continue
        order = np.argsort(score[m], kind='stable')
        s, f = score[m][order], frame[m][order]
        sizes = np.bincount(f, minlength=n_frames)
        weights = (sizes > 0).astype(np.int64)
        out[k] = core.hcp_sorted(s, f, sizes, weights)
    return out


def frame_uniform(covered, frame, cls=None):
    """Plain frame average, or a complete-dictionary macro (NaN when an evaluation class is missing)."""
    if cls is None:
        tot = collections.defaultdict(lambda: [0, 0])
        for c, f in zip(covered, frame):
            tot[f][0] += c; tot[f][1] += 1
        return float(np.mean([a / b for a, b in tot.values()])) if tot else float('nan')
    vals = []
    for k in range(len(CLASSES)):
        m = cls == k
        vals.append(frame_uniform(covered[m], frame[m]))
    return float(np.mean(vals)) if vals else float('nan')


def boot(covered, frame, cls, rng, macro=False):
    """Object-pooled 95% interval over test-frame resamples, thresholds fixed.

    Class macros must use boot_macro so undefined draws cannot be silently omitted.
    """
    if macro:
        raise ValueError('use boot_macro for fixed-dictionary class summaries')
    frames = np.unique(frame)
    if len(frames) == 0:
        return None
    pos = {f: i for i, f in enumerate(frames)}
    fi = np.array([pos[f] for f in frame])
    n = len(frames)
    W = rng.multinomial(n, np.full(n, 1 / n), size=B).astype(float)
    cov_sum = np.bincount(fi, weights=covered.astype(float), minlength=n)
    cnt = np.bincount(fi, minlength=n).astype(float)
    num, den = W @ cov_sum, W @ cnt
    v = num / den
    return [float(np.quantile(v, .025)), float(np.quantile(v, .975))]


def class_frame_uniform(covered, frame, cls):
    """Twelve class entries; a class without evaluation TP is undefined, not zero."""
    return [frame_uniform(covered[cls == k], frame[cls == k]) for k in range(len(CLASSES))]


def macro_summary(covered, frame, cls, q_by_class=None):
    per = class_frame_uniform(covered, frame, cls)
    supported = [k for k in range(len(CLASSES)) if math.isfinite(per[k])]
    vacuous = [k for k in supported if q_by_class is not None and math.isinf(q_by_class[k])]
    complete = len(supported) == len(CLASSES)
    return dict(complete_dictionary_macro=float(np.mean(per)) if complete else None,
                complete_dictionary_status='AVAILABLE' if complete else 'UNAVAILABLE_MISSING_EVALUATION_CLASS',
                supported_class_macro=float(np.mean([per[k] for k in supported])) if supported else None,
                supported_classes=[CLASSES[k] for k in supported], n_supported=len(supported),
                n_dictionary=len(CLASSES), missing_evaluation_classes=[CLASSES[k] for k in range(len(CLASSES)) if k not in supported],
                vacuous_infinite_threshold_classes=[CLASSES[k] for k in vacuous] if q_by_class is not None else None,
                per_class={CLASSES[k]: (per[k] if math.isfinite(per[k]) else None) for k in range(len(CLASSES))})


def _strict_interval(v):
    bad = int(np.sum(~np.isfinite(v)))
    if bad or not len(v):
        return dict(interval=None, undefined_draws=bad, draws=int(len(v)), status='UNAVAILABLE_MISSING_REQUIRED_DRAWS')
    return dict(interval=[float(np.quantile(v, .025)), float(np.quantile(v, .975))], undefined_draws=0, draws=int(len(v)),
                status='AVAILABLE_APPROXIMATE_PERCENTILE')


def boot_macro(covered, frame, cls, rng):
    """Fixed-dictionary and supported-class 95% frame-resample intervals, thresholds fixed.

    Consumes the same multinomial draw as the v2 macro bootstrap; undefined draws are counted, never dropped.
    """
    frames = np.unique(frame)
    if len(frames) == 0:
        return None
    pos = {f: i for i, f in enumerate(frames)}
    fi = np.array([pos[f] for f in frame])
    n = len(frames)
    W = rng.multinomial(n, np.full(n, 1 / n), size=B).astype(float)
    per_class = np.full((len(CLASSES), B), np.nan)
    for k in range(len(CLASSES)):
        m = cls == k
        if not m.any():
            continue
        cov_sum = np.bincount(fi[m], weights=covered[m].astype(float), minlength=n)
        cnt = np.bincount(fi[m], minlength=n).astype(float)
        frac = np.divide(cov_sum, cnt, out=np.zeros(n), where=cnt > 0)
        present = (cnt > 0).astype(float)
        with np.errstate(invalid='ignore', divide='ignore'):
            per_class[k] = (W @ frac) / (W @ present)
    supported = [k for k in range(len(CLASSES)) if (cls == k).any()]
    supp = per_class[supported].mean(axis=0) if supported else np.full(B, np.nan)
    return dict(complete_dictionary=_strict_interval(per_class.mean(axis=0)), supported_classes=_strict_interval(supp),
                supported_class_list=[CLASSES[k] for k in supported])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exports', required=True)
    ap.add_argument('--eav-repo', required=True)
    ap.add_argument('--s1', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--procs', type=int, default=96)
    ap.add_argument('--allow-partial', action='store_true', help='diagnostic only: accept intentionally limited exports')
    a = ap.parse_args()
    if a.procs < 1:
        ap.error('procs must be positive')
    if os.path.exists(a.out):
        raise SystemExit('output exists')
    t0 = time.time()
    S = {s: load_split(a.exports, s, a.allow_partial) for s in ('val', 'test')}
    for key in ('checkpoint_sha256', 'config_sha256', 'min_score'):
        if S['val'][key] != S['test'][key]:
            raise ValueError(f'val/test exports disagree on {key}')
    sys.path.insert(0, a.eav_repo)
    sys.path.insert(0, a.s1)
    import core                                     # frozen S1 primitives (hcp_sorted, angle_halfwidth)
    from vendor.gwd import obb_gwd                  # frozen GWD score (le90, radians)
    from src.zoo.eavdetr.oriented_ops import oriented_box_iou
    rng = np.random.default_rng(20261004)
    res = dict(schema='rotcert.eav-codrone-analysis.v3', alpha=ALPHA, inputs={}, splits={},
               diagnostic_only=a.allow_partial,
               class_dictionary=list(CLASSES),
               bootstrap=dict(replicates=B, seed=20261004, native_macro_seed=20261005, unit='frame', thresholds='fixed',
                              interval='approximate percentile, 95%',
                              missing_class_draws='counted; interval unavailable if any required draw is undefined',
                              sampling_population='evaluation frames with at least one matched TP under each protocol'),
               export_receipts={s: S[s]['receipts'] for s in S},
               limitations=['Coverage is conditional on matched detections, not all ground-truth objects.',
                            'Official val/test recordings and locations overlap; no independent-scene certificate is claimed.',
                            'Native pose is a filename-enabled reconstruction; native global reproduces released loader behavior.',
                            'Infinite thresholds are uninformative; conditional coverage alone is not detection quality.'],
               dependencies={os.path.join(a.s1, 'core.py'): sha256(os.path.join(a.s1, 'core.py')),
                             os.path.join(a.s1, 'vendor/gwd.py'): sha256(os.path.join(a.s1, 'vendor/gwd.py')),
                             os.path.join(a.eav_repo, 'tools/geometry_utils.py'): sha256(os.path.join(a.eav_repo, 'tools/geometry_utils.py')),
                             os.path.join(a.eav_repo, 'src/zoo/eavdetr/oriented_ops.py'): sha256(os.path.join(a.eav_repo, 'src/zoo/eavdetr/oriented_ops.py'))})
    for s, d in S.items():
        res['inputs'].update(d['files'])
        frames = sorted({frame_of(t) for t in d['names']})
        res['splits'][s] = dict(tiles=len(d['names']), frames=len(frames), gt=int(len(d['gt_box'])),
                                kept_detections=int(len(d['pred_box'])), complete=d['complete'],
                                split_tiles=d['split_tiles'], shards=d['shards'], num_shards=d['num_shards'])
    fv = {frame_of(t) for t in S['val']['names']}; ft = {frame_of(t) for t in S['test']['names']}
    res['independence'] = dict(
        frames_shared=len(fv & ft),
        locations_val=len({location_of(f) for f in fv}), locations_test=len({location_of(f) for f in ft}),
        locations_shared=len({location_of(f) for f in fv} & {location_of(f) for f in ft}),
        recordings_val=len({recording_of(f) for f in fv}), recordings_test=len({recording_of(f) for f in ft}),
        recordings_shared=len({recording_of(f) for f in fv} & {recording_of(f) for f in ft}))
    print(json.dumps(res['independence']), flush=True)
    # ---------------- native EAV
    nat = {}
    for s in ('val', 'test'):
        d = S[s]
        pairs = native_match(d, oriented_box_iou)
        mg = margins(d, pairs, a.eav_repo, a.procs)
        tiles = [d['names'][int(t)] for t in d['pred_tile'][pairs[:, 0]]] if len(pairs) else []
        nat[s] = dict(pairs=pairs, margin=mg, cond=np.array([condition_of(t) for t in tiles]),
                      frame=np.array([frame_of(t) for t in tiles]), cls=d['gt_label'][pairs[:, 1]] if len(pairs) else np.zeros(0, int),
                      pw=np.sqrt(np.prod(to_px(d['pred_box'][pairs[:, 0]])[:, 2:4], axis=1)) if len(pairs) else np.zeros(0))
        print('native', s, len(pairs), flush=True)
    v, t = nat['val'], nat['test']
    conds = sorted(set(v['cond']) | set(t['cond']))
    out_native = {}
    for variant, qf in (('linear', q_linear), ('conformal', q_conformal)):
        q_global = qf(v['margin'])
        q_cond = {c: qf(v['margin'][v['cond'] == c]) for c in conds}
        for mode in ('pose', 'global'):
            q = np.array([q_cond[c] if mode == 'pose' else q_global for c in t['cond']])
            cov = t['margin'] <= q
            out_native[f'{mode}-{variant}'] = dict(
                q_global=q_global, q_by_condition={c: q_cond[c] for c in conds} if mode == 'pose' else None,
                infinite_conditions=[c for c in conds if math.isinf(q_cond[c])] if mode == 'pose' else None,
                uninformative_test_pairs=int(np.isinf(q).sum()),
                uninformative_test_pairs_by_gt_class={CLASSES[k]: int(np.sum(np.isinf(q) & (t['cls'] == k)))
                                                     for k in range(len(CLASSES))},
                calib_pairs_by_condition={c: int((v['cond'] == c).sum()) for c in conds},
                object_coverage=float(cov.mean()) if len(cov) else None,
                object_coverage_interval=boot(cov, t['frame'], t['cls'], rng, macro=False) if len(cov) else None,
                coverage_by_condition={c: float(cov[t['cond'] == c].mean()) for c in conds if (t['cond'] == c).any()},
                frame_uniform_class_macro=macro_summary(cov, t['frame'], t['cls']),
                frame_uniform_interval=boot_macro(cov, t['frame'], t['cls'], np.random.default_rng(20261005)),
                geometry_summary_scope='object-pooled over all evaluation geometric matches; includes infinite thresholds',
                finite_threshold_test_pairs=int(np.isfinite(q).sum()),
                normalized_margin_median=float(np.median(q / t['pw'])) if len(cov) else None,
                margin_px_median=float(np.median(q)) if len(cov) else None)
    res['native'] = dict(
        test_pairs=int(len(t['pairs'])), val_pairs=int(len(v['pairs'])), variants=out_native,
        class_support_scope='ground-truth class of class-agnostic geometric matches, not class-correct recall',
        class_support={s: {CLASSES[k]: dict(
            gt=int(np.sum(S[s]['gt_label'] == k)),
            geometric_pairs=int(np.sum(nat[s]['cls'] == k)),
            tp_bearing_frames=int(len(set(nat[s]['frame'][nat[s]['cls'] == k]))))
            for k in range(len(CLASSES))} for s in ('val', 'test')})
    # ---------------- this paper's protocol on the same detections
    prot = {}
    for s in ('val', 'test'):
        d = S[s]
        pairs, fp = greedy_match(d)
        P, G = to_px(d['pred_box'][pairs[:, 0]]), to_px(d['gt_box'][pairs[:, 1]])
        tiles = [d['names'][int(x)] for x in d['pred_tile'][pairs[:, 0]]]
        frames = np.array([frame_of(x) for x in tiles])
        fu = {f: i for i, f in enumerate(sorted(set(frames)))}
        prot[s] = dict(pairs=pairs, fp=fp, P=P, G=G, cls=d['gt_label'][pairs[:, 1]], frame=frames,
                       fidx=np.array([fu[f] for f in frames], dtype=np.int64), nf=len(fu),
                       gwd=obb_gwd(P, G), margin=margins(d, pairs, a.eav_repo, a.procs))
        print('protocol', s, len(pairs), flush=True)
    v, t = prot['val'], prot['test']
    dt = S['test']
    gt_per_class = np.bincount(dt['gt_label'], minlength=len(CLASSES))
    tp_per_class = np.bincount(t['cls'], minlength=len(CLASSES))
    res['detection_test'] = dict(
        gt=int(len(dt['gt_box'])), tp=int(len(t['pairs'])), fp=int(t['fp'].sum()),
        recall_pooled=float(len(t['pairs']) / len(dt['gt_box'])) if len(dt['gt_box']) else None,
        recall_supported_gt_class_macro=float(np.mean([tp_per_class[k] / gt_per_class[k] for k in range(len(CLASSES)) if gt_per_class[k]])) if gt_per_class.any() else None,
        recall_supported_gt_classes=[CLASSES[k] for k in range(len(CLASSES)) if gt_per_class[k]],
        recall_complete_dictionary_macro=float(np.mean(tp_per_class / gt_per_class)) if gt_per_class.all() else None,
        fp_share=float(t['fp'].sum() / (t['fp'].sum() + len(t['pairs']))) if t['fp'].sum() + len(t['pairs']) else None,
        scope='class-exact common matching; identical for margin-HCP and GWD-HCP',
        per_class={CLASSES[k]: dict(gt=int(gt_per_class[k]), tp=int(tp_per_class[k]),
                                  fp=int(t['fp'][k]), fn=int(gt_per_class[k] - tp_per_class[k])) for k in range(len(CLASSES))})
    out_prot = {}
    for name, key in (('gwd-hcp', 'gwd'), ('margin-hcp', 'margin')):
        q = hcp_by_class(core, v['cls'], v['fidx'], v[key], v['nf'])
        qt = np.array([q[int(k)] for k in t['cls']])
        cov = t[key] <= qt
        pw = np.maximum(t['P'][:, 2], t['P'][:, 3]); ph = np.minimum(t['P'][:, 2], t['P'][:, 3])
        entry = dict(q_by_class={CLASSES[k]: q[k] for k in range(len(CLASSES))},
                     test_pairs=int(len(cov)), uninformative_test_pairs=int(np.isinf(qt).sum()),
                     infinite_classes=[CLASSES[k] for k in range(len(CLASSES)) if math.isinf(q[k])],
                     calib_pairs_by_class={CLASSES[k]: int(np.sum(v['cls'] == k)) for k in range(len(CLASSES))},
                     test_pairs_by_class={CLASSES[k]: int(np.sum(t['cls'] == k)) for k in range(len(CLASSES))},
                     calib_sources_by_class={CLASSES[k]: int(len(set(v['frame'][v['cls'] == k]))) for k in range(len(CLASSES))},
                     frame_uniform_class_macro=macro_summary(cov, t['frame'], t['cls'], q_by_class=q),
                     frame_uniform_interval=boot_macro(cov, t['frame'], t['cls'], rng),
                     object_coverage=float(cov.mean()) if len(cov) else None,
                     object_coverage_interval=boot(cov, t['frame'], t['cls'], rng, macro=False),
                     per_class_frame_uniform={CLASSES[k]: (frame_uniform(cov[t['cls'] == k], t['frame'][t['cls'] == k])
                                                           if (t['cls'] == k).any() else None) for k in range(len(CLASSES))})
        finite = np.isfinite(qt)
        entry['geometry_summary_scope'] = 'object-pooled over all evaluation TPs; includes infinite thresholds'
        entry['finite_threshold_test_pairs'] = int(finite.sum())
        if key == 'gwd':
            entry['normalized_center_radius_median'] = float(np.median(qt / np.sqrt(pw * ph))) if len(cov) else None
            entry['normalized_center_radius_finite_threshold_only_median'] = float(np.median(qt[finite] / np.sqrt(pw * ph)[finite])) if finite.any() else None
            half = np.concatenate([core.angle_halfwidth(pw[t['cls'] == k], ph[t['cls'] == k], q[int(k)]) for k in np.unique(t['cls'])]) if len(cov) else np.zeros(0)
            entry['informative_orientation_share'] = float(np.mean(half < np.pi / 2)) if len(half) else None
            entry['informative_orientation_share_scope'] = 'object-pooled over evaluation TPs, not frame-uniform'
        else:
            entry['normalized_margin_median'] = float(np.median(qt / np.sqrt(pw * ph))) if len(cov) else None
            entry['normalized_margin_finite_threshold_only_median'] = float(np.median(qt[finite] / np.sqrt(pw * ph)[finite])) if finite.any() else None
        out_prot[name] = entry
    res['protocol'] = out_prot
    res['seconds'] = time.time() - t0
    res['seconds_scope'] = ('analysis wall time including loading, matching, scores, bootstrap and summaries; '
                            'excludes detector inference; not a per-arm or complete-system benchmark')
    res['script_sha256'] = sha256(__file__)
    res['contract_sha256'] = sha256(os.path.join(os.path.dirname(__file__), 'export_contract.py'))
    with open(a.out, 'x') as f:
        json.dump(strict_json(res), f, indent=1, allow_nan=False)
    print(json.dumps({k: res[k] for k in ('splits', 'independence', 'detection_test')}, default=float)[:1500])
    print('native', json.dumps({k: (v['object_coverage'], v['frame_uniform_class_macro']['complete_dictionary_macro']) for k, v in out_native.items()}))
    print('protocol', json.dumps({k: (v['frame_uniform_class_macro']['complete_dictionary_macro'], v['object_coverage']) for k, v in out_prot.items()}))


if __name__ == '__main__':
    main()

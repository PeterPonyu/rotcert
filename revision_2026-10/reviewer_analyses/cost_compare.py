#!/usr/bin/env python3
"""Same-scope cost comparison of post-hoc scores for R3-5 (2026-10-05). One DIOR-R cell (O-RCNN seed 0, frozen
cache), archived fixed split r=0, alpha=0.10, one CPU thread, warm cached detections; per stage: scoring all matched
pairs, then class-wise source-HCP calibration on the calibration role plus coverage evaluation on the evaluation role.
Arms: GWD (vectorized, the analysis implementation), doubled-angle coordinates (the package's per-pair residuals and an
equivalent vectorized form, checked equal on every pair; six coordinates at alpha/6 each), and the released EAV
expansion margin (the authors' per-pair routine). Excludes decoding, inference, matching and serialization. Read-only on
frozen trees; writes cost/COST-COMPARE.json."""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import sys
sys.dont_write_bytecode = True
import hashlib, json, time
from pathlib import Path
import numpy as np

R = Path(os.environ.get("ROTCERT_ROOT", Path(__file__).resolve().parents[2]))
S1 = R / 'revision_2026-10/execution/s1'
GEO = Path(os.environ.get('EAV_DETR_ROOT', 'EAV-DETR')) / 'tools'
sys.path.insert(0, str(S1)); sys.path.insert(0, str(GEO)); sys.path.insert(0, str(R))
from core import hcp_sorted, original_roles, CAL, EVAL        # noqa: E402
from run_s1 import CELLS                                        # noqa: E402
from vendor.gwd import obb_gwd                                  # noqa: E402
import geometry_utils as geo                                    # noqa: E402
from rotcert.scores import _doubled_angle_residuals, canonicalize_le90   # noqa: E402

OUT = Path(__file__).resolve().parent / 'cost' / 'COST-COMPARE.json'
REPS = 5
COORDS = ('cx', 'cy', 'w', 'h', 'cos2t', 'sin2t')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def doubled_vectorized(P, G):
    def canon(b):
        w, h, t = b[:, 2].copy(), b[:, 3].copy(), b[:, 4].copy()
        swap = w < h
        w[swap], h[swap] = b[swap, 3], b[swap, 2]
        t = np.where(swap, t + np.pi / 2, t)
        t = (t + np.pi / 2) % np.pi - np.pi / 2
        return w, h, t
    wp, hp, tp = canon(P); wg, hg, tg = canon(G)
    return np.stack([np.abs(P[:, 0] - G[:, 0]), np.abs(P[:, 1] - G[:, 1]), np.abs(wp - wg), np.abs(hp - hg),
                     np.abs(np.cos(2 * tp) - np.cos(2 * tg)), np.abs(np.sin(2 * tp) - np.sin(2 * tg))], axis=1)


def doubled_loop(P, G):
    return np.array([[r[c] for c in COORDS] for r in (_doubled_angle_residuals(p, g) for p, g in zip(P, G))])


def margin_loop(P, G):
    return np.array([geo.calculate_expansion_margin(geo.convert_to_corners(p), geo.convert_to_corners(g)) for p, g in zip(P, G)])


def calibrate_evaluate(scores, scene, cls, roles, alpha):
    """Class-wise source-HCP on the calibration role, source-uniform coverage on the evaluation role (all coordinates
    jointly for a vector score, each at alpha/d)."""
    S = scores if scores.ndim == 2 else scores[:, None]
    d = S.shape[1]
    ns = int(scene.max()) + 1
    cov = []
    for k in np.unique(cls):
        m = cls == k
        sc, sk = scene[m], S[m]
        cal = roles[sc] == CAL
        ev = roles[sc] == EVAL
        if not cal.any() or not ev.any():
            continue
        ok = np.ones(ev.sum(), dtype=bool)
        for j in range(d):
            order = np.argsort(sk[cal, j], kind='stable')
            s, src = sk[cal, j][order], sc[cal][order]
            sizes = np.bincount(src, minlength=ns)
            q = hcp_sorted(s, src, sizes, (sizes > 0).astype(np.int64)) if alpha / d == 0.1 else hcp_alpha(s, src, sizes, alpha / d)
            ok &= sk[ev, j] <= q
        e_src = sc[ev]
        per = np.bincount(e_src, weights=ok.astype(float), minlength=ns) / np.maximum(np.bincount(e_src, minlength=ns), 1)
        cov.append(per[np.unique(e_src)].mean())
    return float(np.mean(cov))


def hcp_alpha(s, src, sizes, a):
    """Source-uniform HCP quantile at a general level (floating point; used only for the alpha/6 coordinates)."""
    present = sizes > 0
    k = int(present.sum())
    w = 1.0 / ((k + 1) * sizes[src])
    cum = np.cumsum(w)
    idx = np.searchsorted(cum, (1 - a) - 1e-12)
    return float(s[idx]) if idx < len(s) else np.inf


def timed(fn, *args):
    ts, out = [], None
    for _ in range(REPS):
        t = time.perf_counter(); out = fn(*args); ts.append(time.perf_counter() - t)
    return out, dict(median_s=float(np.median(ts)), min_s=float(np.min(ts)), max_s=float(np.max(ts)), reps=REPS)


def main():
    cell = next(c for c in CELLS if c['name'] == 'dior-orcnn-s0')
    assert sha(cell['cache']) == cell['cache_sha256']
    z = np.load(cell['cache'], allow_pickle=False)
    P, G, scene, cls = z['tp_pred'], z['tp_gt'], z['tp_scene'], z['tp_cls']
    universe = np.unique(scene)
    roles = original_roles(z['scene_names'], universe, 0)
    res = {}
    g, t_g = timed(obb_gwd, P, G)
    dv, t_dv = timed(doubled_vectorized, P, G)
    dl, t_dl = timed(doubled_loop, P, G)
    assert np.allclose(dv, dl, rtol=0, atol=1e-9), 'vectorized doubled-angle residuals differ from the package'
    mg, t_m = timed(margin_loop, P.astype(np.float32), G.astype(np.float32))
    for name, sc, t_score, impl in (('gwd', g, t_g, 'vectorized NumPy (analysis implementation)'),
                                    ('doubled_angle', dv, t_dv, 'vectorized NumPy, equal to the package per-pair residuals'),
                                    ('eav_margin', mg, t_m, 'released per-pair Python routine (authors)')):
        cov, t_ce = timed(calibrate_evaluate, sc, scene, cls, roles, 0.10)
        res[name] = dict(implementation=impl, score=t_score, calibrate_evaluate=t_ce, eval_coverage_class_macro=cov)
    res['doubled_angle']['score_package_per_pair'] = t_dl
    out = dict(schema='rotcert.cost-compare.v1', cell='dior-orcnn-s0', pairs=int(len(P)), sources=int(len(universe)),
               split='archived fixed split r=0', alpha=0.10, threads=1, arms=res, host=dict(machine=os.uname().machine),
               scope='warm cached detections; one CPU thread; scoring and class-wise source-HCP calibration plus evaluation; '
                     'excludes decoding, inference, matching and serialization; implementation-dependent',
               inputs={cell['cache']: cell['cache_sha256'], str(GEO / 'geometry_utils.py'): sha(GEO / 'geometry_utils.py'),
                       str(R / 'rotcert/scores.py'): sha(R / 'rotcert/scores.py')},
               script_sha256=sha(__file__))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + '\n')
    for k, v in res.items():
        print(k, 'score %.4f s' % v['score']['median_s'], 'cal+eval %.4f s' % v['calibrate_evaluate']['median_s'],
              'coverage %.3f' % (100 * v['eval_coverage_class_macro']))
    print('doubled per-pair package %.3f s' % t_dl['median_s'])


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Pre-specified synthesis of the validation-split check (DESIGN-FROZEN.md sections 5-6; AMENDMENT-01 section 2).
Read-only on results; the output is strict JSON (non-finite values become null with an explicit flag).

Per cell: pooled and HCP class-macro coverage (mean of the 20 source splits), HCP minus pooled with the verifier's
99.1667% interval, percentile intervals of the HCP class-macro coverage from the saved primary draws at 99.1667% and
95% (the verifier's own interval function), and the fixed-role sensitivity. Readings:
  frozen rule      CHECK_FAILED if the 99.1667% HCP interval lies entirely below 0.90, otherwise
                   NO_UNDERCOVERAGE_DETECTED (absence of detected under-coverage, not a verification of the guarantee);
  amendment rule   WITHIN_1PP_SUPPORTED if that interval's lower end is at least 0.89, INCONCLUSIVE otherwise
                   (reported only when the frozen rule does not fail);
  MISSING          for a family without a completed job; UNAVAILABLE_MISSING_DRAWS if required draws are missing.

  synthesize_val.py --package var/diorval-heldout-20261004/s1 --cells var/diorval-heldout-20261004/CELLS.json \
                   --out var/diorval-heldout-20261004/VAL-RESULTS.json
  synthesize_val.py --package <frozen S1 dir> --dry-run    (test-split cells; code check only, writes nothing)
"""
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

LEVEL = 1 - 0.05 / 6
NOMINAL, MARGIN = 0.90, 0.01


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(x):
    return float(x) if x is not None and math.isfinite(float(x)) else None


def verifier(package):
    sys.path.insert(0, str(package))          # the verifier imports vendor.gwd from its own package
    spec = importlib.util.spec_from_file_location('verify_s1_pkg', package / 'verify_s1.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def draws(folder):
    cp = json.loads((folder / 'checkpoint.json').read_text())
    parts = []
    for chunk in cp['chunks']:
        assert sha(folder / chunk['file']) == chunk['sha256'], chunk['file']
        with np.load(folder / chunk['file'], allow_pickle=False) as z:
            parts.append(z['coverage'])
    cover = np.concatenate(parts)               # draws x inner splits x (pooled, HCP) x classes; NaN = class absent
    return np.mean(np.mean(cover, axis=1), axis=2)   # strict: a missing class makes that draw's macro NaN


def point(folder):
    with np.load(folder / 'point-estimate.npz', allow_pickle=False) as z:
        per_class = np.mean(z['metrics'][:, :, :, 0], axis=0)
    return per_class.mean(axis=1)               # NaN if a class is missing in the point estimate


def readings(interval):
    if interval is None:
        return 'UNAVAILABLE_MISSING_DRAWS', None
    low, high = interval
    if high < NOMINAL:
        return 'CHECK_FAILED', None
    return 'NO_UNDERCOVERAGE_DETECTED', ('WITHIN_1PP_SUPPORTED' if low >= NOMINAL - MARGIN else 'INCONCLUSIVE')


def mc_sensitivity(result):
    """Amendment 2: is a label sensitive to Monte Carlo error of the tail quantiles? Labels are never changed; the
    95% order-statistic brackets of the two endpoints and the five pre-specified 1,000-draw blocks are reported."""
    if result.get('interval') is None:
        return None
    (lo_b, hi_b) = [q['bounds'] for q in result['quantile_mc']]
    straddles = lambda b, t: b[0] is not None and b[1] is not None and b[0] < t <= b[1]
    blocks = result.get('five_block_intervals') or []
    return dict(lower_end_bracket=lo_b, upper_end_bracket=hi_b,
                frozen_rule_mc_sensitive=straddles(hi_b, NOMINAL),
                amendment_rule_mc_sensitive=straddles(lo_b, NOMINAL - MARGIN),
                five_block_labels=[list(readings(b)) for b in blocks],
                five_blocks_agree=len({tuple(readings(b)) for b in blocks}) <= 1 if blocks else None)


def load_detection_counts(path, manifest):
    """Load and bind TP/FP/FN counts to the exact S1 cell manifest.

    Requiring the command-line argument alone is not enough: an empty or stale
    ``CELLS.json`` would otherwise be accepted and formal output would contain
    ``detection_counts=None``.  The cell builder is the producer of this file,
    so the formal synthesizer checks its schema, names, missing-family set and
    integer counts before any result is written.
    """
    if sha(path) != manifest.get('cells_sha256'):
        raise SystemExit('CELLS.json hash does not match EXECUTION-MANIFEST.json')
    doc = json.loads(Path(path).read_text())
    if not isinstance(doc, dict) or doc.get('schema') != 'rotcert.diorval-cells.v2':
        raise SystemExit('CELLS.json has an unexpected schema')
    if not doc.get('design_sha256') or doc['design_sha256'] != manifest.get('design_sha256'):
        raise SystemExit('CELLS.json design does not match EXECUTION-MANIFEST.json')
    entries = doc.get('cells')
    if not isinstance(entries, list):
        raise SystemExit('CELLS.json has no cells list')
    names = [entry.get('cell') if isinstance(entry, dict) else None for entry in entries]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise SystemExit('CELLS.json has missing or duplicate cell names')
    expected_entries = {cell['name']: cell for cell in manifest['cells']}
    expected = list(expected_entries)
    if set(names) != set(expected) or len(names) != len(expected):
        raise SystemExit('CELLS.json cell set does not match EXECUTION-MANIFEST.json: expected %s, got %s' %
                         (sorted(expected), sorted(names)))
    actual_missing = doc.get('missing_families', {})
    if not isinstance(actual_missing, dict) or actual_missing != manifest.get('missing_families', {}):
        raise SystemExit('CELLS.json missing-family record does not match EXECUTION-MANIFEST.json')

    counts = {}
    for entry in entries:
        expected_entry = expected_entries[entry['cell']]
        if (not entry.get('cache_sha256') or entry['cache_sha256'] != expected_entry.get('cache_sha256')
                or not entry.get('matched_sha256')
                or entry['matched_sha256'] != expected_entry.get('metadata', {}).get('sha256')):
            raise SystemExit('CELLS.json cache or matching identity differs for %s' % entry['cell'])
        raw = entry.get('counts')
        if not isinstance(raw, dict) or any(k not in raw for k in ('tp', 'fp', 'fn')):
            raise SystemExit('CELLS.json has incomplete TP/FP/FN counts for %s' % entry['cell'])
        clean = {}
        for key in ('tp', 'fp', 'fn'):
            value = raw[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SystemExit('CELLS.json has invalid %s count for %s' % (key, entry['cell']))
            clean[key] = value
        gt_images = entry.get('validation_images_with_ground_truth')
        if type(gt_images) is not int or not 0 <= gt_images <= doc.get('validation_images', -1):
            raise SystemExit('CELLS.json has invalid validation-image support for %s' % entry['cell'])
        clean['val_images_with_gt'] = gt_images
        counts[entry['cell']] = clean
    return counts


def summarise(v, package, kind, name):
    folder = package / 'results' / kind / name
    verified = json.loads((folder / 'verified.json').read_text())
    assert verified['status'] == 'VERIFIED', (name, kind)
    macro = draws(folder)
    pt = point(folder)
    hcp99 = v.interval(macro[:, 1], LEVEL, blocks=True)
    hcp95 = v.interval(macro[:, 1], .95, blocks=False)
    out: dict = dict(pooled=finite(pt[0]), hcp=finite(pt[1]), difference=finite(pt[1] - pt[0]),
               point_complete=bool(np.isfinite(pt).all()),
               difference_interval=(verified.get('macro_delta_interval') or {}).get('interval'),
               hcp_interval_99_1667=hcp99['interval'], hcp_interval_95=hcp95['interval'],
               missing_draws=int(np.isnan(macro[:, 1]).sum()), draws=int(len(macro)),
               tp_bearing_sources=verified['unique_support']['original_TP_sources'],
               per_class=[dict(cls=c['class'], cal_unique_min=c['cal_unique_min'], eval_unique_min=c['eval_unique_min'],
                               missing_inner_evaluations=c['missing_inner_evaluations'],
                               infinite_threshold_fraction_pooled_hcp=c['infinite_threshold_fraction'],
                               coverage_intervals95_marginal_pooled_hcp=[(iv or {}).get('interval') for iv in
                                                                          c.get('coverage_intervals95_marginal', [])])
                          for c in verified['per_class']])
    if kind == 'primary':
        out['reading_frozen_rule'], out['reading_amendment_rule'] = readings(hcp99['interval'])
        out['tail_monte_carlo'] = mc_sensitivity(hcp99)
    return out


def fmt(x):
    return 'NA' if x is None else '%.3f' % (100 * x)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--cells', type=Path, help='CELLS.json of prepare_val_cells.py, for TP/FP/FN counts')
    a = p.parse_args()
    if not a.dry_run and a.cells is None:
        p.error('--cells is required for the formal synthesis: amendment 2 reports matched TP, FP and FN counts')
    if not a.dry_run and a.out is None:
        p.error('--out is required for the formal synthesis')
    manifest = json.loads((a.package / 'EXECUTION-MANIFEST.json').read_text())
    counts = {}
    if a.cells:
        counts = load_detection_counts(a.cells, manifest)
    v = verifier(a.package)
    rows = []
    for cell in manifest['cells']:
        name = cell['name']
        if a.dry_run and not name.startswith('dior-'):
            continue
        rows.append(dict(cell=name, primary=summarise(v, a.package, 'primary', name),
                         fixed_role=summarise(v, a.package, 'fixed-role', name), detection_counts=counts.get(name)))
    for job, reason in manifest.get('missing_families', {}).items():
        rows.append(dict(cell=job, missing=reason, reading_frozen_rule='MISSING'))
    for r in rows:
        if 'missing' in r:
            print('%-20s MISSING (%s)' % (r['cell'], r['missing']))
            continue
        q = r['primary']
        print('%-20s pooled %s  HCP %s  diff %s %s  HCP99 %s  reading %s / %s' % (
            r['cell'], fmt(q['pooled']), fmt(q['hcp']), fmt(q['difference']),
            [round(100 * x, 3) for x in q['difference_interval']] if q['difference_interval'] else None,
            [round(100 * x, 3) for x in q['hcp_interval_99_1667']] if q['hcp_interval_99_1667'] else None,
            q['reading_frozen_rule'], q['reading_amendment_rule']))
        mc = q['tail_monte_carlo']
        if mc:
            print('%-20s tail MC brackets: lower %s upper %s; sensitive frozen=%s amendment=%s; blocks agree=%s' % (
                '', [round(100 * x, 3) for x in mc['lower_end_bracket']], [round(100 * x, 3) for x in mc['upper_end_bracket']],
                mc['frozen_rule_mc_sensitive'], mc['amendment_rule_mc_sensitive'], mc['five_blocks_agree']))
    if a.dry_run:
        return
    out = dict(schema='rotcert.diorval-results.v2', design_sha256=manifest['design_sha256'],
               manifest_sha256=sha(a.package / 'EXECUTION-MANIFEST.json'), cells_sha256=sha(a.cells), level=LEVEL, cells=rows,
               reading_rules=dict(frozen='CHECK_FAILED if the 99.1667% HCP class-macro interval lies entirely below 0.90; '
                                         'otherwise NO_UNDERCOVERAGE_DETECTED (not a verification)',
                                  amendment='WITHIN_1PP_SUPPORTED if its lower end is at least 0.89, else INCONCLUSIVE'),
               script_sha256=sha(__file__))
    a.out.write_text(json.dumps(out, indent=1, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()

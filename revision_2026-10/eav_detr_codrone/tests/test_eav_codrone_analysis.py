"""Fixed-dictionary summaries and the formal analyzer's CPU-only output paths.

Matching/scoring are stubbed only in main() integration tests: no GPU or external
training process is touched. Numerical HCP primitives have separate export tests.
"""
import json
import math
from pathlib import Path
import sys
import types

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import analyze_eav as analysis
from export_contract import CLASSES, strict_json


def data(classes=range(12), frames=60, per=2, rare=None):
    rng = np.random.default_rng(0)
    f, c, v = [], [], []
    for fr in range(frames):
        for k in classes:
            for _ in range(per):
                f.append(fr)
                c.append(k)
                v.append(rng.random() < .9)
    if rare is not None:
        f.append(3)
        c.append(rare)
        v.append(True)
    return np.array(v, bool), np.array(f), np.array(c, int)


def test_dictionary_size_is_twelve():
    assert len(CLASSES) == 12


def test_full_support():
    v, f, c = data()
    s = analysis.macro_summary(v, f, c)
    assert s['complete_dictionary_status'] == 'AVAILABLE'
    assert s['n_supported'] == s['n_dictionary'] == 12
    assert s['complete_dictionary_macro'] == s['supported_class_macro']
    b = analysis.boot_macro(v, f, c, np.random.default_rng(1))
    assert b['complete_dictionary']['status'] == 'AVAILABLE_APPROXIMATE_PERCENTILE'


def test_missing_evaluation_class_is_na_not_dropped():
    v, f, c = data(range(11))
    s = analysis.macro_summary(v, f, c)
    assert s['complete_dictionary_macro'] is None
    assert s['missing_evaluation_classes'] == [CLASSES[11]]
    assert s['per_class'][CLASSES[11]] is None and s['n_supported'] == 11
    assert math.isnan(analysis.frame_uniform(v, f, c))
    b = analysis.boot_macro(v, f, c, np.random.default_rng(1))
    assert b['complete_dictionary']['undefined_draws'] == analysis.B
    assert b['complete_dictionary']['interval'] is None
    assert b['supported_classes']['status'] == 'AVAILABLE_APPROXIMATE_PERCENTILE'


def test_rare_class_absent_from_resamples_is_counted():
    v, f, c = data(range(11), rare=11)
    b = analysis.boot_macro(v, f, c, np.random.default_rng(2))
    expected = analysis.B * (1 - 1 / 60) ** 60
    assert abs(b['complete_dictionary']['undefined_draws'] - expected) < 60
    assert b['complete_dictionary']['interval'] is None
    assert b['supported_classes']['undefined_draws'] == b['complete_dictionary']['undefined_draws']


def test_infinite_threshold_is_vacuous_not_missing_evaluation():
    v, f, c = data()
    q = {k: math.inf if k == 5 else 1. for k in range(12)}
    s = analysis.macro_summary(v, f, c, q)
    assert s['vacuous_infinite_threshold_classes'] == [CLASSES[5]]
    assert s['complete_dictionary_status'] == 'AVAILABLE'


def test_empty_input_still_has_twelve_na_entries():
    v, f, c = data([], frames=0)
    s = analysis.macro_summary(v, f, c)
    assert s['complete_dictionary_macro'] is s['supported_class_macro'] is None
    assert s['n_supported'] == 0 and s['missing_evaluation_classes'] == list(CLASSES)
    assert list(s['per_class'].values()) == [None] * 12
    assert analysis.boot_macro(v, f, c, np.random.default_rng(0)) is None


def test_generator_stream_for_later_object_intervals_is_unchanged():
    v, f, c = data()
    before, after = np.random.default_rng(7), np.random.default_rng(7)
    frames = np.unique(f)
    # This is the exact random operation made by the old macro=True path.
    before.multinomial(len(frames), np.full(len(frames), 1 / len(frames)), size=analysis.B)
    analysis.boot_macro(v, f, c, after)
    assert before.bit_generator.state == after.bit_generator.state
    assert analysis.boot(v, f, c, before) == analysis.boot(v, f, c, after)


def test_legacy_macro_bootstrap_cannot_silently_drop_classes():
    v, f, c = data(range(11))
    with pytest.raises(ValueError, match='use boot_macro'):
        analysis.boot(v, f, c, np.random.default_rng(0), macro=True)


def test_class_macro_is_source_uniform_not_object_pooled():
    v = np.array([True] * 100 + [False])
    f = np.array(['dense'] * 100 + ['sparse'])
    c = np.zeros(101, int)
    s = analysis.macro_summary(v, f, c)
    assert s['per_class'][CLASSES[0]] == .5
    assert s['supported_class_macro'] == .5
    assert s['complete_dictionary_macro'] is None
    assert v.mean() > .99


def test_strict_json_keeps_na_distinct_from_infinity():
    v, f, c = data(range(11))
    s = analysis.macro_summary(v, f, c, {k: math.inf for k in range(12)})
    out = json.loads(json.dumps(strict_json({'macro': s, 'threshold': math.inf}), allow_nan=False))
    assert out['macro']['per_class'][CLASSES[11]] is None
    assert '/macro/per_class/' + CLASSES[11] not in out['nonfinite_values']
    assert out['nonfinite_values']['/threshold'] == '+Infinity'


@pytest.fixture
def synthetic_main(tmp_path, monkeypatch):
    repo, s1 = tmp_path / 'repo', tmp_path / 's1'
    for path in (s1 / 'core.py', s1 / 'vendor/gwd.py', repo / 'tools/geometry_utils.py',
                 repo / 'src/zoo/eavdetr/oriented_ops.py'):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# CPU integration fixture; not an implementation\n')
    core = types.ModuleType('core')
    core.hcp_sorted = lambda score, frame, sizes, weights: float(np.max(score)) if weights.sum() >= 9 else math.inf
    core.angle_halfwidth = lambda pw, ph, q: np.full(len(pw), np.pi / 2 if math.isinf(q) else .1)
    gwd = types.ModuleType('vendor.gwd')
    gwd.obb_gwd = lambda p, g: np.zeros(len(p))
    ops = types.ModuleType('src.zoo.eavdetr.oriented_ops')
    ops.oriented_box_iou = None
    for name, module in [('core', core), ('vendor.gwd', gwd), ('src.zoo.eavdetr.oriented_ops', ops)]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(analysis, 'margins', lambda split, pairs, repo, procs: np.zeros(len(pairs)))

    def split(classes):
        _, frames, labels = data(classes, frames=12, per=1)
        n = len(labels)
        prob = np.full((n, 12), .001)
        if n:
            prob[np.arange(n), labels] = .8
        names = [f'site_day_40m_30c_frame_{fr}__1__0___0' for fr in range(12)]
        return dict(names=names, gt_box=np.tile([.5, .5, .2, .1, .0], (n, 1)),
                    pred_box=np.tile([.5, .5, .2, .1, .0], (n, 1)),
                    gt_label=labels, gt_tile=frames, pred_tile=frames, pred_prob=prob,
                    checkpoint_sha256='a' * 64, config_sha256='b' * 64, min_score=.05,
                    receipts={}, files={}, complete=True, split_tiles=12, shards=[0], num_shards=1)

    def run(val_classes=range(12), test_classes=range(12), no_pairs=False):
        splits = {'val': split(val_classes), 'test': split(test_classes)}
        monkeypatch.setattr(analysis, 'load_split', lambda exports, name, allow_partial: splits[name])

        def pairs(d):
            ix = np.arange(0 if no_pairs else len(d['gt_box']))
            return np.column_stack((ix, ix))

        monkeypatch.setattr(analysis, 'native_match', lambda d, iou: pairs(d))
        monkeypatch.setattr(analysis, 'greedy_match', lambda d: (pairs(d), np.zeros(12, int)))
        out = tmp_path / 'analysis.json'
        monkeypatch.setattr(sys, 'argv', ['analyze_eav.py', '--exports', 'unused', '--eav-repo', str(repo),
                                         '--s1', str(s1), '--out', str(out), '--procs', '1'])
        analysis.main()
        return json.loads(out.read_text(), parse_constant=lambda v: pytest.fail(f'invalid JSON {v}'))

    return run


def test_formal_main_routes_all_six_variants_to_fixed_dictionary(synthetic_main):
    out = synthetic_main()
    assert out['schema'] == 'rotcert.eav-codrone-analysis.v3'
    assert out['class_dictionary'] == list(CLASSES)
    assert out['bootstrap']['native_macro_seed'] == 20261005
    for entry in list(out['native']['variants'].values()) + list(out['protocol'].values()):
        assert entry['frame_uniform_class_macro']['complete_dictionary_macro'] == 1.
        assert entry['frame_uniform_class_macro']['n_dictionary'] == 12
        assert entry['frame_uniform_interval']['complete_dictionary']['undefined_draws'] == 0
    assert out['detection_test']['recall_complete_dictionary_macro'] == 1.
    assert 'recall_class_macro' not in out['detection_test']


def test_formal_main_reports_missing_evaluation_and_recall_denominators(synthetic_main):
    out = synthetic_main(test_classes=range(11))
    for entry in list(out['native']['variants'].values()) + list(out['protocol'].values()):
        macro = entry['frame_uniform_class_macro']
        assert macro['complete_dictionary_macro'] is None
        assert macro['supported_class_macro'] == 1.
        assert macro['missing_evaluation_classes'] == [CLASSES[11]]
    assert out['detection_test']['recall_complete_dictionary_macro'] is None
    assert out['detection_test']['recall_supported_gt_classes'] == list(CLASSES[:11])
    assert out['native']['class_support']['test'][CLASSES[11]]['gt'] == 0
    assert out['protocol']['gwd-hcp']['test_pairs_by_class'][CLASSES[11]] == 0


def test_formal_main_reports_missing_calibration_as_infinite(synthetic_main):
    out = synthetic_main(val_classes=range(11))
    for name in ('gwd-hcp', 'margin-hcp'):
        entry = out['protocol'][name]
        assert entry['calib_sources_by_class'][CLASSES[11]] == 0
        assert entry['q_by_class'][CLASSES[11]] is None
        assert entry['frame_uniform_class_macro']['vacuous_infinite_threshold_classes'] == [CLASSES[11]]
        assert out['nonfinite_values'][f'/protocol/{name}/q_by_class/{CLASSES[11]}'] == '+Infinity'


def test_formal_main_zero_tp_has_explicit_na_not_an_incomplete_entry(synthetic_main):
    out = synthetic_main(no_pairs=True)
    for entry in list(out['native']['variants'].values()) + list(out['protocol'].values()):
        assert entry['frame_uniform_class_macro']['missing_evaluation_classes'] == list(CLASSES)
        assert entry['frame_uniform_class_macro']['n_supported'] == 0
        assert entry['object_coverage'] is None
    assert out['detection_test']['recall_complete_dictionary_macro'] == 0.
    assert out['protocol']['gwd-hcp']['informative_orientation_share'] is None
    assert out['protocol']['gwd-hcp']['normalized_center_radius_median'] is None
    assert '/protocol/gwd-hcp/normalized_center_radius_median' not in out['nonfinite_values']
    assert out['protocol']['gwd-hcp']['finite_threshold_test_pairs'] == 0
    assert out['detection_test']['fp_share'] is None
    assert out['detection_test']['per_class'][CLASSES[0]]['fn'] == 12


def test_formal_main_no_ground_truth_has_undefined_recall(synthetic_main):
    out = synthetic_main(val_classes=[], test_classes=[])
    assert out['detection_test']['recall_pooled'] is None
    assert out['detection_test']['recall_supported_gt_class_macro'] is None
    assert out['detection_test']['recall_complete_dictionary_macro'] is None
    assert out['detection_test']['fp_share'] is None


def test_formal_main_geometry_medians_include_infinite_thresholds(synthetic_main):
    out = synthetic_main(val_classes=[0])
    for name, field in [('gwd-hcp', 'normalized_center_radius'), ('margin-hcp', 'normalized_margin')]:
        entry = out['protocol'][name]
        assert entry['finite_threshold_test_pairs'] == 12
        assert entry['uninformative_test_pairs'] == 132
        assert entry[field + '_median'] is None
        assert out['nonfinite_values'][f'/protocol/{name}/{field}_median'] == '+Infinity'
        assert entry[field + '_finite_threshold_only_median'] == 0.
        assert entry['geometry_summary_scope'].endswith('includes infinite thresholds')


def test_formal_main_all_infinite_geometry_is_not_missing_evaluation(synthetic_main):
    out = synthetic_main(val_classes=[])
    for name, field in [('gwd-hcp', 'normalized_center_radius'), ('margin-hcp', 'normalized_margin')]:
        entry = out['protocol'][name]
        assert entry['test_pairs'] == entry['uninformative_test_pairs'] == 144
        assert entry['finite_threshold_test_pairs'] == 0
        assert out['nonfinite_values'][f'/protocol/{name}/{field}_median'] == '+Infinity'
        assert entry[field + '_finite_threshold_only_median'] is None
        assert f'/protocol/{name}/{field}_finite_threshold_only_median' not in out['nonfinite_values']
    for variant, entry in out['native']['variants'].items():
        assert entry['finite_threshold_test_pairs'] == 0
        assert out['nonfinite_values'][f'/native/variants/{variant}/normalized_margin_median'] == '+Infinity'

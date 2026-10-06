"""Fail-closed export accounting and numerical edge cases without a GPU."""
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import analyze_eav as analysis
from export_contract import CLASSES, load_shards, sha256, strict_json, validate_arrays
import run_stage


def arrays(indices):
    n = len(indices)
    prob = np.full((n, 12), .001, dtype=np.float32)
    prob[:, 0] = .8
    boxes = np.tile([.5, .5, .2, .1, .3], (n, 1)).astype(np.float32)
    return dict(tile_names=np.array([f'frame_{i}__1__0___0' for i in indices], dtype=np.str_),
                tile_index=np.array(indices, dtype=np.int64), gt_tile=np.arange(n, dtype=np.int32),
                gt_box=boxes.copy(), gt_label=np.zeros(n, dtype=np.int64),
                pred_tile=np.arange(n, dtype=np.int32), pred_box=boxes.copy(), pred_prob=prob)


def shard(root, i, count=2, total=6, indices=None, split='val', **changes):
    path = root / f'{split}-shard{i:02d}-of{count:02d}.npz'
    data = arrays(list(range(i, total, count)) if indices is None else indices)
    np.savez_compressed(path, **data)
    rec = dict(schema='rotcert.eav-export.v2', split=split, shard=i, num_shards=count,
               split_tiles=total, tiles=len(data['tile_names']), min_score=.05,
               checkpoint_sha256='a' * 64, config_sha256='b' * 64, out_sha256=sha256(path),
               class_names=list(CLASSES), box_format='normalized-cxcywh-angle-radians')
    rec.update(changes)
    path.with_suffix('.json').write_text(json.dumps(rec))
    return path


def rewrite(path, mutate):
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    mutate(data)
    np.savez_compressed(path, **data)
    rec = json.loads(path.with_suffix('.json').read_text())
    rec['out_sha256'] = sha256(path)
    path.with_suffix('.json').write_text(json.dumps(rec))


def test_full_shard_union_and_local_offsets(tmp_path):
    for i in range(2):
        shard(tmp_path, i)
    result = load_shards(tmp_path, 'val')
    assert result['complete'] is True
    assert result['shards'] == [0, 1]
    assert result['tile_index'].tolist() == [0, 2, 4, 1, 3, 5]
    assert result['gt_tile'].tolist() == list(range(6))
    assert len(result['files']) == 4  # both data and completion receipts are hashed


def test_missing_shard_rejected_but_diagnostic_explicit(tmp_path):
    shard(tmp_path, 0)
    with pytest.raises(ValueError, match='missing val shards'):
        load_shards(tmp_path, 'val')
    assert load_shards(tmp_path, 'val', allow_partial=True)['complete'] is False


def test_truncated_shard_cannot_be_full_result(tmp_path):
    shard(tmp_path, 0, indices=[0])
    shard(tmp_path, 1, indices=[1])
    with pytest.raises(ValueError, match='incomplete val shard'):
        load_shards(tmp_path, 'val')
    assert len(load_shards(tmp_path, 'val', True)['names']) == 2


def test_empty_shards_preserve_shapes(tmp_path):
    for i in range(5):
        shard(tmp_path, i, count=5, total=2)
    result = load_shards(tmp_path, 'val')
    assert result['complete'] and result['gt_box'].shape == (2, 5)
    assert result['pred_prob'].shape == (2, 12)


@pytest.mark.parametrize('changes', [{'checkpoint_sha256': 'c' * 64}, {'config_sha256': 'c' * 64},
                                    {'min_score': .01}, {'split_tiles': 7}])
def test_mixed_inputs_rejected(tmp_path, changes):
    shard(tmp_path, 0)
    shard(tmp_path, 1, **changes)
    with pytest.raises(ValueError, match='mixed checkpoint'):
        load_shards(tmp_path, 'val')


def test_missing_receipt_rejected(tmp_path):
    p = shard(tmp_path, 0)
    p.with_suffix('.json').unlink()
    with pytest.raises(ValueError, match='missing completion receipt'):
        load_shards(tmp_path, 'val', True)


def test_tampered_npz_rejected(tmp_path):
    p = shard(tmp_path, 0)
    with p.open('ab') as f:
        f.write(b'tampered')
    with pytest.raises(ValueError, match='export hash mismatch'):
        load_shards(tmp_path, 'val', True)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), .06])
def test_incompatible_score_floor_rejected(tmp_path, bad):
    shard(tmp_path, 0, min_score=bad)
    with pytest.raises(ValueError, match='min_score'):
        load_shards(tmp_path, 'val', True)


def test_wrong_modulo_ownership_rejected(tmp_path):
    shard(tmp_path, 0, indices=[1, 3, 5])
    with pytest.raises(ValueError, match='modulo shard ownership'):
        load_shards(tmp_path, 'val', True)


def test_duplicate_tile_name_across_shards_rejected(tmp_path):
    first = shard(tmp_path, 0)
    second = shard(tmp_path, 1)
    rewrite(second, lambda d: d['tile_names'].__setitem__(0, 'frame_0__1__0___0'))
    with pytest.raises(ValueError, match='duplicate tile names'):
        load_shards(tmp_path, 'val')


@pytest.mark.parametrize('key', ['gt_box', 'pred_box', 'pred_prob'])
def test_nan_rejected_even_with_updated_receipt_hash(tmp_path, key):
    p = shard(tmp_path, 0)
    rewrite(p, lambda d: d[key].__setitem__((0, 0), np.nan))
    with pytest.raises(ValueError, match='non-finite|NaN'):
        load_shards(tmp_path, 'val', True)


@pytest.mark.parametrize('key, bad', [('gt_tile', 10), ('pred_tile', -1), ('gt_label', 12)])
def test_invalid_indices_rejected(tmp_path, key, bad):
    p = shard(tmp_path, 0)
    rewrite(p, lambda d: d[key].__setitem__(0, bad))
    with pytest.raises(ValueError, match='nonexistent|class index'):
        load_shards(tmp_path, 'val', True)


def test_invalid_shapes_and_nonpositive_sides():
    data = arrays([0])
    data['pred_prob'] = np.zeros((1, 11))
    with pytest.raises(ValueError, match='twelve-class'):
        validate_arrays(data)
    data = arrays([0])
    data['pred_box'][0, 2] = 0
    with pytest.raises(ValueError, match='nonpositive'):
        validate_arrays(data)


def test_strict_json_nulls_have_explicit_infinity_annotations():
    safe = strict_json({'thresholds': {'car': math.inf}, 'coverage': float('nan'),
                        'nested': [np.float64(-math.inf), np.int64(4)]})
    text = json.dumps(safe, allow_nan=False)
    rec = json.loads(text, parse_constant=lambda v: pytest.fail(f'nonstandard JSON token: {v}'))
    assert rec['thresholds']['car'] is None
    assert rec['nonfinite_values']['/thresholds/car'] == '+Infinity'
    assert rec['nonfinite_values']['/coverage'] == 'NaN'
    assert rec['nested'] == [None, 4]


def test_empty_quantiles_bootstrap_and_macro():
    assert math.isinf(analysis.q_linear(np.array([])))
    assert math.isinf(analysis.q_conformal(np.array([])))
    assert analysis.boot(np.array([], bool), np.array([]), np.array([]), np.random.default_rng(1)) is None
    assert math.isnan(analysis.frame_uniform(np.array([], bool), np.array([])))


@pytest.mark.parametrize('n', [1, 8, 9, 10, 69, 89, 999])
def test_exact_finite_sample_quantile_rank(n):
    rank = (9 * (n + 1) + 9) // 10
    q = analysis.q_conformal(np.arange(n))
    assert q == (math.inf if rank > n else rank - 1)


def test_frame_uniform_does_not_weight_dense_frames_more():
    covered = np.array([True] * 100 + [False])
    frames = np.array(['dense'] * 100 + ['sparse'])
    assert analysis.frame_uniform(covered, frames) == .5
    assert covered.mean() > .99


def test_nonempty_bootstrap_is_deterministic_and_bounded():
    cov = np.array([True, False, True, True])
    frames = np.array(['a', 'a', 'b', 'c'])
    cls = np.zeros(4, int)
    first = analysis.boot(cov, frames, cls, np.random.default_rng(2))
    second = analysis.boot(cov, frames, cls, np.random.default_rng(2))
    assert first == second and 0 <= first[0] <= first[1] <= 1


def test_frozen_hcp_has_nonempty_finite_and_empty_infinite_thresholds():
    path = SCRIPTS.parent / 'execution/s1/core.py'
    spec = importlib.util.spec_from_file_location('eav_test_frozen_core', path)
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    q = analysis.hcp_by_class(core, np.zeros(10, int), np.arange(10), np.arange(1., 11.), 10)
    assert q[0] == 10 and all(math.isinf(q[k]) for k in range(1, 12))
    q_small = analysis.hcp_by_class(core, np.zeros(8, int), np.arange(8), np.arange(8.), 8)
    assert all(math.isinf(x) for x in q_small.values())


@pytest.mark.parametrize('script, options', [('export_eav.py', ['--shard', '-1']),
                                           ('export_eav.py', ['--min-score', 'nan']),
                                           ('analyze_eav.py', ['--procs', '0'])])
def test_argument_errors_fail_before_model_loading(tmp_path, script, options):
    args = ['--config', 'unused', '--checkpoint', 'unused', '--split', 'val', '--out', str(tmp_path / 'unused')]
    if script == 'analyze_eav.py':
        args = ['--exports', 'unused', '--eav-repo', 'unused', '--s1', 'unused', '--out', str(tmp_path / 'unused')]
    result = subprocess.run([sys.executable, '-B', str(SCRIPTS / script), *args, *options], capture_output=True, text=True)
    assert result.returncode == 2 and 'error:' in result.stderr


def test_val_test_checkpoint_mismatch_rejected_before_model_loading(tmp_path):
    shard(tmp_path, 0, count=1, split='val')
    shard(tmp_path, 0, count=1, split='test', checkpoint_sha256='c' * 64)
    result = subprocess.run([sys.executable, '-B', str(SCRIPTS / 'analyze_eav.py'), '--exports', str(tmp_path),
                             '--eav-repo', 'unused', '--s1', 'unused', '--out', str(tmp_path / 'analysis.json')],
                            capture_output=True, text=True)
    assert result.returncode != 0 and 'val/test exports disagree' in result.stderr
    assert not (tmp_path / 'analysis.json').exists()


@pytest.mark.parametrize('stage', ['first-epoch', 'final'])
def test_stage_not_ready_does_not_create_outputs(tmp_path, stage):
    result = subprocess.run([sys.executable, '-B', str(SCRIPTS / 'run_stage.py'), stage,
                             '--root', str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 42
    assert json.loads(result.stdout)['ready'] is False
    assert not (tmp_path / 'diagnostics').exists() and not (tmp_path / 'analysis-final-v2').exists()


def test_final_stage_requires_successful_training_exit(tmp_path):
    run = tmp_path / 'runs/eavdetr-codrone-s42'
    run.mkdir(parents=True)
    (run / 'final_epoch_11.pth').write_bytes(b'not used before readiness checks')
    (tmp_path / 'logs').mkdir()
    (tmp_path / 'logs/train-full.meta').write_text('start 2026-10-04T13:16:19Z cfg=test nproc=5 host=test\n')
    result = subprocess.run([sys.executable, '-B', str(SCRIPTS / 'run_stage.py'), 'final', '--root', str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode == 42 and 'has not exited successfully' in result.stdout
    assert not (tmp_path / 'analysis-final-v2').exists()


def test_stage_config_drift_fails_before_copying_checkpoint(tmp_path):
    run = tmp_path / 'runs/eavdetr-codrone-s42'
    run.mkdir(parents=True)
    (run / 'latest.pth').write_bytes(b'unused')
    config = tmp_path / 'src/EAV-DETR/configs/eavdetr/r50vd_codrone_rotcert.yml'
    config.parent.mkdir(parents=True)
    config.write_text('unverified configuration')
    result = subprocess.run([sys.executable, '-B', str(SCRIPTS / 'run_stage.py'), 'first-epoch', '--root', str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode != 0 and 'training config drifted' in result.stderr
    assert not (tmp_path / 'diagnostics').exists()


def test_stage_preflight_is_read_only_with_verified_pins(tmp_path, monkeypatch, capsys):
    run = tmp_path / 'runs/eavdetr-codrone-s42'
    run.mkdir(parents=True)
    (run / 'latest.pth').write_bytes(b'unused')
    monkeypatch.setattr(sys, 'argv', ['run_stage.py', 'first-epoch', '--root', str(tmp_path), '--preflight-only'])
    monkeypatch.setattr(run_stage, 'sha256', lambda p: run_stage.CONFIG_HASH if p.name.endswith('.yml') else run_stage.S1_HASHES[str(p.relative_to(tmp_path / 's1'))])
    assert run_stage.main() == 0
    assert json.loads(capsys.readouterr().out)['ready'] is True
    assert not (tmp_path / 'diagnostics').exists()

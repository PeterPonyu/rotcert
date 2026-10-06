#!/usr/bin/env python3
"""Turn pulled train-only exports into the four validation-split cells (DESIGN-FROZEN.md, sections 2 and 4).

For each job: check the run record and provenance, keep only the validation images (the exports also hold the
training images, whose outputs are discarded here and never matched), match with the frozen accelerated matcher after
its reference-parity check, and build the cache with the frozen builder. The frozen code is imported read-only and
hash-checked; no bytecode is written.

  prepare_val_cells.py parity                 reproduce the frozen dior-orcnn-s0 test cache from its frozen export
  prepare_val_cells.py cells --pulled DIR     build the validation cells from byte-verified pulled job directories
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = Path(os.environ.get("ROTCERT_ROOT", Path(__file__).resolve().parents[2]))
CLOSEOUT = R / 'revision_2026-10/closeout'
WORKTREE = R
OUT = R / 'var/diorval-heldout-20261004'
FROZEN = {CLOSEOUT / 'match_local.py': '0f16cd5a03de49f401cb8efd08c9d4c2b6a45f84867394db21f6cb0403e1361b',
          WORKTREE / 'rotcert/matching.py': '1c78de38a3042d05f402ebb28ec192c14735c700999a455c4bd9cd6dc3834d54',
          WORKTREE / 'revision_2026-10/wpR/build_cache.py': '708f105a9398d6d211b2aa9a5f06ca0237f360de3ef9ccb1fc70da097fbe18a2',
          WORKTREE / 'rotcert/io.py': 'dbb95f1162961afc9b23886a88a00c08b2cf46d1f4891e2c4af324c2e721b4d6'}
LISTS = {'train.txt': '28ea2f250d5cab7c9de7488d8ad6b434d9552c0e3b7ea27f958848a501e5b690',
         'val.txt': '042756fbfb8d29312aefa7c503d62baa5f1410f2bb33c580d40e777ba60e7ecb',
         'trainval.txt': '917340372f784511bbde9e801c76fbb3ca93bde1c3321c9fed22bf8e63618bf0'}
DATA_MANIFEST_SHA = '54f405055ac16ace8f1543b4d4fcd055557ce0c6b1c0823838331677e4e5dccc'
DIOR_CLASSES = sorted(['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 'chimney',
                       'expressway-service-area', 'expressway-toll-station', 'dam', 'golffield', 'groundtrackfield',
                       'harbor', 'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle',
                       'windmill'])
FAMILIES = ('orcnn', 'roit', 'rtmdet', 's2anet')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def frozen_matcher():
    for path, digest in FROZEN.items():
        if sha(path) != digest:
            raise SystemExit('frozen code changed: %s' % path)
    spec = importlib.util.spec_from_file_location('match_local_frozen', CLOSEOUT / 'match_local.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)       # defines functions only; its CLI runs under __main__
    return module


def select_lines(path, keep):
    """Original export lines whose image is in keep (None keeps every line), byte for byte."""
    out = []
    with Path(path).open('rb') as f:
        for line in f:
            if keep is None or json.loads(line)['image_id'] in keep:
                out.append(line)
    return out


def build_cell(ml, name, det_lines, gt_lines, out):
    """Same rows, order and checks as the frozen closeout matcher (match_local.process)."""
    out.mkdir(parents=True, exist_ok=True)
    det_path, gt_path = out / 'detections.jsonl', out / 'ground_truth.jsonl'
    det_path.write_bytes(b''.join(det_lines))
    gt_path.write_bytes(b''.join(gt_lines))
    t = time.monotonic()
    dets, gts = ml.load_group(det_path, 'det'), ml.load_group(gt_path, 'gt')
    images = sorted(set(dets) | set(gts))
    candidates = [i for i in images if dets.get(i) and gts.get(i)]
    picks = sorted(set(candidates[::max(1, len(candidates) // 12)][:12]
                       + sorted(candidates, key=lambda i: abs(len(gts[i]) - 12))[:3]))
    for i in picks:
        ml.assert_equal(dets[i], gts[i])
    counts = {'tp': 0, 'fp': 0, 'fn': 0}
    tmp = out / 'matched.jsonl.partial'
    with tmp.open('w') as f:
        for image in images:
            d, g = dets.get(image, []), gts.get(image, [])
            scene = (d or g)[0]['scene_id']
            assert all(x['scene_id'] == scene for x in d + g)
            match = ml.accelerated(d, g)
            for m in match['matches']:
                di, gi = m['det_index'], m['gt_index']
                row = {'image_id': image, 'scene_id': scene, 'class': d[di]['class'], 'pred_obb': d[di]['obb'],
                       'gt_obb': g[gi]['obb'], 'pred_score': d[di]['score'], 'iou': m['iou'], 'match_type': 'tp'}
                f.write(json.dumps(row) + '\n'); counts['tp'] += 1
            for di in match['unmatched_det_indices']:
                row = {'image_id': image, 'scene_id': scene, 'class': d[di]['class'], 'pred_obb': d[di]['obb'],
                       'gt_obb': None, 'pred_score': d[di]['score'], 'iou': None, 'match_type': 'fp'}
                f.write(json.dumps(row) + '\n'); counts['fp'] += 1
            for gi in match['unmatched_gt_indices']:
                row = {'image_id': image, 'scene_id': scene, 'class': g[gi]['class'], 'pred_obb': None,
                       'gt_obb': g[gi]['obb'], 'pred_score': None, 'iou': None, 'match_type': 'fn'}
                f.write(json.dumps(row) + '\n'); counts['fn'] += 1
    assert counts['tp'] + counts['fp'] == sum(map(len, dets.values()))
    assert counts['tp'] + counts['fn'] == sum(map(len, gts.values()))
    matched = out / 'matched.jsonl'
    tmp.replace(matched)
    cache = out / (name + '.npz')
    ml.build(str(matched), str(cache), scene_rule='image')
    with np.load(cache, allow_pickle=False) as z:
        classes = z['class_names'].tolist()
        scenes = len(z['scene_names'])
    if classes != DIOR_CLASSES:
        raise SystemExit('%s: class dictionary differs from the 20 DIOR-R classes: %s' % (name, classes))
    return dict(cell=name, counts=counts, images=len(images), scenes=scenes, parity_images=len(picks),
                detections_sha256=sha(det_path), ground_truth_sha256=sha(gt_path), matched_sha256=sha(matched),
                cache=str(cache), cache_sha256=sha(cache), class_names=classes, seconds=round(time.monotonic() - t, 1))


def parity(args):
    """End-to-end check before any validation data exist: the frozen test export must give the frozen cache."""
    ml = frozen_matcher()
    src = CLOSEOUT / 'collected-prefix/run/jobs/dior-orcnn-s0'
    ref = CLOSEOUT / 'analysis/dior-orcnn-s0'
    out = args.work / 'parity-dior-orcnn-s0'
    if out.exists():
        raise SystemExit('REFUSE: %s exists' % out)
    rec = build_cell(ml, 'dior-orcnn-s0', select_lines(src / 'detections.jsonl', None),
                     select_lines(src / 'ground_truth.jsonl', None), out)
    same_matched = rec['matched_sha256'] == sha(ref / 'matched.jsonl')
    with np.load(out / 'dior-orcnn-s0.npz', allow_pickle=False) as a, \
            np.load(ref / 'dior-orcnn-s0.npz', allow_pickle=False) as b:
        arrays = {k: bool(np.array_equal(a[k], b[k])) for k in b.files}
        same_keys = sorted(a.files) == sorted(b.files)
    result = dict(schema='rotcert.diorval-parity.v1', matched_identical=same_matched, cache_keys_identical=same_keys,
                  arrays_identical=arrays, all_identical=same_matched and same_keys and all(arrays.values()),
                  frozen_hashes={str(k): v for k, v in FROZEN.items()}, record=rec)
    (args.work / 'PARITY.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: result[k] for k in ('matched_identical', 'cache_keys_identical', 'all_identical')}))
    if not result['all_identical']:
        raise SystemExit('PARITY FAILED')


def read_list(path):
    return [x.strip() for x in Path(path).read_text().splitlines() if x.strip()]


def verify_pull_receipt(pulled, receipt):
    """Rehash the exact producer file set before consuming any pulled data."""
    from transfer import FILES, LISTS as PULL_LISTS, NAMES, PASSED, TERMINAL

    if receipt.get('schema') not in ('rotcert.diorval-pull.v3', 'rotcert.diorval-pull.v4'):
        raise SystemExit('unrecognized final pull schema')
    if receipt.get('final') is not True or receipt.get('all_verified') is not True:
        raise SystemExit('pull is not a byte-verified final receipt')
    passed = receipt.get('families_pulled', [])
    missing = receipt.get('families_missing', {})
    if (not isinstance(passed, list) or not isinstance(missing, dict) or len(set(passed)) != len(passed)
            or set(passed) & set(missing) or set(passed) | set(missing) != set(NAMES)):
        raise SystemExit('pull families are incomplete, duplicated or contradictory')
    states = receipt.get('family_states', {})
    if set(states) != set(NAMES):
        raise SystemExit('pull has no complete family-state record')
    for job in NAMES:
        state = states[job].get('state')
        if state not in TERMINAL or (job in passed) != (state in PASSED):
            raise SystemExit('pull family-state mismatch: ' + job)
    recovered = [job for job in passed if states[job]['state'] == 'PASS_EXIT_UNRECORDED']
    if receipt.get('output_pass_exit_unrecorded', []) != recovered:
        raise SystemExit('pull is missing the unrecorded-exit recovery evidence')
    expected = {pulled / 'lists' / Path(name).name for name in PULL_LISTS}
    expected.update(pulled / job / name for job in passed for name in FILES)
    reruns = receipt.get('reruns', {})
    if not isinstance(reruns, dict) or not set(reruns) <= set(NAMES):
        raise SystemExit('pull has an invalid rerun family mapping')
    for job in NAMES:
        if job in reruns:
            expected.add(pulled / job / 'records/rerun.json')
        if states[job]['state'] == 'FINAL_MISSING':
            expected.add(pulled / job / 'records/final_missing.json')
        if states[job]['state'] == 'PASS':
            expected.add(pulled / job / 'records/exit.json')
    entries = receipt.get('files', [])
    seen = set()
    for record in entries:
        path = Path(record['local'])
        # Optional exit evidence for failed families is accepted, but no arbitrary files.
        permitted = path in expected or any(path == pulled / job / 'records/exit.json' for job in missing)
        if not permitted or path in seen:
            raise SystemExit('unexpected or duplicate pulled file: ' + str(path))
        seen.add(path)
        digest = record.get('remote_sha256')
        if (record.get('verified') is not True or not path.is_file() or sha(path) != digest
                or record.get('local_sha256') != digest or path.stat().st_size != record.get('bytes')):
            raise SystemExit('pulled bytes changed or are unverified: ' + str(path))
    if not expected <= seen:
        raise SystemExit('pull receipt omits required files: ' + str(sorted(map(str, expected - seen))))
    for job, rerun in reruns.items():
        path = pulled / job / 'records/rerun.json'
        actual = json.loads(path.read_text())
        if actual != rerun or actual.get('job') != job or actual.get('mode') not in ('fresh', 'reuse-training'):
            raise SystemExit('rerun receipt identity or mode differs: ' + job)


def cells(args):
    pulled = args.pulled
    receipt = json.loads((pulled / 'PULL-RECEIPT.json').read_text())
    verify_pull_receipt(pulled, receipt)
    ml = frozen_matcher()
    lists = pulled / 'lists'
    for name, digest in LISTS.items():
        assert sha(lists / name) == digest, name
    assert sha(lists / 'data_manifest.json') == DATA_MANIFEST_SHA
    train, val, trainval = (set(read_list(lists / n)) for n in ('train.txt', 'val.txt', 'trainval.txt'))
    assert len(train) == 5862 and len(val) == 5863 and not train & val and trainval == train | val
    kit = json.loads((HERE / 'kit/KIT-MANIFEST.json').read_text())
    design = json.loads((HERE / 'DESIGN-RECEIPT.json').read_text())
    assert sha(HERE / design['design_file']) == design['design_sha256'] == kit['design_sha256']
    gt_sha = json.loads((lists / 'data_manifest.json').read_text())['splits']['trainval']['gt_sha256']
    records, missing = [], dict(receipt.get('families_missing', {}))
    for family in FAMILIES:
        job = 'dior-%s-s0-trainonly' % family
        name = 'diorval-%s-s0' % family
        if job not in receipt['families_pulled']:
            assert job in missing, job
            continue
        src = pulled / job
        run = json.loads((src / 'run.json').read_text())
        prov = json.loads((src / 'detections.jsonl.provenance.json').read_text())
        rerun_path = src / 'records/rerun.json'
        rerun = json.loads(rerun_path.read_text()) if rerun_path.exists() else None
        assert run['status'] == 'PASS' and run['scope'] == 'training_and_frozen_export', job
        assert run['prediction_images'] == 11725 and not prov['smoke_only'], job
        # reused training is allowed only for the single recorded technical rerun in reuse-training mode
        assert not prov['train_reuse'] or (rerun is not None and rerun['mode'] == 'reuse-training'), job
        assert prov['output_sha256'] == sha(src / 'detections.jsonl'), job
        assert prov['data_manifest_sha256'] == DATA_MANIFEST_SHA and prov['seed'] == 0, job
        assert prov['checkpoint_sha256'] == run['checkpoint_sha256'], job
        assert sha(src / 'ground_truth.jsonl') == gt_sha, job
        gt_images = {json.loads(l)['image_id'] for l in select_lines(src / 'ground_truth.jsonl', None)}
        det_images = {json.loads(l)['image_id'] for l in select_lines(src / 'detections.jsonl', None)}
        assert gt_images <= trainval and det_images <= trainval, job
        rec = build_cell(ml, name, select_lines(src / 'detections.jsonl', val),
                         select_lines(src / 'ground_truth.jsonl', val), args.work / 'cells' / name)
        assert int(rec['images']) <= len(val)
        rec.update(job=job, family=family, run_checkpoint_sha256=run['checkpoint_sha256'], technical_rerun=rerun,
                   validation_images_with_ground_truth=len(gt_images & val),
                   validation_images_with_detections=len(det_images & val),
                   discarded_training_images_with_detections=len(det_images & train))
        records.append(rec)
    if not records:
        raise SystemExit('no family completed; report all four as missing')
    out = dict(schema='rotcert.diorval-cells.v2', design_sha256=design['design_sha256'], kit_jobs_sha256=kit['jobs_file_sha256'],
               validation_images=len(val), cells=records, missing_families=missing, script_sha256=sha(__file__),
               pull_receipt_sha256=sha(pulled / 'PULL-RECEIPT.json'),
               output_pass_exit_unrecorded=receipt.get('output_pass_exit_unrecorded', []),
               note='outputs on the 5,862 training images were discarded before matching')
    (args.work / 'CELLS.json').write_text(json.dumps(out, indent=1, allow_nan=False) + '\n')
    print(json.dumps(dict(cells=[{k: r[k] for k in ('cell', 'counts', 'scenes', 'parity_images')} for r in records],
                          missing=missing), indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['parity', 'cells'])
    p.add_argument('--pulled', type=Path, default=OUT / 'pulled')
    p.add_argument('--work', type=Path, default=OUT)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    (parity if a.mode == 'parity' else cells)(a)


if __name__ == '__main__':
    main()

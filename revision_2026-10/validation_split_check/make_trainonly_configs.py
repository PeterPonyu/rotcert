#!/usr/bin/env python3
"""Build the train-only DIOR-R jobs of the frozen validation-split check (DESIGN-FROZEN.md, sections 3 and 9).

Each config is the formal seed-0 job's actually used input config (the hashed inputs/config.py of the finished formal
run) with exactly three changes: the training dataset keeps only its train.txt leaf, the inference dataset reads
trainval.txt from JPEGImages-trainval, and the work directory and prediction dump name the new job. The script proves
that no other field differs, that the written file reads back to the intended dictionary, and that the jobs keep the
formal job fields. It writes only below kit/ and refuses to overwrite.
Usage: make_trainonly_configs.py [--out kit]
"""
import argparse
import copy
import hashlib
import json
import pprint
from pathlib import Path
import os

HERE = Path(__file__).resolve().parent
R = Path(os.environ.get("ROTCERT_ROOT", Path(__file__).resolve().parents[2]))
FORMAL = R / 'revision_2026-10/closeout/collected-prefix'
FORMAL_JOBS = FORMAL / 'toolkit/jobs/jobs.json'
BOX_LANE = '/root/autodl-tmp/diorval-heldout-20261004'
BOX_JOBS = '/root/autodl-tmp/rotcert-planB/run/jobs'
RUNNER = '/root/autodl-tmp/rotcert-planB/toolkit/jobs/run_mm.py'
FAMILIES = ('orcnn', 'roit', 'rtmdet', 's2anet')
# expected wall hours: half of the formal trainval training plus inference on 11,725 images (extrapolated, not measured)
EST_HOURS = {'orcnn': 5.9, 'roit': 3.7, 'rtmdet': 4.5, 's2anet': 2.1}
SAFE_BUILTINS = {'dict': dict, 'list': list, 'tuple': tuple}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(text, name):
    namespace = {}
    exec(compile(text, name, 'exec'), {'__builtins__': SAFE_BUILTINS}, namespace)  # dumped literals and dict() only
    return namespace


def render(namespace, header):
    lines = [header.rstrip('\n')]
    for key, value in namespace.items():
        lines.append('%s = %s' % (key, pprint.pformat(value, width=110, sort_dicts=False)))
    return '\n'.join(lines) + '\n'


def differences(a, b, path=()):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b), key=str):
            if key not in a or key not in b:
                out.append(path + (key,))
            else:
                out += differences(a[key], b[key], path + (key,))
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and type(a) is type(b) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, path + (i,))]
    return [] if a == b else [path]


def transform(formal, new_id):
    cfg = copy.deepcopy(formal)
    train = cfg['train_dataloader']['dataset']
    leaves = train['datasets']
    assert train['type'] == 'ConcatDataset' and [d['ann_file'] for d in leaves] == \
        ['ImageSets/Main/train.txt', 'ImageSets/Main/val.txt'], 'unexpected formal training data'
    assert all(d['data_prefix'] == {'img_path': 'JPEGImages-trainval'} for d in leaves)
    train['datasets'] = [leaves[0]]
    test = cfg['test_dataloader']['dataset']
    assert test['ann_file'] == 'ImageSets/Main/test.txt' and test['data_prefix'] == {'img_path': 'JPEGImages-test'}, \
        'unexpected formal inference data'
    assert test.get('test_mode') is True
    test['ann_file'] = 'ImageSets/Main/trainval.txt'
    test['data_prefix'] = {'img_path': 'JPEGImages-trainval'}
    cfg['work_dir'] = '%s/%s' % (BOX_JOBS, new_id)
    assert cfg['test_evaluator']['type'] == 'DumpResults'
    cfg['test_evaluator'] = dict(cfg['test_evaluator'], out_file_path='%s/%s/predictions.pkl' % (BOX_JOBS, new_id))
    assert cfg.get('val_dataloader') is None and cfg.get('val_cfg') is None, 'formal config evaluates during training'
    expected = {('train_dataloader', 'dataset', 'datasets'), ('test_dataloader', 'dataset', 'ann_file'),
                ('test_dataloader', 'dataset', 'data_prefix', 'img_path'), ('work_dir',),
                ('test_evaluator', 'out_file_path')}
    changed = set(differences(formal, cfg))
    assert changed == expected, sorted(changed ^ expected)
    return cfg, sorted('.'.join(map(str, p)) for p in changed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=HERE / 'kit')
    a = p.parse_args()
    if (a.out / 'jobs.trainonly.json').exists():
        raise SystemExit('REFUSE: kit exists: %s' % a.out)
    design = json.loads((HERE / 'DESIGN-RECEIPT.json').read_text())
    assert sha(HERE / design['design_file']) == design['design_sha256'], 'design changed after freezing'
    (a.out / 'configs').mkdir(parents=True, exist_ok=True)
    formal_jobs = {j['id']: j for j in json.loads(FORMAL_JOBS.read_text())['jobs']}
    jobs, records = [], []
    for family in FAMILIES:
        base_id, new_id = 'dior-%s-s0' % family, 'dior-%s-s0-trainonly' % family
        work = FORMAL / 'run/jobs' / base_id
        run = json.loads((work / 'run.json').read_text())
        formal_path = work / 'inputs/config.py'
        assert run['status'] == 'PASS' and run['scope'] == 'training_and_frozen_export'
        assert sha(formal_path) == run['fingerprint']['config'], 'formal input config is not the hashed one'
        formal = load_config(formal_path.read_text(), str(formal_path))
        cfg, changed = transform(formal, new_id)
        assert cfg['randomness']['seed'] == 0
        header = ('# %s: frozen DIOR-R validation-split check (DESIGN-FROZEN.md sha256 %s).\n'
                  '# Formal %s input config (sha256 %s) with only these changes: %s.\n'
                  % (new_id, design['design_sha256'], base_id, sha(formal_path), ', '.join(changed)))
        path = a.out / 'configs' / (new_id + '.py')
        text = render(cfg, header)
        assert load_config(text, str(path)) == cfg, 'rendered config does not read back'
        path.write_text(text)
        base = formal_jobs[base_id]
        assert base['config'].endswith(base_id + '.py') and base['split'] == 'test' and base['train'] is True
        job = copy.deepcopy(base)
        job.update(id=new_id, config='%s/kit/configs/%s.py' % (BOX_LANE, new_id), split='trainval', train=True,
                   checkpoint=None, work_dir='$PLANB_RUN/jobs/' + new_id, est_hours=EST_HOURS[family],
                   estimate_basis='half of the formal trainval training plus trainval inference; extrapolated, not measured',
                   outputs=[o.replace(base_id, new_id) for o in base['outputs']],
                   cmd={'train': '${PLANB_RUN}/envs/mmrot/bin/python %s --job %s --jobs-file %s/kit/jobs.trainonly.json'
                                 % (RUNNER, new_id, BOX_LANE)},
                   trainonly=dict(base_job=base_id, base_config_sha256=sha(formal_path), changed_fields=changed,
                                  design_sha256=design['design_sha256'], no_smoke_run=True,
                                  inference_outputs_on_training_images='discarded before matching'))
        jobs.append(job)
        records.append(dict(job=new_id, base_job=base_id, base_config_sha256=sha(formal_path), config=path.name,
                            config_sha256=sha(path), changed_fields=changed,
                            max_epochs=cfg['train_cfg']['max_epochs'],
                            train_batch_size=cfg['train_dataloader']['batch_size']))
    jobs_path = a.out / 'jobs.trainonly.json'
    jobs_path.write_text(json.dumps({'jobs': jobs}, indent=1, sort_keys=True) + '\n')
    manifest = dict(schema='rotcert.diorval-kit.v1', design_sha256=design['design_sha256'],
                    jobs_file_sha256=sha(jobs_path), runner=RUNNER, runner_sha256_on_box='e797a8d53f31eb5f2de762104a583354a0861ff3547e6db8564e6ef5635fd0e3',
                    box_lane=BOX_LANE, configs=records, generator_sha256=sha(__file__))
    (a.out / 'KIT-MANIFEST.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps(records, indent=1))


if __name__ == '__main__':
    main()

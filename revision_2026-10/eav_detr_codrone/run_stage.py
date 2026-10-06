#!/usr/bin/env python3
"""Run the predeclared diagnostic/final inference stages without touching training."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from export_contract import load_shards, sha256

CONFIG_HASH = '10ebacb7f37f6679cd982a1cc73b0e3a6d1efc000f366f57848f76ec8c1bcc8e'
S1_HASHES = {'core.py': '3e8cd87d5cee1effbffad0966c96ae5c09701d51a8edb806fbcf4678aac0fe5d',
             'vendor/gwd.py': '4438a38b0773f5ceb095ac092d5474af41b7091b81b8add6becdda23474d52de'}


def write_receipt(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)


def run_commands(jobs, cwd, env):
    """One GPU per export worker; persist individual exit codes and full logs."""
    running = []
    try:
        for command, gpu, log in jobs:
            handle = log.open('x')
            try:
                proc = subprocess.Popen(command, cwd=cwd, env={**env, 'CUDA_VISIBLE_DEVICES': str(gpu)},
                                        stdout=handle, stderr=subprocess.STDOUT)
            except BaseException:
                handle.close()
                raise
            running.append((proc, handle, command, gpu, log))
        result = []
        for proc, handle, command, gpu, log in running:
            code = proc.wait()
            handle.close()
            result.append(dict(command=command, gpu=gpu, log=str(log), exit_code=code))
        if any(r['exit_code'] != 0 for r in result):
            raise RuntimeError(f'stage subprocess failed: {result}')
        return result
    finally:
        # Never touch a training PID; terminate only direct stage children on errors.
        for proc, handle, *_ in running:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
            handle.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=('first-epoch', 'final'))
    ap.add_argument('--root', default='/root/autodl-tmp/eav-codrone-20261004')
    ap.add_argument('--preflight-only', action='store_true', help='check readiness and pins; do not run inference')
    args = ap.parse_args()
    root = Path(args.root).resolve()
    script_dir = Path(__file__).resolve().parent
    repo = root / 'src/EAV-DETR'
    config = repo / 'configs/eavdetr/r50vd_codrone_rotcert.yml'
    run = root / 'runs/eavdetr-codrone-s42'
    checkpoint = run / ('latest.pth' if args.stage == 'first-epoch' else 'final_epoch_11.pth')
    if not checkpoint.is_file():
        print(json.dumps(dict(stage=args.stage, ready=False, reason='checkpoint not yet available')))
        return 42
    if args.stage == 'final':
        meta = (root / 'logs/train-full.meta').read_text()
        if not re.search(r'^end \S+ rc=0$', meta, re.M):
            print(json.dumps(dict(stage=args.stage, ready=False, reason='training has not exited successfully')))
            return 42
    if sha256(config) != CONFIG_HASH:
        raise ValueError('training config drifted from the verified running configuration')
    for file, expected in S1_HASHES.items():
        if sha256(root / 's1' / file) != expected:
            raise ValueError(f'frozen S1 drift: {file}')
    output = root / ('diagnostics/epoch-0-v2' if args.stage == 'first-epoch' else 'analysis-final-v2')
    if args.preflight_only:
        print(json.dumps(dict(stage=args.stage, ready=True, checkpoint=str(checkpoint),
                              output=str(output), already_started=output.exists())))
        return 0
    output.mkdir(parents=True, exist_ok=False)  # fail closed on repeated launch; no existing data overwritten
    try:
        pinned = output / 'checkpoint.pth'
        with checkpoint.open('rb') as source, pinned.open('xb') as destination:
            shutil.copyfileobj(source, destination, length=1 << 22)
        import torch
        sys.path.insert(0, str(repo))
        saved = torch.load(pinned, map_location='cpu')
        expected_epoch = 0 if args.stage == 'first-epoch' else 11
        if saved.get('last_epoch') != expected_epoch:
            raise ValueError(f'expected epoch {expected_epoch}, got {saved.get("last_epoch")}; do not silently substitute a checkpoint')
        del saved
        exports = output / 'exports'
        exports.mkdir()
        logs = output / 'logs'
        logs.mkdir()
        env = {**os.environ, 'TORCH_HOME': str(root / 'torch-home'), 'OMP_NUM_THREADS': '1',
               'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
        python = str(root / 'venv-eav/bin/python')
        receipt = dict(schema='rotcert.eav-stage.v1', stage=args.stage,
                       started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                       diagnostic_only=args.stage == 'first-epoch', checkpoint_epoch=expected_epoch,
                       checkpoint_sha256=sha256(pinned), config_sha256=sha256(config),
                       script_sha256={p.name: sha256(p) for p in script_dir.glob('*.py')},
                       output=str(output), steps=[])
        write_receipt(output / 'STARTED.json', receipt)
        for split in ('val', 'test'):
            jobs = []
            for shard in (range(5) if args.stage == 'final' else (0,)):
                command = [python, '-B', str(script_dir / 'export_eav.py'), '--config', str(config),
                           '--checkpoint', str(pinned), '--expected-epoch', str(expected_epoch), '--split', split,
                           '--shard', str(shard), '--num-shards', '5', '--out', str(exports)]
                if args.stage == 'first-epoch':
                    command += ['--limit', '200']
                jobs.append((command, shard, logs / f'export-{split}-{shard}.log'))
            receipt['steps'] += run_commands(jobs, repo, env)
        parts = {s: load_shards(exports, s, allow_partial=args.stage == 'first-epoch') for s in ('val', 'test')}
        receipt['split_tiles'] = {s: len(d['names']) for s, d in parts.items()}
        if args.stage == 'final' and any(not d['complete'] for d in parts.values()):
            raise ValueError('incomplete final exports')
        result_path = output / 'ANALYSIS.json'
        command = [python, '-B', str(script_dir / 'analyze_eav.py'), '--exports', str(exports),
                   '--eav-repo', str(repo), '--s1', str(root / 's1'), '--out', str(result_path),
                   '--procs', '8' if args.stage == 'final' else '2']
        if args.stage == 'first-epoch':
            command.append('--allow-partial')
        receipt['steps'] += run_commands([(command, 0, logs / 'analysis.log')], root, env)
        result = json.loads(result_path.read_text())
        if args.stage == 'final' and (result['diagnostic_only'] or any(not d['complete'] for d in result['splits'].values())):
            raise ValueError('final analysis is marked diagnostic or incomplete')
        receipt.update(completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                       analysis=str(result_path), analysis_sha256=sha256(result_path))
        write_receipt(output / 'COMPLETED.json', receipt)
        print(json.dumps(receipt))
        return 0
    except BaseException as error:
        write_receipt(output / 'FAILED.json', dict(stage=args.stage, error=repr(error),
                                                  observed_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        raise


if __name__ == '__main__':
    raise SystemExit(main())

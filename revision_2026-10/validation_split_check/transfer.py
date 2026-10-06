#!/usr/bin/env python3
"""Upload the kit to the box and pull the job exports back, byte-verified on both sides.

  transfer.py upload              copy kit/ to the box lane and verify every file hash remotely
  transfer.py sizes               read-only: the state of every family and the size of what a pull would copy
  transfer.py pull --go --interim copy the families that have already PASSED (verified), without a final receipt;
                                  may be repeated while other families run
  transfer.py pull --go           the final pull (amendment 3): only when every family has a terminal disposition --
                                  PASS, its single rerun failed (FAILED_FINAL), or declared missing by the executor
                                  (FINAL_MISSING). Running, interrupted or still-rerunnable families block it, so a
                                  family that has not ended is never frozen as missing. Writes PULL-RECEIPT.json once.
Uses the existing key-based host alias of the EAV lane; never asks for or stores a password.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = Path(os.environ.get("ROTCERT_ROOT", Path(__file__).resolve().parents[2]))
CONFIG = os.environ.get('ROTCERT_SSH_CONFIG', 'connection.ssh-config')
SSH = ['ssh', '-F', CONFIG, 'eavbox']
SCP = ['scp', '-C', '-F', CONFIG]
LANE = '/root/autodl-tmp/diorval-heldout-20261004'
JOBS = '/root/autodl-tmp/rotcert-planB/run/jobs'
DIOR = '/root/autodl-tmp/rotcert-planB/data/dior'
NAMES = ['dior-%s-s0-trainonly' % f for f in ('orcnn', 'roit', 'rtmdet', 's2anet')]
FILES = ['detections.jsonl', 'detections.jsonl.provenance.json', 'ground_truth.jsonl', 'run.json', 'inputs/config.py',
         'train_complete.json']
LISTS = ['ImageSets/Main/train.txt', 'ImageSets/Main/val.txt', 'ImageSets/Main/trainval.txt', 'data_manifest.json']
PULLED = R / 'var/diorval-heldout-20261004/pulled'
MARK = '@@FAMILY@@'
SECTIONS = ('RUN', 'RERUN', 'EXIT', 'MISSING', 'ALIVE', 'LAUNCHED')
PASSED = ('PASS', 'PASS_EXIT_UNRECORDED')
TERMINAL = (*PASSED, 'FAILED_FINAL', 'FINAL_MISSING')
RECORDS = {'exit': '%s/status/%s.exit.json', 'rerun': '%s/status/%s.rerun.json',
           'final_missing': '%s/status/%s.final-missing.json'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def remote(cmd):
    return subprocess.run(SSH + [cmd], check=True, capture_output=True, text=True, timeout=600).stdout


def remote_hashes(paths):
    out = remote('sha256sum ' + ' '.join(paths) + ' && stat -c "%n %s" ' + ' '.join(paths))
    lines = out.strip().splitlines()
    hashes = {l.split()[1]: l.split()[0] for l in lines[:len(paths)]}
    sizes = {l.rsplit(' ', 1)[0]: int(l.rsplit(' ', 1)[1]) for l in lines[len(paths):]}
    return hashes, sizes


def _json(text):
    text = text.strip()
    return json.loads(text) if text else None


def parse_states(text):
    """Raw records per family from the remote listing: run, rerun, exit and final-missing records, the exit code of
    the runner-process query (0 alive, 1 none, other: the query failed) and whether the family was launched."""
    raw = {}
    for block in text.split(MARK)[1:]:
        name, chunk = block.split('\n', 1)
        values = {}
        for i, section in enumerate(SECTIONS):          # sections appear in order; cut each at the next marker
            start = chunk.index('@@%s@@' % section) + len('@@%s@@' % section)
            end = chunk.index('@@%s@@' % SECTIONS[i + 1]) if i + 1 < len(SECTIONS) else len(chunk)
            values[section] = chunk[start:end]
        alive = values['ALIVE'].strip()
        raw[name.strip()] = dict(run=_json(values['RUN']), rerun=_json(values['RERUN']), exit=_json(values['EXIT']),
                                 final_missing=_json(values['MISSING']),
                                 alive_rc=int(alive) if alive.lstrip('-').isdigit() else -1,
                                 launched=values['LAUNCHED'].strip() not in ('', '0'))
    assert sorted(raw) == sorted(NAMES), raw
    return {n: dict(raw[n], **classify(raw[n])) for n in NAMES}


def classify(r):
    """State of one family. Terminal states are PASS, FAILED_FINAL and FINAL_MISSING; every other state blocks the
    final pull, because a family that has not ended must not be frozen as missing."""
    run_pass = bool(r['run']) and r['run'].get('status') == 'PASS'
    exit_rc = None if r['exit'] is None else r['exit'].get('rc')
    if r['alive_rc'] not in (0, 1):
        state, note = 'UNKNOWN', 'the runner-process query failed'
    elif r['alive_rc'] == 0:
        state, note = 'RUNNING', 'a runner process is alive'
    elif not r['launched']:
        state, note = 'NOT_LAUNCHED', 'no launch record'
    elif run_pass and exit_rc == 0:
        state, note = 'PASS', 'run record PASS and exit code 0'
    elif run_pass and r['exit'] is None:
        state, note = 'PASS_EXIT_UNRECORDED', ('output verifier recorded PASS, no runner or wrapper is alive, '
                                               'but wrapper exit was not recorded; retain this missing-exit evidence')
    elif run_pass or exit_rc == 0:
        state, note = 'INCONSISTENT', 'run record and exit record disagree; inspect before any decision'
    elif r['final_missing']:
        state, note = 'FINAL_MISSING', 'declared missing: %s' % r['final_missing'].get('reason')
    elif r['rerun'] and r['exit'] is None:
        state, note = 'FAILED_FINAL', 'the single technical rerun ended without an exit record'
    elif r['rerun']:
        state, note = 'FAILED_FINAL', 'the single technical rerun also failed (exit code %s)' % exit_rc
    elif r['exit'] is None:
        state, note = 'INTERRUPTED', 'no exit record and no live runner: rerun with --interrupted or declare missing'
    else:
        state, note = 'FAILED_RERUNNABLE', 'technical failure (exit code %s); one rerun is allowed' % exit_rc
    return dict(state=state, terminal=state in TERMINAL, note=note)


def family_states():
    parts = []
    for n in NAMES:
        parts.append(
            "echo '{m}{n}'; echo '@@RUN@@'; cat {j}/{n}/run.json 2>/dev/null; echo; echo '@@RERUN@@'; "
            "cat {l}/status/{n}.rerun.json 2>/dev/null; echo; echo '@@EXIT@@'; cat {l}/status/{n}.exit.json 2>/dev/null; "
            "echo; echo '@@MISSING@@'; cat {l}/status/{n}.final-missing.json 2>/dev/null; echo; echo '@@ALIVE@@'; "
            "pgrep -f -- '([r]un_mm[.]py|[r]un_job[.]sh).*{n}' >/dev/null; echo $?; echo '@@LAUNCHED@@'; "
            "grep -c '\"{n}\"' {l}/status/launch.json 2>/dev/null; echo".format(m=MARK, n=n, j=JOBS, l=LANE))
    return parse_states(remote('; '.join(parts) + '; true'))


def plan(states, final):
    """(remote, local) pairs and the families reported as missing; refuses a final pull before every family ended."""
    if final:
        open_ = {n: s['state'] for n, s in states.items() if not s['terminal']}
        if open_:
            raise SystemExit('REFUSE final pull: families without a terminal disposition: %s' % json.dumps(open_))
    passed = [n for n in NAMES if states[n]['state'] in PASSED]
    pairs = [('%s/%s/%s' % (JOBS, n, f), PULLED / n / f) for n in passed for f in FILES]
    for n in NAMES:
        if states[n]['state'] in TERMINAL:
            for key, pattern in RECORDS.items():
                if states[n][key] is not None:
                    pairs.append((pattern % (LANE, n), PULLED / n / 'records' / ('%s.json' % key)))
    pairs += [('%s/%s' % (DIOR, f), PULLED / 'lists' / Path(f).name) for f in LISTS]
    missing = {n: '%s: %s' % (states[n]['state'], states[n]['note']) for n in NAMES
               if states[n]['state'] in ('FAILED_FINAL', 'FINAL_MISSING')}
    return passed, pairs, missing


def upload(_):
    files = sorted(p for p in (HERE / 'kit').rglob('*') if p.is_file())
    remote('test ! -e %s/kit && mkdir -p %s' % (LANE, LANE))
    subprocess.run(SCP + ['-r', str(HERE / 'kit'), 'eavbox:%s/kit' % LANE], check=True, timeout=600)
    paths = ['%s/kit/%s' % (LANE, p.relative_to(HERE / 'kit').as_posix()) for p in files]
    hashes, _ = remote_hashes(paths)
    bad = [p for p, f in zip(paths, files) if hashes.get(p) != sha(f)]
    print(json.dumps({'uploaded': len(files), 'mismatched': bad}))
    if bad:
        sys.exit(1)


def sizes(_):
    states = family_states()
    passed, pairs, missing = plan(states, final=False)
    _, sz = remote_hashes([r for r, _ in pairs])
    print(json.dumps({'families': {n: '%s (%s)' % (s['state'], s['note']) for n, s in states.items()},
                      'final_pull_allowed': all(s['terminal'] for s in states.values()),
                      'files': len(pairs), 'total_MiB': round(sum(sz.values()) / 2 ** 20, 1)}, indent=1))


def copy_verified(pairs):
    hashes, sz = remote_hashes([r for r, _ in pairs])
    records = []
    for rpath, local in pairs:
        local.parent.mkdir(parents=True, exist_ok=True)
        present = local.is_file() and sha(local) == hashes[rpath]
        if not present:
            subprocess.run(SCP + ['eavbox:' + rpath, str(local)], check=True, timeout=3600)
        records.append(dict(remote=rpath, local=str(local), bytes=sz[rpath], remote_sha256=hashes[rpath],
                            local_sha256=sha(local), verified=sha(local) == hashes[rpath], already_present=present))
    return records


def write_receipt(path, receipt):
    """Atomically publish a new receipt without overwriting a previous committed one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(receipt, stream, indent=1, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, path)  # atomic no-clobber commit in the same directory
    finally:
        Path(name).unlink(missing_ok=True)


def pull(args):
    if not args.go:
        raise SystemExit('run "transfer.py sizes" first, then pull with --go')
    final_path = PULLED / 'PULL-RECEIPT.json'
    if final_path.exists():
        previous = json.loads(final_path.read_text())
        if previous.get('all_verified') is not False:
            raise SystemExit('REFUSE: the final pull receipt already exists')
        # Preserve an old failed pull, rather than treating it as a committed success.
        final_path.rename(PULLED / ('FAILED-PULL-legacy-' + uuid.uuid4().hex + '.json'))
    states = family_states()
    passed, pairs, missing = plan(states, final=not args.interim)
    if args.interim and not passed:
        raise SystemExit('no family has passed yet; nothing to pull')
    records = copy_verified(pairs)
    utc = datetime.now(timezone.utc)
    receipt = dict(schema='rotcert.diorval-pull.v4', utc=utc.isoformat(), final=not args.interim, files=records,
                   all_verified=all(r['verified'] for r in records), families_pulled=passed, families_missing=missing,
                   family_states={n: {k: s[k] for k in ('state', 'note')} for n, s in states.items()},
                   reruns={n: states[n]['rerun'] for n in NAMES if states[n]['rerun']},
                   output_pass_exit_unrecorded=[n for n in passed if states[n]['state'] == 'PASS_EXIT_UNRECORDED'])
    if not receipt['all_verified']:
        receipt['final'] = False
        name = 'FAILED-PULL-%s-%s.json' % (utc.strftime('%Y%m%dT%H%M%SZ'), uuid.uuid4().hex)
    else:
        name = 'PULL-RECEIPT.json' if not args.interim else 'INTERIM-PULL-%s-%s.json' % (
            utc.strftime('%Y%m%dT%H%M%SZ'), uuid.uuid4().hex)
    write_receipt(PULLED / name, receipt)
    print(json.dumps({k: receipt[k] for k in ('final', 'all_verified', 'families_pulled', 'families_missing')}))
    if not receipt['all_verified']:
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['upload', 'sizes', 'pull'])
    p.add_argument('--go', action='store_true')
    p.add_argument('--interim', action='store_true', help='pull the passed families only, without a final receipt')
    a = p.parse_args()
    {'upload': upload, 'sizes': sizes, 'pull': pull}[a.mode](a)


if __name__ == '__main__':
    main()

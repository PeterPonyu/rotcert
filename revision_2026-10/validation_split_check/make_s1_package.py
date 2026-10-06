#!/usr/bin/env python3
"""Copy the frozen S1 package unchanged and give it the manifest of the four validation-split cells
(DESIGN-FROZEN.md, section 5). Every copied file must carry its frozen hash; the manifest keeps every protocol field of
the frozen S1 manifest and replaces only the cell list (cell IDs 10-13) and the design reference.
Usage: make_s1_package.py [--cells var/diorval-heldout-20261004/CELLS.json] [--out var/diorval-heldout-20261004/s1]
Then, inside the package: python -B run_s1.py run --workers 3; python -B run_s1.py run --kind fixed-role --workers 3;
python -B verify_s1.py; python -B verify_s1.py --kind fixed-role.
"""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import os

HERE = Path(__file__).resolve().parent
R = Path(os.environ.get("ROTCERT_ROOT", Path(__file__).resolve().parents[2]))
S1 = R / 'revision_2026-10/execution/s1'
OUT = R / 'var/diorval-heldout-20261004'
FILES = {'core.py': '3e8cd87d5cee1effbffad0966c96ae5c09701d51a8edb806fbcf4678aac0fe5d',
         'run_s1.py': '7d67b72c1086a6f3d3e63914eb2408e08e4b5a97c9e9636fb77fb1dc99c85c86',
         'verify_s1.py': '7ac69e1ca0460374f4f0f2a3b7a17f058fa2789c0b432b4f27a40ce470abb42f',
         'vendor/gwd.py': '4438a38b0773f5ceb095ac092d5474af41b7091b81b8add6becdda23474d52de',
         'vendor/splits.py': '44ce45f2db44883f829b540478689dddaf7c65718e6d9766310d80978dfaa541'}
FROZEN_MANIFEST_SHA = '5021bd73690b1d2c7c933d60355ae9ae868b224dfb7657a1a312f9f45bbbbc11'
CELL_IDS = {'diorval-orcnn-s0': 10, 'diorval-roit-s0': 11, 'diorval-rtmdet-s0': 12, 'diorval-s2anet-s0': 13}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cells', type=Path, default=OUT / 'CELLS.json')
    p.add_argument('--out', type=Path, default=OUT / 's1')
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit('REFUSE: %s exists' % a.out)
    design = json.loads((HERE / 'DESIGN-RECEIPT.json').read_text())
    assert sha(HERE / design['design_file']) == design['design_sha256'], 'design changed after freezing'
    assert sha(S1 / 'EXECUTION-MANIFEST.json') == FROZEN_MANIFEST_SHA
    cells = json.loads(a.cells.read_text())
    assert cells['design_sha256'] == design['design_sha256']
    (a.out / 'vendor').mkdir(parents=True)
    for name, digest in FILES.items():
        assert sha(S1 / name) == digest, name
        shutil.copyfile(S1 / name, a.out / name)
        assert sha(a.out / name) == digest
    frozen = json.loads((S1 / 'EXECUTION-MANIFEST.json').read_text())
    entries = []
    for rec in cells['cells']:
        assert sha(rec['cache']) == rec['cache_sha256'], rec['cell']
        entries.append(dict(cell_id=CELL_IDS[rec['cell']], name=rec['cell'], cache=rec['cache'], cache_sha256=rec['cache_sha256'],
                            metadata=dict(cell_id=rec['cell'], path=str(Path(rec['cache']).parent / 'matched.jsonl'),
                                          sha256=rec['matched_sha256'], cache_sha256=rec['cache_sha256'], max_per_img=2000,
                                          scene_rule='image', source_kind='train-only detector, DIOR-R validation images',
                                          protocol='Frozen design DESIGN-FROZEN.md; matching at score >=0.05 and IoU>=0.5; '
                                                   'source independence assumed, not established')))
    names = [e['name'] for e in entries]
    assert names and names == [n for n in CELL_IDS if n in names], names   # subset in fixed order, IDs fixed per family
    manifest = dict(frozen)
    manifest.update(schema='diorval-heldout-S1-execution-v1', utc=datetime.now(timezone.utc).isoformat(),
                    design_sha256=design['design_sha256'], cells=entries, cells_sha256=sha(a.cells),
                    missing_families=cells.get('missing_families', {}),
                    cell_id_mapping='explicit cell_id 10-13 per DESIGN-FROZEN.md; listed order immutable',
                    inherited_from=dict(manifest=str(S1 / 'EXECUTION-MANIFEST.json'), sha256=FROZEN_MANIFEST_SHA,
                                        unchanged_fields=sorted(k for k in frozen if k not in
                                                                ('schema', 'utc', 'design_sha256', 'cells', 'cell_id_mapping'))))
    (a.out / 'EXECUTION-MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'package': str(a.out), 'cells': [e['name'] for e in entries],
                      'manifest_sha256': sha(a.out / 'EXECUTION-MANIFEST.json')}, indent=1))


if __name__ == '__main__':
    main()

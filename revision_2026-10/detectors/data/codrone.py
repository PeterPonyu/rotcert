#!/usr/bin/env python3
"""CODrone location-disjoint preparation. Never publishes DATA_OK.

The public stage/verify interface is deliberately independent of data/stage.py.
It prepares a candidate only; integration/loader review remains mandatory.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
from fractions import Fraction
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import stat
import time
import zipfile

GIB = 1024 ** 3
RESERVE_BYTES = 40 * GIB
CONTROL_ALLOWANCE = 2 * GIB
EXPECTED_ORIGINAL = {'train': 5002, 'val': 2000, 'test': 3002}
CLASSES = ['car', 'truck', 'traffic-sign', 'people', 'motor', 'bicycle',
           'traffic-light', 'tricycle', 'bridge', 'bus', 'boat', 'ship']
SOURCE_REVISION = 'f4343670f0c2692fe395c6da562d00b0c72f71b9'
UPSTREAM = {'repository': 'AHideoKuzeA/CODrone-A-Comprehensive-Oriented-Object-Detection-benchmark-for-UAV',
            'commit': 'eb0668ad56e678c0556cae1cfe1696a1e8dc6253',
            'script': 'CODrone_Split.py',
            'sha256': '9df2b1563c6abb63937f4bbd3bb318ff33b8ea10fceadd8067ff7fd81fd85110'}
RECIPE = {'id': 'codrone-official1024-location-v1', 'upstream': UPSTREAM,
          'size': 1024, 'gap': 512, 'rate': 1, 'imgfraction': .6,
          'window_order': 'x outer, y inner; final start shifted to boundary',
          'window_fallback': 'if none >.6, retain coverage within .01 of maximum',
          'iof': .7, 'iof_comparator': '>=',
          'geometry': 'official float32 source polygon, float32 area denominator max(area,1e-6); no clipping',
          'truncated': 'IOF < 1 sets difficulty 2; otherwise preserve source difficulty',
          'padding': 0, 'image_format': 'PNG', 'encoder': 'cv2.imencode default options',
          'class_order': CLASSES, 'class_aliases': {},
          'ignored': 'exclude explicit non-target sentinel ignored; audit every source row; not a thirteenth class or spatial ignore region',
          'finite_zero_area': 'exclude only exact collinear/coincident source polygons; audit; no HBB fallback or pixel expansion',
          'malformed': 'reject nonfinite, unknown classes, invalid noncollinear or float32-collapsed geometry',
          'difficulty_policy': 'retain all emitted objects including difficulty 2 in loader and canonical GT',
          'loader_contract': {'ai4rs_diff_thr': 100, 'eav_use_difficult': True, 'filter_empty_gt': False},
          'coordinates': 'tile-local; no source-frame merge/NMS; source offsets recorded separately',
          'scene_id': 'exact parsed location; never recording/frame/tile',
          'split': {'algorithm': 'sorted locations; random.Random(0).shuffle; floor(.65*n) clamped [1,n-1]',
                    'seed': 0, 'train_fraction': .65, 'heldout_loader_name': 'test',
                    'test_meaning': 'entire heldout calibration/design/evaluation pool, NOT original official test'}}
# Frozen original archives; filled from the already reviewed source manifest.
ARCHIVES = {'test/annfile.zip': {'bytes': 3069083, 'sha256': '0a2a104153c45a0724dcc8ddd9da3c171ae8e5c867e7554fe89b3715fb13e766'}, 'test/images.zip': {'bytes': 2198703339, 'sha256': 'e192b778764511cba8d9e63598ef78e61d8b17821516c2a6704215d430972862'}, 'test/labels.zip': {'bytes': 8522402, 'sha256': '9a418e2d1e1990c4466478d7459d34cb4e5db6a04166d5e46c6723871c9aa427'}, 'train/annfile.zip': {'bytes': 5029832, 'sha256': '4a9a769e4b6572cba0fcb3bd1ccd2a100caf1fb49d55acfa3199140ef4e383f9'}, 'train/images.zip': {'bytes': 3620367471, 'sha256': '1270202715a7ebcbd3462851f9d1f86a4001956221b6bd8836e2ccba2a5b8534'}, 'train/labels.zip': {'bytes': 13975206, 'sha256': 'be9c0933c42ab7bb9c84bbee911d32152fdb2d54fb1ff940bd051580a737567f'}, 'val/annfile.zip': {'bytes': 2050111, 'sha256': 'd75f3857eccc45045772ec6dedaa975ea768e6bdf614103fb16648ecb5f1c994'}, 'val/images.zip': {'bytes': 1376716283, 'sha256': 'a1b11046bd85e2ef94901f60f1158cd6d996d97da274921d0a4a66d127ec4ef4'}, 'val/labels.zip': {'bytes': 3789943, 'sha256': 'f58049b905c9337679b5ad232c16a79b029a0a30080b69e120a8526d27fb1e85'}}

def json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()

def digest(content):
    return hashlib.sha256(content).hexdigest()

def signature(path):
    s = Path(path).stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

def snapshot(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Expected regular nonsymlink file: ' + str(path))
    before = signature(path)
    content = path.read_bytes()
    if before != signature(path) or len(content) != before[2]:
        raise ValueError('File changed during read: ' + str(path))
    return content

def file_hash(path):
    before = signature(path); h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''): h.update(chunk)
    if before != signature(path): raise ValueError('File changed during hash')
    return h.hexdigest()

def strict_json(content):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out: raise ValueError('Duplicate JSON key: ' + k)
            out[k] = v
        return out
    def bad(value): raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(content, object_pairs_hook=pairs, parse_constant=bad)

def identity(raw_id):
    if not isinstance(raw_id, str) or not raw_id or '/' in raw_id or '\\' in raw_id or '__' in raw_id:
        raise ValueError('Invalid raw frame ID')
    m = re.fullmatch(r'(.+)_frame_([0-9]+)', raw_id)
    if not m: raise ValueError('Unknown CODrone frame identity: ' + raw_id)
    prefix, frame = m.groups(); tokens = [t.strip() for t in prefix.split('_')]
    if any(not t for t in tokens): raise ValueError('Empty identity token')
    take = None
    if tokens[-1].isdigit(): take = int(tokens.pop())
    if len(tokens) < 4 or not re.fullmatch(r'[0-9]+m', tokens[-2]) or not re.fullmatch(r'[0-9]+c', tokens[-1]):
        raise ValueError('Unknown recording identity: ' + raw_id)
    altitude, angle = int(tokens[-2][:-1]), int(tokens[-1][:-1]); head = tokens[:-2]
    lights = {'day', 'night', 'dusk', 'dawn'}
    if head[0] in lights and head[-1] not in lights:
        light, location, layout = head[0], '_'.join(head[1:]), 'light_first'
    elif head[-1] in lights and head[0] not in lights:
        light, location, layout = head[-1], '_'.join(head[:-1]), 'location_first'
    else: raise ValueError('Ambiguous light/location identity: ' + raw_id)
    if not re.fullmatch(r'[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*', location):
        raise ValueError('Unknown location token: ' + location)
    return {'raw_frame_id': raw_id, 'raw_recording_id': prefix, 'location': location,
            'scene_id': location, 'light': light, 'altitude_m': altitude, 'angle_deg': angle,
            'take': take, 'frame_index': int(frame), 'layout': layout,
            'parser_version': 1, 'metadata_token_trimmed': prefix != '_'.join(t.strip() for t in prefix.split('_')),
            'recording_key': [light, location, altitude, angle, take]}

def split_locations(frames):
    ids = [f['raw_frame_id'] for f in frames]
    if len(ids) != len(set(ids)) or not ids: raise ValueError('Duplicate/empty source IDs')
    locations = sorted({f['location'] for f in frames})
    if len(locations) < 2: raise ValueError('Need at least two locations')
    shuffled = locations[:]; random.Random(0).shuffle(shuffled)
    n = max(1, min(len(shuffled) - 1, math.floor(.65 * len(shuffled))))
    train = set(shuffled[:n]); groups = {'train': sorted(train), 'test': sorted(set(locations) - train)}
    parts = {key: sorted(f['raw_frame_id'] for f in frames if f['location'] in set(values))
             for key, values in groups.items()}
    return {'policy': RECIPE['split'], 'location_groups': groups, 'frames': parts,
            'all_raw_ids': sorted(ids), 'location_intersection': [],
            'heldout_locations': len(groups['test']),
            'warning': 'Heldout locations are the entire calibration/design/evaluation pool; small independent sample sizes can require infinite conformal thresholds. Tiles do not increase independent groups.'}

def windows(width, height):
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError('Invalid image dimensions')
    def starts(length):
        count = 1 if length <= 1024 else math.ceil((length - 1024) / 512 + 1)
        result = [512 * i for i in range(count)]
        if count > 1 and result[-1] + 1024 > length: result[-1] = length - 1024
        return result
    candidates = [[x, y, x + 1024, y + 1024] for x, y in itertools.product(starts(width), starts(height))]
    rates = [max(0, min(w[2], width) - w[0]) * max(0, min(w[3], height) - w[1]) / 1024 ** 2 for w in candidates]
    if not any(r > .6 for r in rates):
        best = max(rates); rates = [1 if abs(r - best) < .01 else r for r in rates]
    return [w for w, rate in zip(candidates, rates) if rate > .6]

def obb(coords):
    import cv2
    import numpy as np
    (x, y), (w, h), angle = cv2.minAreaRect(np.asarray(coords, dtype=np.float32).reshape(-1, 2))
    if not all(math.isfinite(v) for v in [x, y, w, h, angle]) or min(w, h) <= 0:
        raise ValueError('Nonfinite or nonpositive canonical OBB')
    a = math.radians(angle)
    if w < h: w, h, a = h, w, a + math.pi / 2
    return [x, y, w, h, (a + math.pi / 2) % math.pi - math.pi / 2]

def annotations(content, frame_id):
    import numpy as np
    from shapely.geometry import Polygon
    result = []
    for line_number, line in enumerate(content.decode('utf-8-sig').splitlines(), 1):
        if not line.strip() or line.startswith(('imagesource:', 'gsd:')): continue
        t = line.split()
        if len(t) != 10 or t[8] not in CLASSES + ['ignored'] or not re.fullmatch(r'[0-9]+', t[9]):
            raise ValueError('Unknown/malformed annotation: ' + frame_id + ':' + str(line_number))
        difficulty = int(t[9])
        if difficulty > 100: raise ValueError('Unsupported source difficulty: ' + str(difficulty))
        try:
            exact = [Fraction(v) for v in t[:8]]; coords = [float(v) for v in exact]
            if not all(math.isfinite(v) for v in coords): raise ValueError('Nonfinite polygon')
            p = list(zip(exact[::2], exact[1::2])); x, y = p[0]; other = next((q for q in p if q != p[0]), None)
            zero = other is None or all((other[0] - x) * (b - y) == (other[1] - y) * (a - x) for a, b in p)
            fp = np.asarray(coords, dtype=np.float32)
            if not np.isfinite(fp).all(): raise ValueError('Nonfinite float32 polygon')
            poly = Polygon(fp.reshape(4, 2))
            if not zero and (not poly.is_valid or poly.area <= 0): raise ValueError('Invalid positive polygon')
            if not zero: obb(fp.tolist())
        except (ValueError, OverflowError, ZeroDivisionError) as error:
            raise ValueError('Malformed/nonfinite/unrepresentable polygon: ' + frame_id + ':' + str(line_number)) from error
        reasons = ([] if t[8] != 'ignored' else ['non_target_ignored']) + ([] if not zero else ['finite_exact_zero_area'])
        result.append({'source_object_id': frame_id + ':' + str(line_number), 'line': line_number,
                       'source_class': t[8], 'canonical_class': t[8] if t[8] in CLASSES else None,
                       'source_coordinate_tokens': t[:8], 'source_polygon_float32': fp.tolist(),
                       'source_difficulty': difficulty, 'exclusion_reasons': reasons})
    return result

def window_objects(objects, window):
    import numpy as np
    from shapely.geometry import Polygon, box
    x, y, x2, y2 = window; region = box(x, y, x2, y2)
    emitted, decisions = [], []
    for obj in objects:
        if obj['exclusion_reasons']:
            decisions.append({'id': obj['source_object_id'], 'iof': None, 'reason': 'source_excluded'})
            continue
        polygon = Polygon(np.asarray(obj['source_polygon_float32']).reshape(4, 2))
        area = float(np.float32(polygon.area))
        iof = polygon.intersection(region).area / max(area, 1e-6)
        if not math.isfinite(iof) or iof < 0: raise ValueError('Invalid IOF')
        selected = iof >= .7
        record = {'id': obj['source_object_id'], 'iof': iof, 'reason': 'emitted' if selected else 'below_iof'}
        if selected:
            shifted = (np.asarray(obj['source_polygon_float32'], dtype=np.float32) + np.asarray([-x, -y] * 4, dtype=np.float32)).tolist()
            diff = 2 if iof < 1 else obj['source_difficulty']
            obb(shifted)
            emitted.append({'id': obj['source_object_id'], 'class': obj['canonical_class'],
                            'polygon': shifted, 'difficulty': diff})
            record['difficulty'] = diff
        decisions.append(record)
    return emitted, decisions

def label_bytes(objects):
    # Float32 scalar formatting matches upstream map(str, float32 coordinates).
    import numpy as np
    return ''.join(' '.join(str(v) for v in np.asarray(o['polygon'], dtype=np.float32)) + ' ' +
                   o['class'] + ' ' + str(o['difficulty']) + '\n' for o in objects).encode()

def decode(content):
    import cv2
    import numpy as np
    from PIL import Image
    try:
        with Image.open(io.BytesIO(content)) as im: im.verify()
        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3 or image.shape[2] != 3: raise ValueError('Missing pixels')
    except Exception as error: raise ValueError('Undecodable image') from error
    return image

def crop(image, window):
    import numpy as np
    x, y, x2, y2 = window; result = np.zeros((1024, 1024, 3), dtype=np.uint8)
    part = image[y:y2, x:x2]; result[:part.shape[0], :part.shape[1]] = part
    return result

def encode(image):
    import cv2
    ok, buf = cv2.imencode('.png', image)
    if not ok: raise ValueError('PNG encoder failed')
    return buf.tobytes()

class ArchiveSet:
    """Verified read-only ZIPs, exact paired member sets, stable open snapshots."""
    def __init__(self, root):
        self.root = Path(root).resolve(); self.stack = ExitStack(); self.zips = {}; self.frames = []
        self.file_signatures = {}; self.records = []
        try:
            for name, expected in sorted(ARCHIVES.items()):
                path = self.root / name
                if path.is_symlink() or not path.is_file() or path.resolve().parent != (self.root / name).parent.resolve():
                    raise ValueError('Invalid archive path')
                self.file_signatures[name] = signature(path)
                actual = {'path': name, 'bytes': path.stat().st_size, 'sha256': file_hash(path)}
                if actual['bytes'] != expected['bytes'] or actual['sha256'] != expected['sha256']:
                    raise ValueError('Unverified original archive: ' + name)
                self.records.append(actual); z = self.stack.enter_context(zipfile.ZipFile(path)); self.zips[name] = z
            seen = set(); normalized = {}
            for split, count in EXPECTED_ORIGINAL.items():
                members = {}
                for kind, suffix in [('images', '.jpg'), ('annfile', '.txt'), ('labels', '.xml')]:
                    z = self.zips[split + '/' + kind + '.zip']; names = [i.filename for i in z.infolist()]
                    if len(names) != len(set(names)) or len(names) != count: raise ValueError('Duplicate/wrong original member count')
                    for i in z.infolist():
                        p = PurePosixPath(i.filename); mode = i.external_attr >> 16
                        if len(p.parts) != 1 or p.suffix != suffix or i.is_dir() or '\\' in i.filename or stat.S_ISLNK(mode):
                            raise ValueError('Nonflat/unsafe original member')
                    members[kind] = {PurePosixPath(n).stem: n for n in names}
                    if len(members[kind]) != count: raise ValueError('Duplicate original stems')
                if not set(members['images']) == set(members['annfile']) == set(members['labels']):
                    raise ValueError('Original image/annotation/XML pair mismatch')
                for raw_id in sorted(members['images']):
                    if raw_id in seen: raise ValueError('Original frame occurs in multiple splits')
                    seen.add(raw_id); meta = identity(raw_id); key = tuple(meta['recording_key'])
                    if key in normalized and normalized[key] != meta['raw_recording_id']:
                        raise ValueError('Normalized recording alias collision')
                    normalized[key] = meta['raw_recording_id']
                    self.frames.append(dict(meta, original_split=split, members={k: members[k][raw_id] for k in members}))
            self.frames.sort(key=lambda f: f['raw_frame_id']); self.assert_unchanged()
        except BaseException:
            self.stack.close(); raise
    def __enter__(self): return self
    def __exit__(self, *args): return self.stack.__exit__(*args)
    def assert_unchanged(self):
        if any(signature(self.root / k) != value for k, value in self.file_signatures.items()):
            raise ValueError('Original archive changed during operation')
    def read(self, frame, kind):
        content = self.zips[frame['original_split'] + '/' + kind + '.zip'].read(frame['members'][kind])
        return content

def raw_audit(source):
    from PIL import Image
    frames = []; classes = Counter(); difficulties = Counter(); reasons = Counter(); excluded = []
    for frame in source.frames:
        image, ann, xml = (source.read(frame, kind) for kind in ['images', 'annfile', 'labels'])
        with Image.open(io.BytesIO(image)) as im: width, height = im.size
        objs = annotations(ann, frame['raw_frame_id'])
        for obj in objs:
            classes[obj['source_class']] += 1; difficulties[str(obj['source_difficulty'])] += 1
            if obj['exclusion_reasons']:
                reasons.update(obj['exclusion_reasons']); excluded.append(dict(obj, original_split=frame['original_split'], location=frame['location']))
        frames.append(dict(frame, width=width, height=height, expected_tiles=len(windows(width, height)),
                           contents={kind: {'bytes': len(buf), 'sha256': digest(buf)} for kind, buf in zip(['images','annfile','labels'], [image,ann,xml])},
                           source_objects=len(objs), valid_objects=sum(not o['exclusion_reasons'] for o in objs)))
    source.assert_unchanged()
    return {'source_revision': SOURCE_REVISION, 'archives': source.records, 'frames': frames,
            'classes': dict(classes), 'source_difficulty': dict(difficulties),
            'source_objects': sum(classes.values()), 'valid_objects': sum(classes.values()) - len(excluded),
            'excluded_objects': len(excluded), 'exclusion_reasons': dict(reasons), 'exclusions': excluded}

def pilot_ids(frames, count=20):
    if type(count) is not int or not 1 <= count <= 20: raise ValueError('Pilot requires 1..20 frames')
    return [f['raw_frame_id'] for f in sorted(frames, key=lambda f: digest(('codrone-pilot-v1:' + f['raw_frame_id']).encode()))[:count]]

def pilot_contract():
    import cv2
    import numpy as np
    return {'selection': 'lowest sha256(codrone-pilot-v1:raw_id), no labels/outcomes',
            'encoder_version': cv2.__version__, 'numpy_version': np.__version__,
            'disk_writes': 0, 'claim': 'Measured sample, not proof that all tiles fit'}

def validate_pilot_metadata(pilot):
    contract = pilot_contract()
    if set(pilot) != set(contract) | {'recipe_sha256', 'raw_sha256', 'frame_count', 'records', 'elapsed_seconds'}:
        raise ValueError('Capacity pilot metadata fields differ')
    if json_bytes({key: pilot[key] for key in contract}) != json_bytes(contract):
        raise ValueError('Capacity pilot selection/disk_writes/runtime contract differs')
    elapsed = pilot['elapsed_seconds']
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError('Invalid capacity pilot elapsed observation')
    if type(pilot['frame_count']) is not int:
        raise ValueError('Invalid capacity pilot frame count')

def capacity_pilot(source, raw, count=20):
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    selected = pilot_ids(raw['frames'], count); lookup = {f['raw_frame_id']: f for f in raw['frames']}
    records = []; start = time.monotonic()
    for key in selected:
        frame = lookup[key]; content = source.read(frame, 'images')
        if digest(content) != frame['contents']['images']['sha256']: raise ValueError('Pilot source content changed')
        image = decode(content)
        if list(image.shape[:2]) != [frame['height'], frame['width']]: raise ValueError('Pilot shape mismatch')
        sizes = []
        for win in windows(frame['width'], frame['height']):
            patch = crop(image, win); encoded = encode(patch)
            if not np.array_equal(decode(encoded), patch): raise ValueError('PNG is not lossless')
            sizes.append(len(encoded))
        records.append({'raw_frame_id': key, 'source_sha256': digest(content), 'png_bytes': sizes, 'tiles': len(sizes)})
    source.assert_unchanged()
    return dict(pilot_contract(),
                recipe_sha256=digest(json_bytes(RECIPE)), raw_sha256=digest(json_bytes(raw)),
                frame_count=len(records), records=records, elapsed_seconds=time.monotonic() - start)

def capacity_gate(raw, pilot, free_bytes):
    validate_pilot_metadata(pilot)
    if type(free_bytes) is not int or free_bytes < 0: raise ValueError('Invalid free capacity')
    if pilot.get('recipe_sha256') != digest(json_bytes(RECIPE)) or pilot.get('raw_sha256') != digest(json_bytes(raw)):
        raise ValueError('Capacity pilot snapshot differs')
    expected = pilot_ids(raw['frames'], min(20, len(raw['frames'])))
    records = pilot.get('records', [])
    if [r['raw_frame_id'] for r in records] != expected or pilot.get('frame_count') != len(expected):
        raise ValueError('Full deterministic <=20-frame capacity pilot required')
    sizes = [n for r in records for n in r['png_bytes']]
    if not sizes or any(type(n) is not int or n <= 0 for n in sizes): raise ValueError('Invalid measured PNG bytes')
    lookup = {f['raw_frame_id']: f for f in raw['frames']}
    for r in records:
        f = lookup[r['raw_frame_id']]
        if r['source_sha256'] != f['contents']['images']['sha256'] or len(r['png_bytes']) != r['tiles'] or r['tiles'] != f['expected_tiles']:
            raise ValueError('Incomplete capacity window measurements')
    tiles = sum(f['expected_tiles'] for f in raw['frames'])
    mean = sum(sizes) / len(sizes)
    # Conservative sample bound, still not a worst-case proof. Every write also
    # checks the live reserve and total cap; underestimation leaves no manifest.
    required = math.ceil(max(sizes) * tiles * 1.25) + CONTROL_ALLOWANCE
    return {'status': 'ADMIT_WITH_LIVE_BUDGET' if free_bytes >= required + RESERVE_BYTES else 'BLOCK_CAPACITY',
            'estimated_png_bytes_mean': math.ceil(mean * tiles), 'budget_bytes': required,
            'free_bytes': free_bytes, 'reserve_bytes': RESERVE_BYTES, 'expected_tiles': tiles,
            'sample_tiles': len(sizes), 'sample_png_bytes_min': min(sizes), 'sample_png_bytes_max': max(sizes),
            'all_data_fit_proven': False}

def runtime_provenance():
    import cv2
    import numpy
    import PIL
    import shapely
    return {'implementation_sha256': digest(snapshot(Path(__file__))), 'opencv': cv2.__version__,
            'numpy': numpy.__version__, 'Pillow': PIL.__version__, 'shapely': shapely.__version__}

def safe_relative(root, name):
    p = PurePosixPath(name)
    if not isinstance(name, str) or not p.parts or p.is_absolute() or any(x in ('.', '..') for x in p.parts) or '\\' in name or p.as_posix() != name:
        raise ValueError('Unsafe prepared path')
    target = Path(root).joinpath(*p.parts)
    for parent in [target, *target.parents]:
        if parent == Path(root).parent: break
        if parent.is_symlink(): raise ValueError('Symlink in prepared path')
    if not target.resolve().is_relative_to(Path(root).resolve()): raise ValueError('Path escape')
    return target

class OutputBudget:
    def __init__(self, root, cap): self.root, self.cap, self.used = Path(root), cap, 0
    def charge(self, size):
        # Preserve a live 40 GiB floor. Other jobs can still race disk usage;
        # there is no host-wide filesystem reservation. Recheck every write.
        if self.used + size > self.cap or shutil.disk_usage(self.root).free - size < RESERVE_BYTES:
            raise ValueError('Live capacity budget exhausted; incomplete candidate, no DATA_OK')
        self.used += size
    def write(self, name, content):
        self.charge(len(content)); path = safe_relative(self.root, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as handle: handle.write(content)
        return {'file': name, 'bytes': len(content), 'sha256': digest(content)}

class JsonLines:
    def __init__(self, budget, name):
        self.budget, self.name = budget, name; self.h = hashlib.sha256(); self.size = 0
        self.handle = safe_relative(budget.root, name).open('xb')
    def write(self, value):
        b = json_bytes(value); self.budget.charge(len(b)); self.handle.write(b); self.h.update(b); self.size += len(b)
    def close(self): self.handle.close()
    def record(self): return {'file': self.name, 'bytes': self.size, 'sha256': self.h.hexdigest()}

def checked_control(root, record):
    if set(record) != {'file', 'bytes', 'sha256'} or type(record['bytes']) is not int or record['bytes'] < 0:
        raise ValueError('Malformed control record')
    b = snapshot(safe_relative(root, record['file']))
    if len(b) != record['bytes'] or digest(b) != record['sha256']: raise ValueError('Control content changed: ' + record['file'])
    return b

def checked_lines(root, record):
    path = safe_relative(root, record['file']); before = signature(path); h = hashlib.sha256(); size = 0
    with path.open('rb') as handle:
        for line in handle:
            if not line.endswith(b'\n') or not line.strip(): raise ValueError('Malformed JSONL line')
            size += len(line); h.update(line)
            value = strict_json(line)
            if not isinstance(value, dict): raise ValueError('JSONL row is not an object: ' + record['file'])
            yield value
    if before != signature(path) or type(record['bytes']) is not int or size != record['bytes'] or h.hexdigest() != record['sha256']:
        raise ValueError('JSONL content changed: ' + record['file'])

def recheck_control_hash(root, record):
    path = safe_relative(root, record['file'])
    if path.is_symlink() or not path.is_file(): raise ValueError('Control is not a regular file: ' + record['file'])
    before = signature(path); h = hashlib.sha256(); size = 0
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(chunk); size += len(chunk)
    if path.is_symlink() or before != signature(path) or type(record['bytes']) is not int or size != record['bytes'] or h.hexdigest() != record['sha256']:
        raise ValueError('Control content changed: ' + record['file'])


def expect_next(iterator, expected, label):
    sentinel = object(); actual = next(iterator, sentinel)
    # Canonical serialization prevents bool == int and accepts no NaN.
    if actual is sentinel or json_bytes(actual) != json_bytes(expected):
        raise ValueError('Reconstructed ' + label + ' mismatch')

def exhausted(iterator, label):
    sentinel = object()
    if next(iterator, sentinel) is not sentinel: raise ValueError('Extra ' + label + ' rows')

def frames_with_pixels(source, raw):
    for f in raw['frames']:
        image = source.read(f, 'images'); ann = source.read(f, 'annfile')
        for kind, b in [('images', image), ('annfile', ann)]:
            if f['contents'][kind] != {'bytes': len(b), 'sha256': digest(b)}: raise ValueError('Source member changed')
        pixels = decode(image)
        if [pixels.shape[1], pixels.shape[0]] != [f['width'], f['height']]: raise ValueError('Decoded source shape differs')
        yield f, pixels, annotations(ann, f['raw_frame_id'])

def tile_specs(frame, objects, assigned_split):
    for win in windows(frame['width'], frame['height']):
        key = frame['raw_frame_id'] + '__1024__' + str(win[0]) + '___' + str(win[1])
        emitted, decisions = window_objects(objects, win); label = label_bytes(emitted)
        base = {'tile_id': key, 'raw_frame_id': frame['raw_frame_id'],
                'original_split': frame['original_split'], 'assigned_split': assigned_split,
                'scene_id': frame['location'], 'window': win,
                'to_source': {'scale': 1, 'offset': win[:2]},
                'unpadded_width': min(1024, frame['width'] - win[0]),
                'unpadded_height': min(1024, frame['height'] - win[1]),
                'width': 1024, 'height': 1024, 'objects': len(emitted),
                'source_image_sha256': frame['contents']['images']['sha256'],
                'source_annotation_sha256': frame['contents']['annfile']['sha256'],
                'recipe_sha256': digest(json_bytes(RECIPE))}
        gt = [{'image_id': key, 'scene_id': frame['location'], 'class': o['class'],
               'obb': obb(o['polygon']), 'gt_id': key + ':' + str(j),
               'source_object_id': o['id'], 'difficulty': o['difficulty']} for j, o in enumerate(emitted)]
        # Positional source object table gives every decision an unambiguous ID,
        # without repeating a long recording name millions of times.
        audit = {'tile_id': key, 'decisions': [[d['iof'], {'below_iof': 0, 'emitted': 1, 'source_excluded': 2}[d['reason']],
                                             d.get('difficulty')] for d in decisions]}
        yield base, label, gt, audit

def summary_template(raw, split):
    return {'frames': len(raw['frames']), 'tiles': 0, 'source_objects': raw['source_objects'],
            'valid_source_objects': raw['valid_objects'], 'excluded_source_objects': raw['excluded_objects'],
            'emitted_instances': 0, 'unrepresented_valid_source_objects': 0,
            'source_objects_represented': 0, 'decision_counts': {'below_iof': 0, 'emitted': 0, 'source_excluded': 0},
            'splits': {s: {'frames': len(split['frames'][s]), 'locations': len(split['location_groups'][s]),
                           'images': 0, 'objects': 0} for s in ['train', 'test']}}

def account(summary, assigned, objects, audit_rows, gt_ids):
    represented = set(gt_ids); valid = {o['source_object_id'] for o in objects if not o['exclusion_reasons']}
    summary['source_objects_represented'] += len(represented)
    summary['unrepresented_valid_source_objects'] += len(valid - represented)
    for row in audit_rows:
        for _, reason, _ in row['decisions']:
            summary['decision_counts'][['below_iof', 'emitted', 'source_excluded'][reason]] += 1
    emitted = sum(v[1] == 1 for r in audit_rows for v in r['decisions'])
    summary['tiles'] += len(audit_rows); summary['emitted_instances'] += emitted
    summary['splits'][assigned]['images'] += len(audit_rows); summary['splits'][assigned]['objects'] += emitted

def frame_audit(frame, objects, rows, represented):
    return {'raw_frame_id': frame['raw_frame_id'], 'scene_id': frame['location'],
            'original_split': frame['original_split'], 'source_objects': objects,
            'decision_encoding': '[IOF or null, 0=below_iof/1=emitted/2=source_excluded, emitted_difficulty or null]; same source-object order',
            'windows': rows, 'unrepresented_valid_source_ids': sorted(o['source_object_id'] for o in objects
                if not o['exclusion_reasons'] and o['source_object_id'] not in represented)}

def stage(source_root, output_root):
    """Create a new candidate from frozen ZIPs; never touches DATA_OK or raw files."""
    output = Path(output_root).absolute(); source_root = Path(source_root).resolve()
    if output.exists() or output.is_symlink(): raise ValueError('Output must not already exist')
    if not output.parent.is_dir() or output.parent.is_symlink() or output.resolve().is_relative_to(source_root):
        raise ValueError('Use a separate existing output parent, outside original archives')
    with ArchiveSet(source_root) as source:
        raw = raw_audit(source); split = split_locations(raw['frames'])
        pilot = capacity_pilot(source, raw); gate = capacity_gate(raw, pilot, shutil.disk_usage(output.parent).free)
        if gate['status'] != 'ADMIT_WITH_LIVE_BUDGET':
            raise ValueError('BLOCK_CAPACITY ' + json.dumps(gate, sort_keys=True))
        output.mkdir(exist_ok=False); budget = OutputBudget(output, gate['budget_bytes']); controls = []
        for name, value in [('raw.json', raw), ('split.json', split),
                            ('recipe.json', {'recipe': RECIPE, 'runtime': runtime_provenance()}),
                            ('capacity.json', {'pilot': pilot, 'gate': gate})]:
            controls.append(budget.write(name, json_bytes(value)))
        names = ['frame_to_tile.jsonl', 'object_audit.jsonl', 'gt_train.jsonl', 'gt_test.jsonl']
        streams = {name: JsonLines(budget, name) for name in names}
        assignments = {i: s for s, ids in split['frames'].items() for i in ids}
        mapping = {}; summary = summary_template(raw, split)
        try:
            for f, pixels, objects in frames_with_pixels(source, raw):
                assigned = assignments[f['raw_frame_id']]; audits = []; represented = set()
                for base, label, gt, audit in tile_specs(f, objects, assigned):
                    key = base['tile_id']; mapping[key] = f['location']
                    image = budget.write(assigned + '/images/' + key + '.png', encode(crop(pixels, base['window'])))
                    annotation = budget.write(assigned + '/annfiles/' + key + '.txt', label)
                    streams['frame_to_tile.jsonl'].write(dict(base, image=image, annotation=annotation))
                    for row in gt: streams['gt_' + assigned + '.jsonl'].write(row); represented.add(row['source_object_id'])
                    audits.append(audit)
                streams['object_audit.jsonl'].write(frame_audit(f, objects, audits, represented))
                account(summary, assigned, objects, audits, represented)
        finally:
            for stream in streams.values(): stream.close()
        controls.extend(stream.record() for stream in streams.values())
        controls.append(budget.write('scene_map.json', json_bytes(mapping)))
        source.assert_unchanged()
        manifest = {'dataset': 'codrone', 'inventory_version': 2, 'codrone_inventory_version': 1,
                    'status': 'PREPARED_REQUIRES_INTEGRATION_REVIEW', 'data_ready': False,
                    'source_root': str(source_root), 'root': str(output.resolve()),
                    'recipe_sha256': digest(json_bytes(RECIPE)), 'source_revision': SOURCE_REVISION,
                    'controls': controls, 'summary': summary, 'splits': summary['splits'],
                    'mandatory_integration': ['dispatch codrone.verify', 'loader/canonical include difficulty 2',
                        'use complete location scene_map', 'new location-disjoint training',
                        'independent review before main may publish DATA_OK']}
        budget.write('data_manifest.json', json_bytes(manifest))
    # Re-read every raw member, prepared pixel, annotation, mapping, and summary.
    # A failure leaves an explicitly non-ready candidate, never a readiness marker.
    return verify(output, source_root)

def verify(output_root, source_root=None):
    """Full source-to-pixel replay; a verified candidate is still NOT DATA_OK."""
    import numpy as np
    root = Path(output_root).absolute()
    if root.is_symlink(): raise ValueError('Prepared root must not be a symlink')
    root = root.resolve(); manifest_bytes = snapshot(root / 'data_manifest.json'); m = strict_json(manifest_bytes)
    if m.get('dataset') != 'codrone' or m.get('inventory_version') != 2 or m.get('codrone_inventory_version') != 1 or m.get('data_ready') is not False or m.get('status') != 'PREPARED_REQUIRES_INTEGRATION_REVIEW':
        raise ValueError('Not a non-ready CODrone v2 candidate')
    if m.get('root') != str(root) or m.get('recipe_sha256') != digest(json_bytes(RECIPE)) or m.get('source_revision') != SOURCE_REVISION:
        raise ValueError('Prepared root/recipe/source pin differs')
    controls = m.get('controls', []); byname = {r['file']: r for r in controls}
    required = {'raw.json', 'split.json', 'recipe.json', 'capacity.json', 'frame_to_tile.jsonl',
                'object_audit.jsonl', 'gt_train.jsonl', 'gt_test.jsonl', 'scene_map.json'}
    if len(controls) != len(byname) or set(byname) != required: raise ValueError('Missing/duplicate/unknown controls')
    docs = {n: strict_json(checked_control(root, byname[n])) for n in required if n.endswith('.json')}
    if json_bytes(docs['recipe.json']) != json_bytes({'recipe': RECIPE, 'runtime': runtime_provenance()}):
        raise ValueError('Runtime/source/geometry provenance differs; no silent regeneration')
    with ArchiveSet(source_root or m['source_root']) as source:
        raw = raw_audit(source); split = split_locations(raw['frames'])
        if json_bytes(raw) != json_bytes(docs['raw.json']) or json_bytes(split) != json_bytes(docs['split.json']):
            raise ValueError('Raw snapshot or location assignment differs')
        capacity = docs['capacity.json']; pilot = capacity['pilot']
        validate_pilot_metadata(pilot)
        # Historical free space and elapsed time cannot be replayed. Check only
        # their schema/arithmetic here; this is not a new live capacity admission.
        gate = capacity_gate(raw, pilot, capacity['gate']['free_bytes'])
        if json_bytes(gate) != json_bytes(capacity['gate']) or gate['status'] != 'ADMIT_WITH_LIVE_BUDGET':
            raise ValueError('Recorded capacity gate is inconsistent')
        pilot_records = {r['raw_frame_id']: r for r in pilot['records']}
        pilot_frames = set(pilot_ids(raw['frames'], min(20, len(raw['frames']))))
        replayed_pilot = set()
        import cv2
        cv2.setNumThreads(1)
        streams = {n: checked_lines(root, byname[n]) for n in required if n.endswith('.jsonl')}
        assignments = {i: s for s, ids in split['frames'].items() for i in ids}
        mapping = {}; summary = summary_template(raw, split); expected_files = set(required) | {'data_manifest.json'}
        for f, pixels, objects in frames_with_pixels(source, raw):
            assigned = assignments[f['raw_frame_id']]; audits = []; represented = set()
            is_pilot = f['raw_frame_id'] in pilot_frames; pilot_sizes = []
            for base, label, gt, audit in tile_specs(f, objects, assigned):
                key = base['tile_id']; mapping[key] = f['location']
                img_name = assigned + '/images/' + key + '.png'; ann_name = assigned + '/annfiles/' + key + '.txt'
                img = snapshot(safe_relative(root, img_name)); ann = snapshot(safe_relative(root, ann_name))
                if ann != label: raise ValueError('Prepared annotation differs from source/policy')
                patch = crop(pixels, base['window'])
                if not np.array_equal(decode(img), patch):
                    raise ValueError('Prepared tile pixels/parent/window differ')
                if is_pilot:
                    # Re-encode from the verified source pixels, never from the
                    # claimed sizes or the prepared PNG's possibly different encoding.
                    encoded = encode(patch)
                    if not np.array_equal(decode(encoded), patch): raise ValueError('Pilot PNG is not lossless')
                    pilot_sizes.append(len(encoded))
                actual = dict(base, image={'file': img_name, 'bytes': len(img), 'sha256': digest(img)},
                              annotation={'file': ann_name, 'bytes': len(ann), 'sha256': digest(ann)})
                expect_next(streams['frame_to_tile.jsonl'], actual, 'frame-to-tile')
                for row in gt: expect_next(streams['gt_' + assigned + '.jsonl'], row, 'GT'); represented.add(row['source_object_id'])
                expected_files.update([img_name, ann_name]); audits.append(audit)
            if is_pilot:
                measured = {'raw_frame_id': f['raw_frame_id'],
                            'source_sha256': f['contents']['images']['sha256'],
                            'png_bytes': pilot_sizes, 'tiles': len(pilot_sizes)}
                if json_bytes(measured) != json_bytes(pilot_records[f['raw_frame_id']]):
                    raise ValueError('Capacity pilot frame/window PNG measurements differ from source replay')
                replayed_pilot.add(f['raw_frame_id'])
            expect_next(streams['object_audit.jsonl'], frame_audit(f, objects, audits, represented), 'object audit')
            account(summary, assigned, objects, audits, represented)
        if replayed_pilot != pilot_frames: raise ValueError('Incomplete capacity pilot replay')
        for name, stream in streams.items(): exhausted(stream, name)
        if json_bytes(mapping) != json_bytes(docs['scene_map.json']): raise ValueError('Explicit location scene map differs')
        if json_bytes(summary) != json_bytes(m['summary']) or json_bytes(summary['splits']) != json_bytes(m['splits']):
            raise ValueError('Counts or split summary differ')
        if summary['source_objects'] != summary['valid_source_objects'] + summary['excluded_source_objects'] or summary['valid_source_objects'] != summary['source_objects_represented'] + summary['unrepresented_valid_source_objects']:
            raise ValueError('Source count conservation failed')
        # Recheck every control after replay. This catches a control file changed
        # after its cached parse, including scene_map mutations during tile_specs.
        for name in required:
            recheck_control_hash(root, byname[name])
        actual_files = set()
        for p in root.rglob('*'):
            if p.is_symlink(): raise ValueError('Symlink in prepared inventory')
            if p.is_file(): actual_files.add(p.relative_to(root).as_posix())
        if actual_files != expected_files: raise ValueError('Extra/missing prepared files')
        source.assert_unchanged()
    if snapshot(root / 'data_manifest.json') != manifest_bytes: raise ValueError('Manifest changed during verification')
    return {'status': 'VERIFIED_CANDIDATE_NOT_DATA_READY', 'data_ready': False,
            'manifest_sha256': digest(manifest_bytes), 'summary': summary,
            'capacity_evidence': {'pilot_frames_replayed': len(replayed_pilot),
                'pilot_windows_replayed': gate['sample_tiles'], 'live_capacity_verified': False,
                'elapsed_seconds': 'historical observation, not replay verified',
                'free_bytes': 'historical observation, not replay verified',
                'disk_writes': 'zero-write contract checked; historical activity not attested'},
            'all_data_fit_proven': True, 'meaning': 'Existing complete pixels verified; no approval to publish DATA_OK'}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['audit', 'pilot', 'stage', 'verify'])
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.action in ['stage', 'verify']:
        if args.output is None: parser.error('--output required')
        result = (stage if args.action == 'stage' else verify)(args.source, args.output) if args.action == 'stage' else verify(args.output, args.source)
    else:
        with ArchiveSet(args.source) as source:
            raw = raw_audit(source); split = split_locations(raw['frames'])
            result = {'raw_sha256': digest(json_bytes(raw)), 'source_objects': raw['source_objects'],
                      'valid_objects': raw['valid_objects'], 'excluded_objects': raw['excluded_objects'],
                      'classes': raw['classes'], 'exclusion_reasons': raw['exclusion_reasons'], 'split': split, 'data_ready': False}
            if args.action == 'pilot':
                pilot = capacity_pilot(source, raw); result['pilot'] = pilot
                result['capacity'] = capacity_gate(raw, pilot, shutil.disk_usage(args.source).free)
    print(json.dumps(result, sort_keys=True, allow_nan=False))

if __name__ == '__main__': main()

"""Integrity checks for EAV inference artifacts; no training or calibration logic."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

CLASSES = ('car', 'truck', 'traffic-sign', 'people', 'motor', 'bicycle',
           'traffic-light', 'tricycle', 'bridge', 'bus', 'boat', 'ship')
ARRAY_KEYS = ('tile_names', 'tile_index', 'gt_tile', 'gt_box', 'gt_label',
              'pred_tile', 'pred_box', 'pred_prob')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_arrays(arrays, min_score=0.05):
    """Fail on malformed or non-finite data, including empty-shard shapes."""
    missing = set(ARRAY_KEYS) - set(arrays)
    if missing:
        raise ValueError(f'missing export arrays: {sorted(missing)}')
    names, index = arrays['tile_names'], arrays['tile_index']
    if names.ndim != 1 or names.dtype.kind not in 'US':
        raise ValueError('tile_names must be a one-dimensional string array')
    n = len(names)
    if len(set(map(str, names))) != n:
        raise ValueError('duplicate tile names')
    for key in ('tile_index', 'gt_tile', 'gt_label', 'pred_tile'):
        arr = arrays[key]
        if arr.ndim != 1 or arr.dtype.kind not in 'iu':
            raise ValueError(f'{key} must be a one-dimensional integer array')
    if len(index) != n or len(set(index.tolist())) != n or np.any(index < 0):
        raise ValueError('tile_index length, uniqueness or range is invalid')
    for key in ('gt_tile', 'pred_tile'):
        arr = arrays[key]
        if np.any(arr < 0) or np.any(arr >= n):
            raise ValueError(f'{key} refers to a nonexistent local tile')
    for key, tile_key in (('gt_box', 'gt_tile'), ('pred_box', 'pred_tile')):
        box = arrays[key]
        if box.shape != (len(arrays[tile_key]), 5):
            raise ValueError(f'{key} must have shape (number of objects, 5)')
        if not np.isfinite(box).all():
            raise ValueError(f'{key} contains NaN or infinity')
        if np.any(box[:, 2:4] <= 0):
            raise ValueError(f'{key} has nonpositive side lengths')
        if np.any(box[:, :4] < 0) or np.any(box[:, :4] > 1):
            raise ValueError(f'{key} spatial coordinates must be normalized to [0, 1]')
    labels, prob = arrays['gt_label'], arrays['pred_prob']
    if len(labels) != len(arrays['gt_box']) or np.any(labels < 0) or np.any(labels >= len(CLASSES)):
        raise ValueError('ground-truth label length or class index is invalid')
    if prob.shape != (len(arrays['pred_box']), len(CLASSES)):
        raise ValueError('pred_prob must have one twelve-class row per detection')
    if not np.isfinite(prob).all() or np.any(prob < 0) or np.any(prob > 1):
        raise ValueError('pred_prob contains non-finite or out-of-range probabilities')
    if np.any(prob.max(axis=1) < min_score):
        raise ValueError('export includes detections below its declared score floor')


def load_shards(exports, split, allow_partial=False):
    """Validate receipts, hashes, exact modulo ownership and full-split coverage."""
    root = Path(exports)
    files = sorted(root.glob(f'{split}-shard*-of*.npz'))
    if not files:
        raise ValueError(f'no {split} export shards in {root}')
    chunks, receipts, hashes, names_seen, indices_seen, shards_seen = [], {}, {}, set(), set(), set()
    identity = None
    offset = 0
    for path in files:
        receipt_path = path.with_suffix('.json')
        if not receipt_path.is_file():
            raise ValueError(f'missing completion receipt: {receipt_path}')
        with receipt_path.open() as f:
            rec = json.load(f)
        if rec.get('schema') not in ('rotcert.eav-export.v1', 'rotcert.eav-export.v2'):
            raise ValueError(f'unsupported export schema: {receipt_path}')
        required = ('split', 'shard', 'num_shards', 'split_tiles', 'tiles', 'min_score',
                    'checkpoint_sha256', 'config_sha256', 'out_sha256')
        if any(key not in rec for key in required):
            raise ValueError(f'incomplete receipt: {receipt_path}')
        shard, count, total = rec['shard'], rec['num_shards'], rec['split_tiles']
        if any(type(x) is not int for x in (shard, count, total, rec['tiles'])):
            raise ValueError(f'integer shard/count fields required: {receipt_path}')
        if count < 1 or not 0 <= shard < count or total < 1 or rec['tiles'] < 0:
            raise ValueError(f'invalid shard/count range: {receipt_path}')
        expected_name = f'{split}-shard{shard:02d}-of{count:02d}.npz'
        if rec['split'] != split or path.name != expected_name or shard in shards_seen:
            raise ValueError(f'shard name, split or uniqueness mismatch: {path}')
        floor = rec['min_score']
        if not isinstance(floor, (int, float)) or not math.isfinite(floor) or not 0 <= floor <= 0.05:
            raise ValueError('four-method analysis requires finite export min_score <= 0.05')
        for key in ('checkpoint_sha256', 'config_sha256', 'out_sha256'):
            value = rec[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError(f'invalid {key}: {receipt_path}')
        current = (count, total, floor, rec['checkpoint_sha256'], rec['config_sha256'])
        if identity is not None and current != identity:
            raise ValueError('mixed checkpoint/config/score floor/split inventory across shards')
        identity = current
        if rec['schema'].endswith('.v2'):
            if rec.get('class_names') != list(CLASSES) or rec.get('box_format') != 'normalized-cxcywh-angle-radians':
                raise ValueError(f'class order or box convention mismatch: {receipt_path}')
        digest = sha256(path)
        if digest != rec['out_sha256']:
            raise ValueError(f'export hash mismatch: {path}')
        with np.load(path, allow_pickle=False) as z:
            arrays = {key: z[key] for key in z.files}
        validate_arrays(arrays, floor)
        actual = arrays['tile_index']
        expected = np.arange(shard, total, count, dtype=np.int64)
        if len(actual) != rec['tiles'] or not np.array_equal(actual, expected[:len(actual)]):
            raise ValueError(f'tile indices violate modulo shard ownership or receipt count: {path}')
        if not allow_partial and not np.array_equal(actual, expected):
            raise ValueError(f'incomplete {split} shard; --allow-partial is diagnostic-only: {path}')
        new_names, new_indices = set(map(str, arrays['tile_names'])), set(actual.tolist())
        if names_seen & new_names or indices_seen & new_indices:
            raise ValueError(f'duplicate tile names or indices across {split} shards')
        names_seen.update(new_names)
        indices_seen.update(new_indices)
        shards_seen.add(shard)
        arrays['gt_tile'] = arrays['gt_tile'] + offset
        arrays['pred_tile'] = arrays['pred_tile'] + offset
        offset += len(arrays['tile_names'])
        chunks.append(arrays)
        receipts[str(path)] = rec
        hashes[str(path)] = digest
        hashes[str(receipt_path)] = sha256(receipt_path)
    complete = shards_seen == set(range(identity[0])) and len(indices_seen) == identity[1]
    if not allow_partial and not complete:
        raise ValueError(f'missing {split} shards: expected {identity[0]}, found {sorted(shards_seen)}')
    result = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in ARRAY_KEYS}
    result['names'] = result.pop('tile_names').tolist()
    result.update(files=hashes, receipts=receipts, complete=complete,
                  split_tiles=identity[1], num_shards=identity[0], shards=sorted(shards_seen),
                  checkpoint_sha256=identity[3], config_sha256=identity[4], min_score=identity[2])
    return result


def strict_json(data):
    """Represent non-finite numbers as null, with explicit JSON-path annotations."""
    nonfinite = {}

    def clean(value, path):
        if isinstance(value, dict):
            return {str(k): clean(v, f'{path}/{k}') for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [clean(v, f'{path}/{i}') for i, v in enumerate(value)]
        if isinstance(value, (float, np.floating)) and not math.isfinite(value):
            nonfinite[path] = 'NaN' if math.isnan(value) else ('+Infinity' if value > 0 else '-Infinity')
            return None
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        return value

    result = clean(data, '')
    result['nonfinite_values'] = nonfinite
    return result

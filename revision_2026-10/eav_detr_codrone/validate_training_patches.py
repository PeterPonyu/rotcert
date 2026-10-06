#!/usr/bin/env python3
"""Read-only reproduction of the invalid-angle path and actual logging import."""
import argparse
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from export_contract import sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--eav-repo', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    if Path(args.out).exists():
        ap.error('output already exists')
    sys.path.insert(0, args.eav_repo)
    import torch
    import src.misc
    from src.solver import det_engine_codrone
    from src.zoo.eavdetr.decoder import RTDETRTransformer
    from src.zoo.eavdetr.matcher import HungarianMatcher

    probe = SimpleNamespace(eval_spatial_size=[1024, 1024], feat_strides=[8, 16, 32], eps=.01)
    anchors, valid = RTDETRTransformer._generate_anchors(probe)
    invalid = ~valid[..., 0]
    if not invalid.any():
        raise AssertionError('probe must include invalid edge anchors')
    fixed = anchors[invalid].clone()
    original = fixed.clone()
    original[:, 4] = torch.inf  # exactly the unpatched torch.where(..., anchors, inf) angle behavior
    fixed[:, :4] = fixed[:, :4].sigmoid()
    original[:, :4] = original[:, :4].sigmoid()
    original_angle_cost = HungarianMatcher.angle_difference(original[:, 4], torch.zeros(len(original)))
    fixed_angle_cost = HungarianMatcher.angle_difference(fixed[:, 4], torch.zeros(len(fixed)))
    if not torch.isnan(original_angle_cost).all() or not torch.isfinite(fixed).all() or not torch.isfinite(fixed_angle_cost).all():
        raise AssertionError('invalid-angle reproduction disagrees with the patched decoder')
    query = torch.nn.Linear(5, 4)
    with torch.no_grad():
        query.weight.fill_(1.)
        query.bias.zero_()
        bad_query = query(original)
        good_query = query(fixed)
    if torch.isfinite(bad_query).all() or not torch.isfinite(good_query).all():
        raise AssertionError('query-position finite/nonfinite propagation mismatch')
    actual = det_engine_codrone.reduce_dict
    if actual is not src.misc.reduce_dict or actual.__module__ != 'src.misc.logger':
        raise AssertionError('the training engine no longer imports src.misc.logger.reduce_dict')
    logger_source = inspect.getsource(actual)
    if 'all_gather_object' not in logger_source or 'torch.zeros_like(ref)' not in logger_source:
        raise AssertionError('actual logging reducer is missing the key-alignment patch')
    files = [Path(args.eav_repo) / p for p in ('src/zoo/eavdetr/decoder.py', 'src/zoo/eavdetr/matcher.py',
                                              'src/misc/logger.py', 'src/misc/dist.py',
                                              'src/misc/__init__.py', 'src/solver/det_engine_codrone.py')]
    result = dict(schema='rotcert.eav-training-patch-validation.v1', passed=True,
                  invalid_anchors=int(invalid.sum()), original_angle_cost_nan_count=int(torch.isnan(original_angle_cost).sum()),
                  fixed_reference_points_all_finite=bool(torch.isfinite(fixed).all()),
                  fixed_angle_cost_all_finite=bool(torch.isfinite(fixed_angle_cost).all()),
                  training_reduce_dict_module=actual.__module__, training_reduce_dict_path=inspect.getfile(actual),
                  logging_alignment_present=True, inputs={str(p): sha256(p) for p in files},
                  interpretation='Invalid anchors had an infinite unsquashed angle; remainder modulo pi gives NaN. '
                                 'This reproduces the causal path on invalid anchors, not every historical failing batch.')
    with Path(args.out).open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()

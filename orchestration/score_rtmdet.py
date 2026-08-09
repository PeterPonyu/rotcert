#!/usr/bin/env python3
"""Box-side RTMDet-R inference: mmrotate val-split forward pass -> canonical
rotcert detections-JSONL (design §3.1/§3.2, §7 Phase 0).

Runs on the AutoDL box, NOT in CI/tests: mmcv/mmdet/mmrotate/torch are imported
LAZILY inside :func:`_load_model` / :func:`_run_inference` so ``--help`` (and argument
parsing) works in ANY environment, including one with none of them installed --
exercised by ``tests/test_cli_e2e.py``-style ``--help`` smoke tests for the CLI
proper; this script's own ``--help`` should be spot-checked the same way before
relying on it.

mmrotate is stale (design §7 risk register: last release 2023-02) -- ``--mmrotate-commit``
is REQUIRED (no default baked in here) and is stamped into every output record's
sibling ``.provenance.json`` file, per SOTA-REPRODUCTION-PLAN-2026-07-10.md's binding
rule ("the SCORING script must pin a commit; our core never imports mmrotate" -- true
here too: only THIS orchestration script imports mmrotate, ``rotcert``'s core package
never does).

Output schema (one JSON object per line, matches ``rotcert.io.validate_detection``)::

    {"image_id": "<crop-or-image-id>", "scene_id": "<source-image-id-or-null>",
     "class": "<name>", "obb": [cx, cy, w, h, theta_le90], "score": <float>}

``obb``'s ``theta`` is ALWAYS emitted in le90-canonical radians regardless of the
model's native ``angle_version`` (``--angle-convention``): ``rotcert.gwd.
canonicalize_le90`` runs on every box before it is written, so every downstream
``rotcert`` command can assume le90 without re-checking.

Phase-0 VERIFY items (per design §4.1/§7, not resolved by this script)
--------------------------------------------------------------------------
- mmrotate's actual pinned-commit ``angle_version`` for the RTMDet-R-l config
  (assumed ``le90`` per design §2.1, marked [VERIFY]).
- Published VAL (not test) mAP for the reproduction gate (``rotcert.
  orchestration.phase0`` consumes this script's output for that check).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_model(config_path: str, checkpoint_path: str, device: str, mmrotate_commit: str):
    """Lazy import + model construction. Raises ImportError with an actionable
    message if mmcv/mmdet/mmrotate are not installed (expected on any machine that
    is not the AutoDL GPU box)."""
    try:
        import mmrotate  # noqa: F401
        from mmengine.config import Config
        from mmdet.apis import init_detector
    except ImportError as e:
        raise ImportError(
            "score_rtmdet.py requires mmcv/mmdet/mmrotate installed at the pinned "
            f"commit {mmrotate_commit!r} (box-side only; NOT a rotcert core "
            f"dependency -- see this script's module docstring). Original error: {e}"
        ) from e

    cfg = Config.fromfile(config_path)
    if cfg.get('test_dataloader') is None:
        # Train-only configs (by design: inference is this script's job, not
        # tools/test.py, so the vendored config on disk omits eval scaffolding
        # and sets test_dataloader = None) still trip TWO separate mmdet
        # internals that unconditionally dereference cfg.test_dataloader.dataset
        # -- init_detector's default-palette lookup (worked around below via
        # palette='random') and, independently, inference_detector's
        # get_test_pipeline_cfg (no such workaround exists there). Fix at the
        # source: synthesize a minimal inference-only test_dataloader in-memory
        # (never written back to the config file) from the config's own
        # train_pipeline, keeping only image-loading/resize (no GT, no
        # augmentation) plus PackDetInputs.
        train_pipeline = cfg.get('train_pipeline') or cfg.train_dataloader.dataset.pipeline
        resize = next(t for t in train_pipeline if t['type'] == 'mmdet.Resize')
        infer_pipeline = [
            dict(type='mmdet.LoadImageFromFile', backend_args=None),
            dict(type='mmdet.Resize', scale=resize['scale'], keep_ratio=resize.get('keep_ratio', True)),
            dict(type='mmdet.PackDetInputs',
                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
        ]
        cfg.test_dataloader = dict(dataset=dict(
            type=cfg.dataset_type, metainfo=cfg.get('hrsc_metainfo', {}), pipeline=infer_pipeline))

    # palette='random' avoids mmdet init_detector's own default ('none') code
    # path, which separately (and redundantly, now that test_dataloader is
    # synthesized above) dereferences config.test_dataloader.dataset to look up
    # a palette; classes come from the checkpoint's saved dataset_meta either way.
    model = init_detector(cfg, checkpoint_path, palette='random', device=device)
    return model


def _canonicalize_obb(cx: float, cy: float, w: float, h: float, theta: float) -> List[float]:
    # Deferred import: rotcert.gwd is pure numpy and always importable, but keeping
    # the import local to this function documents that everything ABOVE it in the
    # call chain (model loading) is the box-only part.
    from rotcert.gwd import canonicalize_le90

    w2, h2, t2 = canonicalize_le90(w, h, theta)
    return [float(cx), float(cy), float(w2), float(h2), float(t2)]


def _run_inference(
    model, image_paths: List[Path], class_names: List[str], score_thr: float
) -> List[Dict[str, Any]]:
    from mmdet.apis import inference_detector

    records: List[Dict[str, Any]] = []
    for img_path in image_paths:
        result = inference_detector(model, str(img_path))
        image_id = img_path.stem
        if hasattr(result, "pred_instances"):
            # mmdet 3.x / mmrotate dev-1.x: DetDataSample with pred_instances
            # (VERIFIED on-box 2026-07-10 @ mmdet 3.2.0: rotated bboxes are
            # [N,5] (cx, cy, w, h, theta), scores [N], labels [N]).
            inst = result.pred_instances
            bboxes = inst.bboxes.cpu().numpy()
            scores = inst.scores.cpu().numpy()
            labels = inst.labels.cpu().numpy()
            for row, score, cls_idx in zip(bboxes, scores, labels):
                score = float(score)
                if score < score_thr:
                    continue
                cx, cy, w, h, theta = (float(v) for v in row[:5])
                records.append(
                    {
                        "image_id": image_id,
                        "scene_id": None,  # populated downstream via rotcert.io.populate_scene_ids
                        "class": class_names[int(cls_idx)],
                        "obb": _canonicalize_obb(cx, cy, w, h, theta),
                        "score": score,
                    }
                )
        else:
            # legacy mmdet/mmrotate 2.x/0.x: list[np.ndarray] per class, rows
            # [cx, cy, w, h, theta, score]
            for cls_idx, dets in enumerate(result):
                if dets is None or len(dets) == 0:
                    continue
                for row in dets:
                    cx, cy, w, h, theta, score = (float(v) for v in row[:6])
                    if score < score_thr:
                        continue
                    records.append(
                        {
                            "image_id": image_id,
                            "scene_id": None,  # populated downstream via rotcert.io.populate_scene_ids
                            "class": class_names[cls_idx],
                            "obb": _canonicalize_obb(cx, cy, w, h, theta),
                            "score": score,
                        }
                    )
    return records


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Box-side RTMDet-R val-split inference -> rotcert detections-JSONL")
    p.add_argument("--config", required=True, help="mmrotate RTMDet-R-l config path (vendored at the pinned commit)")
    p.add_argument("--checkpoint", required=True, help="frozen zoo checkpoint path")
    p.add_argument("--mmrotate-commit", required=True, help="pinned mmrotate git commit SHA (stamped into provenance)")
    p.add_argument("--images-dir", required=True, help="directory of val-split crop images")
    p.add_argument("--class-names", required=True, help="comma-separated class name list, index-aligned to model output")
    p.add_argument("--score-thr", type=float, default=0.05, help="drop detections below this confidence before writing")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--angle-convention", default="le90", choices=["le90", "le135", "oc"], help="documented for provenance; boxes are ALWAYS re-canonicalized to le90 on output regardless")
    p.add_argument("-o", "--out", required=True)
    args = p.parse_args(argv)

    images_dir = Path(args.images_dir)
    image_paths = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.tif"))
    if not image_paths:
        print(f"error: no images found under {images_dir}", file=sys.stderr)
        return 1
    class_names = args.class_names.split(",")

    model = _load_model(args.config, args.checkpoint, args.device, args.mmrotate_commit)
    records = _run_inference(model, image_paths, class_names, args.score_thr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    provenance = {
        "mmrotate_commit": args.mmrotate_commit,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "angle_convention_declared": args.angle_convention,
        "score_thr": args.score_thr,
        "n_images": len(image_paths),
        "n_detections": len(records),
    }
    prov_path = out_path.with_suffix(out_path.suffix + ".provenance.json")
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    print(f"score_rtmdet: n_images={len(image_paths)} n_detections={len(records)} -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

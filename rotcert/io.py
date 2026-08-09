"""Canonical JSONL schemas for detections, ground truth, and matched pairs.

Every ``rotcert`` command consumes/produces these schemas exclusively -- adapters
(``orchestration/score_rtmdet.py``) are the only format-aware code, mirroring
``asr-gate``'s ``io.py`` design (see that module's docstring for the pattern this
follows).

Canonical fields
------------------
Detection record (``rotcert ingest`` output): ``image_id`` (str, the crop/tile id),
``scene_id`` (str, the SOURCE image id -- design §4.2's exchangeability unit; may be
auto-derived from ``image_id`` via :func:`rotcert.splits.scene_id_from_crop_filename`
if omitted), ``class`` (str), ``obb`` (``[cx, cy, w, h, theta]``, theta radians, ANY
convention on input -- canonicalized to le90 downstream by every scoring function, not
here), ``score`` (float, detector confidence).

Ground-truth record: same as detection minus ``score``, plus optional ``gt_id``.

Matched record (``rotcert match`` output): ``image_id``, ``scene_id``, ``class``,
``pred_obb``, ``gt_obb``, ``pred_score``, ``iou``, ``match_type`` (``"tp"``, ``"fp"``,
``"fn"``) -- one row per detection-or-unmatched-GT, per image, so the file is a
complete record of :func:`rotcert.matching.greedy_match`'s output across a whole
dataset.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

__all__ = [
    "SchemaError",
    "load_jsonl",
    "write_jsonl",
    "to_jsonable",
    "validate_detection",
    "validate_gt",
    "validate_detections",
    "validate_gts",
    "populate_scene_ids",
]


class SchemaError(ValueError):
    """Raised on any rotcert schema validation failure."""


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(path)
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SchemaError(f"{path}:{lineno}: invalid JSON ({e})") from e
    if not records:
        raise SchemaError(f"{path}: contains no records")
    return records


def to_jsonable(obj: Any) -> Any:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None  # type: ignore[assignment]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if np is not None:
        if isinstance(obj, np.ndarray):
            return [to_jsonable(v) for v in obj.tolist()]
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
    return obj


def write_jsonl(path: Union[str, Path], records: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(to_jsonable(rec), ensure_ascii=False) + "\n")


def _validate_obb(obb: Any, tag: str) -> List[float]:
    if not isinstance(obb, (list, tuple)) or len(obb) != 5:
        raise SchemaError(f"{tag}: 'obb' must be a 5-element [cx,cy,w,h,theta] list")
    try:
        vals = [float(v) for v in obb]
    except (TypeError, ValueError) as e:
        raise SchemaError(f"{tag}: 'obb' values must be numeric ({e})") from e
    if vals[2] < 0 or vals[3] < 0:
        raise SchemaError(f"{tag}: 'obb' w/h must be non-negative, got {vals[2:4]}")
    return vals


def validate_detection(rec: Dict[str, Any], tag: str = "detection") -> Dict[str, Any]:
    for field in ("image_id", "class", "obb", "score"):
        if field not in rec:
            raise SchemaError(f"{tag}: missing required field '{field}'")
    obb = _validate_obb(rec["obb"], tag)
    try:
        score = float(rec["score"])
    except (TypeError, ValueError) as e:
        raise SchemaError(f"{tag}: 'score' must be numeric ({e})") from e
    return {
        "image_id": str(rec["image_id"]),
        "scene_id": str(rec["scene_id"]) if rec.get("scene_id") is not None else None,
        "class": str(rec["class"]),
        "obb": obb,
        "score": score,
    }


def validate_gt(rec: Dict[str, Any], tag: str = "gt") -> Dict[str, Any]:
    for field in ("image_id", "class", "obb"):
        if field not in rec:
            raise SchemaError(f"{tag}: missing required field '{field}'")
    obb = _validate_obb(rec["obb"], tag)
    return {
        "image_id": str(rec["image_id"]),
        "scene_id": str(rec["scene_id"]) if rec.get("scene_id") is not None else None,
        "class": str(rec["class"]),
        "obb": obb,
        "gt_id": str(rec["gt_id"]) if rec.get("gt_id") is not None else None,
    }


def validate_detections(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not records:
        raise SchemaError("validate_detections: records must be non-empty")
    return [validate_detection(r, tag=f"detection[{i}]") for i, r in enumerate(records)]


def validate_gts(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not records:
        raise SchemaError("validate_gts: records must be non-empty")
    return [validate_gt(r, tag=f"gt[{i}]") for i, r in enumerate(records)]


def populate_scene_ids(
    records: List[Dict[str, Any]], scene_id_regex: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """Fill in ``scene_id`` from ``image_id`` (via
    :func:`rotcert.splits.scene_id_from_crop_filename`) for any record missing it.
    Records that already carry a ``scene_id`` are left untouched (an explicit
    upstream scene id always wins over the filename-convention guess)."""
    from rotcert.splits import DEFAULT_SCENE_ID_REGEX, scene_id_from_crop_filename

    regex = scene_id_regex or DEFAULT_SCENE_ID_REGEX
    out = []
    for r in records:
        r = dict(r)
        if not r.get("scene_id"):
            r["scene_id"] = scene_id_from_crop_filename(r["image_id"], regex)
        out.append(r)
    return out

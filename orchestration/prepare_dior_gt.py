#!/usr/bin/env python3
"""Prepare the canonical DIOR-R test GT JSONL from the VOC-style OBB xmls.

DIOR-R companion to :mod:`prepare_dota_gt` (whose output schema this mirrors
EXACTLY): reads every ``*.xml`` under ``--annfiles-dir`` (the staged
``Annotations/Oriented Bounding Boxes`` dir -- one xml per image), optionally
restricted to an ImageSet split via ``--imageset-file`` (so the test-split GT
lines up with the test-split detections), converts each object's oriented box
to a canonical ``[cx, cy, w, h, theta]`` le90 OBB via the repo's own
:func:`rotcert.gwd.canonicalize_le90` (so GT and detections share one angle
convention), and writes :func:`rotcert.io`-validated GT records
``{"image_id", "class", "obb", "gt_id"}`` (``scene_id`` is populated
downstream from the image id, same as detections).

DIOR-R release drift -- two on-disk oriented-box encodings exist, and this
script HANDLES BOTH and REFUSES LOUDLY on anything else:

  * **5-parameter** ``<robndbox>`` with ``<cx><cy><w><h><angle>`` children.
    ``<angle>`` is taken as radians by default (``--angle-unit``, env
    ``DIOR_ANGLE_UNIT``); canonicalize_le90 wraps any range into le90 so the
    exact period does not matter, only the unit.
  * **8-point polygon**: the four corners, either as ``<robndbox>`` corner
    children (``x_left_top,y_left_top,x_right_top,...``) or a ``<polygon>``
    with ``x1..y4`` / ``<point>`` children -> ``cv2.minAreaRect`` ->
    canonicalize_le90 (identical path to :func:`prepare_dota_gt.poly_to_le90_obb`).

An ``<object>`` that matches NEITHER (e.g. an axis-aligned ``<bndbox>``-only
HBB annotation -- the WRONG input dir) raises :class:`UnknownDiorSchema` and
aborts the whole run: silently dropping real GT would corrupt every downstream
coverage/recall number. Per-instance geometric degeneracy (a corner-collapsed
polygon that ``cv2.minAreaRect`` cannot orient) is instead skipped and COUNTED,
same as the DOTA donor.

Difficult flag: ``--difficult-policy`` (env ``DIOR_DIFFICULT_POLICY``, named
default ``include``) keeps or drops ``<difficult>1</difficult>`` instances;
counts for both are recorded in ``<out>.provenance.json``.

Box-side, CPU-only, needs numpy + rotcert (+ cv2 only for the 8-point path).
No GPU, no mmrotate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "UnknownDiorSchema",
    "parse_dior_xml",
    "robndbox5_to_le90_obb",
    "poly_to_le90_obb",
    "main",
]

_DIFFICULT_POLICIES = ("include", "drop")
_ANGLE_UNITS = ("radians", "degrees")

# 8-point corner child names, in (x, y) pairs, tried in order. The first naming
# convention is the DIOR-R <robndbox> corner form; the second the <polygon> form.
_CORNER_NAME_SETS: Tuple[Tuple[str, ...], ...] = (
    ("x_left_top", "y_left_top", "x_right_top", "y_right_top",
     "x_right_bottom", "y_right_bottom", "x_left_bottom", "y_left_bottom"),
    ("x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"),
)


class UnknownDiorSchema(Exception):
    """Raised when an <object> carries no recognizable oriented-box encoding.

    A hard refusal (never a silent skip): an unrecognized schema means the
    parser is pointed at the wrong annotation format, and dropping the object
    would silently under-count GT."""


def robndbox5_to_le90_obb(cx: float, cy: float, w: float, h: float,
                          angle: float, angle_unit: str = "radians") -> List[float]:
    """5-param oriented box -> canonical le90 ``[cx, cy, w, h, theta]``.

    ``angle`` is interpreted per ``angle_unit`` (radians|degrees) then handed to
    :func:`rotcert.gwd.canonicalize_le90`, which wraps any range into le90."""
    from rotcert.gwd import canonicalize_le90

    theta = math.radians(float(angle)) if angle_unit == "degrees" else float(angle)
    w2, h2, t2 = canonicalize_le90(float(w), float(h), theta)
    return [float(cx), float(cy), float(w2), float(h2), float(t2)]


def poly_to_le90_obb(poly: "np.ndarray") -> List[float]:
    """8-point polygon -> canonical le90 ``[cx, cy, w, h, theta]`` (cv2.minAreaRect).

    Identical geometry path to :func:`prepare_dota_gt.poly_to_le90_obb` so DIOR-R
    polygon GT and DOTA polygon GT are canonicalized the same way."""
    import cv2  # lazy: only the 8-point path needs it

    from rotcert.gwd import canonicalize_le90

    pts = np.asarray(poly, dtype=np.float32).reshape(4, 2)
    (cx, cy), (w, h), angle_deg = cv2.minAreaRect(pts)
    w2, h2, theta = canonicalize_le90(float(w), float(h), math.radians(float(angle_deg)))
    return [float(cx), float(cy), float(w2), float(h2), float(theta)]


def _floats_or_none(elem: Optional[ET.Element], names: Tuple[str, ...]) -> Optional[List[float]]:
    """Return ``[float(elem.findtext(n)) for n in names]`` iff ALL are present and
    numeric, else ``None`` (so the caller can try the next encoding)."""
    if elem is None:
        return None
    out: List[float] = []
    for n in names:
        txt = elem.findtext(n)
        if txt is None:
            return None
        try:
            out.append(float(txt))
        except (TypeError, ValueError):
            return None
    return out


def parse_dior_xml(path: Path) -> List[Dict[str, Any]]:
    """One DIOR-R xml -> raw rows ``{kind, coords, class, difficult}``.

    ``kind`` is ``"robndbox5"`` (``coords`` = ``[cx,cy,w,h,angle]``) or ``"poly8"``
    (``coords`` = 8 polygon floats). Raises :class:`UnknownDiorSchema` for an
    ``<object>`` with no recognizable oriented box."""
    rows: List[Dict[str, Any]] = []
    root = ET.parse(str(path)).getroot()
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name is None:
            raise UnknownDiorSchema(f"{path.name}: <object> without <name>")
        # DIOR-R xmls capitalize two classes (Expressway-Service-area,
        # Expressway-toll-station) while the detection dumps carry mmrotate's
        # all-lowercase DIORDataset.METAINFO names; the matcher is class-exact,
        # so the mismatch silently zeroes those classes' TPs (2026-07-11 ORCNN
        # cert: 18/20 instead of 20/20). Normalize GT to the METAINFO casing.
        name = name.strip().lower()
        difficult = obj.findtext("difficult")
        difficult = int(difficult) if difficult not in (None, "") else 0

        rb = obj.find("robndbox")
        # (1) 5-param robndbox
        five = _floats_or_none(rb, ("cx", "cy", "w", "h", "angle"))
        if five is not None:
            rows.append({"kind": "robndbox5", "coords": five,
                         "class": name, "difficult": difficult})
            continue
        # (2) 8-point corners, from robndbox or polygon or the object itself
        poly = None
        for container in (rb, obj.find("polygon"), obj):
            for name_set in _CORNER_NAME_SETS:
                cand = _floats_or_none(container, name_set)
                if cand is not None:
                    poly = cand
                    break
            if poly is not None:
                break
        if poly is not None:
            rows.append({"kind": "poly8", "coords": poly,
                         "class": name, "difficult": difficult})
            continue
        # (3) nothing recognizable -> refuse loudly
        raise UnknownDiorSchema(
            f"{path.name}: <object name='{name}'> has no <robndbox> 5-param, "
            f"no 8-corner points, and no <polygon> -- unrecognized DIOR-R OBB "
            f"schema (an axis-aligned <bndbox>-only HBB annotation would land "
            f"here; point --annfiles-dir at the Oriented Bounding Boxes dir)."
        )
    return rows


def _load_imageset(imageset_file: Optional[str]) -> Optional[set]:
    if not imageset_file:
        return None
    ids = set()
    for line in Path(imageset_file).read_text(encoding="utf-8").splitlines():
        stem = line.strip().split()[0] if line.strip() else ""
        if stem:
            ids.add(stem)
    return ids


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="DIOR-R OBB xml -> canonical GT JSONL (mirrors prepare_dota_gt)")
    ap.add_argument("--annfiles-dir", required=True, help="DIOR-R 'Annotations/Oriented Bounding Boxes' dir (one .xml per image)")
    ap.add_argument("--imageset-file", default=os.environ.get("DIOR_IMAGESET_FILE", ""),
                    help="optional ImageSets/Main split list (e.g. test.txt); restricts xmls to these stems")
    ap.add_argument("--difficult-policy", default=os.environ.get("DIOR_DIFFICULT_POLICY", "include"),
                    choices=_DIFFICULT_POLICIES)
    ap.add_argument("--angle-unit", default=os.environ.get("DIOR_ANGLE_UNIT", "radians"),
                    choices=_ANGLE_UNITS, help="unit of <robndbox><angle> (canonicalize_le90 wraps any range)")
    ap.add_argument("-o", "--out", required=True, help="canonical GT JSONL output")
    args = ap.parse_args(argv)

    from rotcert.io import validate_gts, write_jsonl

    ann_dir = Path(args.annfiles_dir)
    files = sorted(ann_dir.glob("*.xml"))
    if not files:
        print(f"error: no .xml annfiles under {ann_dir}", file=sys.stderr)
        return 1

    keep_ids = _load_imageset(args.imageset_file or None)
    if keep_ids is not None:
        files = [f for f in files if f.stem in keep_ids]
        if not files:
            print(f"error: no .xml annfiles under {ann_dir} match imageset {args.imageset_file}", file=sys.stderr)
            return 1

    records: List[Dict[str, Any]] = []
    n_difficult = n_dropped = n_degenerate = 0
    n_robndbox5 = n_poly8 = 0
    for f in files:
        image_id = f.stem
        for i, row in enumerate(parse_dior_xml(f)):  # UnknownDiorSchema propagates (hard refuse)
            if row["difficult"]:
                n_difficult += 1
                if args.difficult_policy == "drop":
                    n_dropped += 1
                    continue
            try:
                if row["kind"] == "robndbox5":
                    cx, cy, w, h, angle = row["coords"]
                    obb = robndbox5_to_le90_obb(cx, cy, w, h, angle, args.angle_unit)
                    n_robndbox5 += 1
                else:
                    obb = poly_to_le90_obb(np.asarray(row["coords"], dtype=np.float32))
                    n_poly8 += 1
            except Exception as e:  # degenerate box geometry -- skip + count (not a schema error)
                n_degenerate += 1
                print(f"warning: {f.name} object {i}: {e} -- skipped", file=sys.stderr)
                continue
            records.append(
                {"image_id": image_id, "class": row["class"], "obb": obb, "gt_id": f"{image_id}:{i}"}
            )

    validated = validate_gts(records)
    write_jsonl(args.out, validated)

    prov = {
        "annfiles_dir": str(ann_dir),
        "imageset_file": args.imageset_file or None,
        "n_annfiles": len(files),
        "n_gt_records": len(validated),
        "n_robndbox5": n_robndbox5,
        "n_poly8": n_poly8,
        "n_difficult_seen": n_difficult,
        "difficult_policy": args.difficult_policy,
        "n_difficult_dropped": n_dropped,
        "n_degenerate_skipped": n_degenerate,
        "angle_unit": args.angle_unit,
        "angle_convention": "le90 (rotcert.gwd.canonicalize_le90)",
    }
    with open(f"{args.out}.provenance.json", "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2)
    print(
        f"prepare_dior_gt: {len(validated)} GT records from {len(files)} xmls "
        f"(robndbox5={n_robndbox5}, poly8={n_poly8}, difficult seen={n_difficult}, "
        f"dropped={n_dropped}, degenerate skipped={n_degenerate}) -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

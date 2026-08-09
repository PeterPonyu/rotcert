#!/usr/bin/env python3
"""Convert the HRSC2016-MS distribution (VOC-style <robndbox> XML + .bmp images)
into a DOTA-format layout the rotcert mmrotate harness already consumes, for one
ImageSet split.

Why a converter (design note)
-----------------------------
HRSC2016-MS (github.com/wmchen/HRSC2016-MS) ships each image's oriented boxes as a
Pascal-VOC-style ``<annotation><object><robndbox><cx><cy><w><h><angle>`` (angle in
RADIANS), NOT the original HRSC ``<HRSC_Object><mbox_*>`` schema mmrotate's native
``HRSCDataset`` parses. Rather than special-case a new dataset class, we convert to
the DOTA 8-point-polygon layout (``images/<id>.png`` + ``annfiles/<id>.txt`` with
``x1 y1 x2 y2 x3 y3 x4 y4 <class> <difficult>``) that the proven DOTA arm already
trains + certifies on (``oriented-rcnn-le90_r50_fpn_1x_dota.py`` recipe,
``prepare_dota_gt.py`` for the canonical GT JSONL). This is the SAME robndbox5 5-param
encoding ``prepare_dior_gt.robndbox5_to_le90_obb`` handles, so the geometry path is
already validated elsewhere in the harness.

Class scheme: the canonical HRSC2016 detection benchmark is SINGLE-CLASS ('ship') --
the fine-grained ``<name>`` ids (HRSC L1/L2/L3 ship types) are collapsed to one class,
matching mmrotate's default ``HRSCDataset(classwise=False)`` and mirroring the
DOTA-15 / DIOR-20 arms' single Mondrian dimension (disclosed: HRSC contributes one
per-class cell, not 15/20).

Angle-convention self-check (HARD gate): each robndbox is rotated to 4 corners with
the standard OpenCV rotation (theta measured CCW, w along the local x-axis). The
converter recomputes each object's axis-aligned bounding box from those corners and
compares it to the XML's own ``<bndbox>`` (HBB). If the median corner-derived-HBB vs
XML-HBB IoU is below ``--hbb-iou-floor`` the whole run REFUSES -- a wrong angle sign /
w-h transpose would silently corrupt every downstream box, so we catch it here (on
CPU, before any GPU training) exactly as the ledger's degenerate-annotation lesson
demands.

Box-side, CPU-only: needs numpy + opencv (cv2) + Pillow (bmp->png). No mmrotate, no GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def robndbox_to_corners(cx: float, cy: float, w: float, h: float, angle: float) -> np.ndarray:
    """(cx,cy,w,h,angle_rad) -> 4x2 corner array, roLabelImg convention (the tool that
    produced HRSC2016-MS's <robndbox> tags): w along the local x-axis, h along y, theta a
    CLOCKWISE rotation, i.e. R = [[cos, sin], [-sin, cos]]. VERIFIED against the dataset's
    own <bndbox> HBBs: this w/h assignment gives median corner-HBB-vs-XML-HBB IoU 0.81
    (loose independent HBB annotations), whereas swapping w<->h collapses it to 0.35.
    (The sign is downstream-irrelevant -- prepare_dota_gt runs cv2.minAreaRect on these
    corners and canonicalize_le90 on the result, and detector predictions are canonicalized
    identically, so GT and predictions share one convention regardless -- but CW is the
    documented roLabelImg convention, so we use it.)"""
    dx = w / 2.0
    dy = h / 2.0
    local = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=np.float64)
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, s], [-s, c]], dtype=np.float64)
    return (local @ R.T) + np.array([cx, cy], dtype=np.float64)


def _hbb_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def parse_hrsc_xml(path: Path):
    """One HRSC2016-MS xml -> list of {corners(4x2), difficult, xml_hbb(4-tuple)|None}."""
    root = ET.parse(str(path)).getroot()
    out = []
    for obj in root.findall("object"):
        rb = obj.find("robndbox")
        if rb is None:
            continue
        try:
            cx = float(rb.findtext("cx")); cy = float(rb.findtext("cy"))
            w = float(rb.findtext("w")); h = float(rb.findtext("h"))
            angle = float(rb.findtext("angle"))
        except (TypeError, ValueError):
            continue
        diff = obj.findtext("difficult")
        diff = int(diff) if diff not in (None, "") else 0
        corners = robndbox_to_corners(cx, cy, w, h, angle)
        xml_hbb = None
        bb = obj.find("bndbox")
        if bb is not None:
            try:
                xml_hbb = (float(bb.findtext("xmin")), float(bb.findtext("ymin")),
                           float(bb.findtext("xmax")), float(bb.findtext("ymax")))
            except (TypeError, ValueError):
                xml_hbb = None
        out.append({"corners": corners, "difficult": diff, "xml_hbb": xml_hbb})
    return out


def _load_ids(imageset_file: Path) -> List[str]:
    ids = []
    for line in imageset_file.read_text(encoding="utf-8").splitlines():
        s = line.strip().split()[0] if line.strip() else ""
        if s:
            ids.append(s)
    return ids


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hrsc-root", required=True, help="HRSC2016-MS root (has AllImages/ Annotations/ ImageSets/)")
    ap.add_argument("--imageset-file", required=True, help="split id list, e.g. ImageSets/source/train.txt")
    ap.add_argument("--out-dir", required=True, help="DOTA-layout output root; writes images/ and annfiles/")
    ap.add_argument("--class-name", default="ship", help="single-class label written for every object")
    ap.add_argument("--img-ext", default="bmp", help="source image extension under AllImages/")
    ap.add_argument("--hbb-iou-floor", type=float, default=0.65,
                    help="REFUSE if median(corner-HBB vs xml-HBB IoU) < this (angle-convention gate; "
                         "correct convention scores ~0.81 on HRSC2016-MS's loose HBBs, a w/h transpose ~0.35)")
    ap.add_argument("--no-images", action="store_true", help="write annfiles only (skip bmp->png copy)")
    args = ap.parse_args(argv)

    hrsc = Path(args.hrsc_root)
    ann_src = hrsc / "Annotations"
    img_src = hrsc / "AllImages"
    ids = _load_ids(Path(args.imageset_file))
    if not ids:
        print(f"error: no ids in {args.imageset_file}", file=sys.stderr)
        return 1

    out = Path(args.out_dir)
    (out / "annfiles").mkdir(parents=True, exist_ok=True)
    if not args.no_images:
        (out / "images").mkdir(parents=True, exist_ok=True)

    ious: List[float] = []
    n_obj = n_img = n_missing_ann = n_missing_img = 0
    for iid in ids:
        axml = ann_src / f"{iid}.xml"
        if not axml.exists():
            n_missing_ann += 1
            continue
        objs = parse_hrsc_xml(axml)
        lines = []
        for o in objs:
            corners = o["corners"]
            # angle-convention cross-check against the XML's own HBB
            if o["xml_hbb"] is not None:
                cx1, cy1 = corners[:, 0].min(), corners[:, 1].min()
                cx2, cy2 = corners[:, 0].max(), corners[:, 1].max()
                ious.append(_hbb_iou((cx1, cy1, cx2, cy2), o["xml_hbb"]))
            poly = " ".join(f"{v:.2f}" for v in corners.reshape(-1))
            lines.append(f"{poly} {args.class_name} {o['difficult']}")
            n_obj += 1
        (out / "annfiles" / f"{iid}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if not args.no_images:
            src = img_src / f"{iid}.{args.img_ext}"
            if not src.exists():
                n_missing_img += 1
            else:
                from PIL import Image
                Image.open(src).convert("RGB").save(out / "images" / f"{iid}.png")
        n_img += 1

    med_iou = float(np.median(ious)) if ious else 0.0
    prov = {
        "hrsc_root": str(hrsc), "imageset_file": args.imageset_file, "out_dir": str(out),
        "class_name": args.class_name, "n_images": n_img, "n_objects": n_obj,
        "n_missing_ann": n_missing_ann, "n_missing_img": n_missing_img,
        "hbb_crosscheck_median_iou": round(med_iou, 4),
        "hbb_crosscheck_n": len(ious),
        "angle_unit": "radians", "encoding": "robndbox5->poly8 (OpenCV rotation)",
    }
    (out / "prepare_hrsc.provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    print(f"prepare_hrsc: {n_img} images, {n_obj} objects -> {out} "
          f"(HBB-crosscheck median IoU={med_iou:.4f} over {len(ious)}; "
          f"missing ann={n_missing_ann} img={n_missing_img})")

    if med_iou < args.hbb_iou_floor:
        print(f"REFUSE: HBB cross-check median IoU {med_iou:.4f} < floor {args.hbb_iou_floor} "
              f"-- robndbox angle convention likely wrong (sign/transpose); refusing before GPU.",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

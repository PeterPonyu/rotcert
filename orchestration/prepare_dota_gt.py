#!/usr/bin/env python3
"""Prepare the canonical DOTA val GT JSONL from mmrotate split annfiles.

Fills the gap next_boot_rotcert.sh assumes away: ``DOTA_VAL_GT`` is
"box-side-prepared" but nothing prepared it until now. Reads every
``*.txt`` under ``--annfiles-dir`` (img_split output: one file per crop,
lines ``x1 y1 x2 y2 x3 y3 x4 y4 class difficult``; original-DOTA header
lines like ``imagesource:``/``gsd:`` are skipped), converts each polygon
to a ``[cx, cy, w, h, theta]`` OBB via ``cv2.minAreaRect`` and the repo's
own :func:`rotcert.gwd.canonicalize_le90` (so GT and detections share one
angle convention), and writes :func:`rotcert.io`-validated GT records:
``{"image_id", "class", "obb", "gt_id"}`` (scene_id is populated
downstream from the crop filename, same as detections).

Difficult flag: ``--difficult-policy`` (env ``DOTA_DIFFICULT_POLICY``,
named default ``include``) either keeps (``include``) or drops (``drop``)
``difficult == 1`` instances; counts for BOTH are recorded in the sidecar
``<out>.provenance.json`` so the prereg can cite the realized numbers
whichever way it freezes the policy.

Box-side, CPU-only, needs cv2 + numpy + rotcert installed (run inside the
mm env). No GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = ["parse_annfile", "poly_to_le90_obb", "main"]

_DIFFICULT_POLICIES = ("include", "drop")


def poly_to_le90_obb(poly: "np.ndarray") -> List[float]:
    """8-point DOTA polygon -> canonical le90 ``[cx, cy, w, h, theta]``.

    cv2.minAreaRect gives ((cx, cy), (w, h), angle_deg); the repo's
    canonicalize_le90 then maps (w, h, theta) to the long-edge form every
    rotcert scoring function uses (w >= h, theta in (-pi/2, pi/2]).
    """
    import cv2  # lazy: box-side dependency, tests may monkeypatch

    from rotcert.gwd import canonicalize_le90

    pts = np.asarray(poly, dtype=np.float32).reshape(4, 2)
    (cx, cy), (w, h), angle_deg = cv2.minAreaRect(pts)
    w2, h2, theta = canonicalize_le90(float(w), float(h), math.radians(float(angle_deg)))
    return [float(cx), float(cy), float(w2), float(h2), float(theta)]


def parse_annfile(path: Path) -> List[Dict[str, Any]]:
    """One split annfile -> raw rows ``{poly, class, difficult}`` (headers skipped)."""
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 10:
            continue  # header ('imagesource:...', 'gsd:...') or blank
        try:
            poly = [float(v) for v in parts[:8]]
            difficult = int(parts[9])
        except ValueError:
            continue
        rows.append({"poly": poly, "class": parts[8], "difficult": difficult})
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annfiles-dir", required=True, help="img_split val annfiles dir (one .txt per crop)")
    ap.add_argument(
        "--difficult-policy",
        default=os.environ.get("DOTA_DIFFICULT_POLICY", "include"),
        choices=_DIFFICULT_POLICIES,
    )
    ap.add_argument("-o", "--out", required=True, help="canonical GT JSONL output")
    args = ap.parse_args(argv)

    from rotcert.io import validate_gts, write_jsonl

    ann_dir = Path(args.annfiles_dir)
    files = sorted(ann_dir.glob("*.txt"))
    if not files:
        print(f"error: no .txt annfiles under {ann_dir}", file=sys.stderr)
        return 1

    records: List[Dict[str, Any]] = []
    n_difficult = n_dropped = n_parse_skipped = 0
    for f in files:
        image_id = f.stem
        for i, row in enumerate(parse_annfile(f)):
            if row["difficult"]:
                n_difficult += 1
                if args.difficult_policy == "drop":
                    n_dropped += 1
                    continue
            try:
                obb = poly_to_le90_obb(row["poly"])
            except Exception as e:  # degenerate polygon after crop-clipping
                n_parse_skipped += 1
                print(f"warning: {f.name} line {i}: {e} -- skipped", file=sys.stderr)
                continue
            records.append(
                {"image_id": image_id, "class": row["class"], "obb": obb, "gt_id": f"{image_id}:{i}"}
            )

    validated = validate_gts(records)
    write_jsonl(args.out, validated)

    prov = {
        "annfiles_dir": str(ann_dir),
        "n_annfiles": len(files),
        "n_gt_records": len(validated),
        "n_difficult_seen": n_difficult,
        "difficult_policy": args.difficult_policy,
        "n_difficult_dropped": n_dropped,
        "n_degenerate_skipped": n_parse_skipped,
        "angle_convention": "le90 (rotcert.gwd.canonicalize_le90)",
    }
    with open(f"{args.out}.provenance.json", "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2)
    print(
        f"prepare_dota_gt: {len(validated)} GT records from {len(files)} annfiles "
        f"(difficult seen={n_difficult}, dropped={n_dropped}, degenerate skipped={n_parse_skipped}) -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

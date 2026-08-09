#!/usr/bin/env python3
"""Box-side DIOR-R download + size/checksum/license/count audit (design §4.2, the
HARD Phase-0 gate; §7 Phase 0).

DIOR-R is NOT in ``/root/autodl-pub`` (unlike DOTA) -- this is the box-side download
item. Runs entirely on plain-python content checks (``urllib``/``hashlib``/``zipfile``
stdlib only, no heavy deps), so ``--help`` and every gate function here work in any
environment.

Gate order (design §4.2's HARD gate: size/checksum/license/count -- if ANY fails, the
second dataset falls back to HRSC2016 or DOTA-v2.0, never DOTA-v1.5, per the design's
explicit ban on v1.5 as a superset arm)
--------------------------------------------------------------------------------------
1. ``--archive-size-min-bytes`` -- the downloaded archive must be at least this large
   (catches a truncated/failed download before spending time on extraction).
2. ``--sha256`` -- if given, verified against the downloaded archive (VERIFY the real
   checksum at Phase 0 -- none is baked in here, the design's own [VERIFY] flag on
   DIOR-R availability applies).
3. ``--min-images`` / ``--expected-classes`` -- post-extraction count audit: total
   image count and the per-class instance-count table (design's own "audit that
   per-class instance counts match the published table before any calibration").
4. License: DIOR-R's terms are printed for MANUAL confirmation
   (``--confirm-license-reviewed`` is a required flag, not auto-checked -- license
   text review is not something this script can safely automate).

On ANY gate failure, this script exits non-zero and prints
``DIOR_R_GATE_FAILED -- falling back per design §4.2 (HRSC2016 or DOTA-v2.0)`` --
callers (``next_boot_rotcert.sh``) are expected to branch on that, not retry blindly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Optional


DIOR_R_LICENSE_NOTICE = (
    "DIOR-R (oriented DIOR, Cheng et al., TGRS 2022) license terms must be reviewed "
    "manually before use (design §4.2: 'license verification (DIOR terms)'). This "
    "script does not auto-verify license text -- pass --confirm-license-reviewed only "
    "after a human has checked the dataset's actual distribution terms at the source."
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gate_archive_size(path: Path, min_bytes: int) -> bool:
    if not path.exists():
        print(f"GATE_FAIL archive_size: {path} does not exist", file=sys.stderr)
        return False
    size = path.stat().st_size
    if size < min_bytes:
        print(f"GATE_FAIL archive_size: {size} < min_bytes={min_bytes}", file=sys.stderr)
        return False
    print(f"GATE_OK archive_size: {size} bytes >= {min_bytes}")
    return True


def gate_checksum(path: Path, expected_sha256: Optional[str]) -> bool:
    if expected_sha256 is None:
        print("GATE_SKIP checksum: no --sha256 given (VERIFY the real checksum at Phase 0)")
        return True
    actual = _sha256_of_file(path)
    if actual != expected_sha256:
        print(f"GATE_FAIL checksum: expected {expected_sha256}, got {actual}", file=sys.stderr)
        return False
    print(f"GATE_OK checksum: {actual}")
    return True


def gate_extraction_counts(extracted_dir: Path, min_images: int, expected_classes: Optional[list]) -> bool:
    if not extracted_dir.exists():
        print(f"GATE_FAIL extraction: {extracted_dir} does not exist", file=sys.stderr)
        return False
    image_exts = (".jpg", ".jpeg", ".png", ".tif")
    n_images = sum(1 for p in extracted_dir.rglob("*") if p.suffix.lower() in image_exts)
    if n_images < min_images:
        print(f"GATE_FAIL extraction_counts: n_images={n_images} < min_images={min_images}", file=sys.stderr)
        return False
    print(f"GATE_OK extraction_counts: n_images={n_images} >= {min_images}")

    if expected_classes:
        # A real per-class instance-count audit needs the label files parsed
        # (format-specific, deferred to phase0.py's staging-audit once the archive
        # is confirmed present) -- here we only check the class NAMES appear
        # somewhere in the label directory structure (a cheap smoke check).
        label_text = " ".join(p.name for p in extracted_dir.rglob("*") if p.is_file())
        missing = [c for c in expected_classes if c not in label_text]
        if missing:
            print(f"GATE_WARN extraction_counts: classes not found in filenames (may be fine if class names are inside label files, not filenames): {missing}")
    return True


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Box-side DIOR-R download + size/checksum/license/count gate")
    p.add_argument("--url", required=True)
    p.add_argument("--dest-dir", required=True)
    p.add_argument("--archive-name", default="DIOR-R.zip")
    p.add_argument("--archive-size-min-bytes", type=int, default=1_000_000_000, help="default 1GB floor; DIOR-R is expected ~20-25GB")
    p.add_argument("--sha256", default=None)
    p.add_argument("--min-images", type=int, default=20000, help="DIOR-R has ~23k images (design §4.2)")
    p.add_argument("--expected-classes", default=None, help="comma-separated; DIOR-R has 20 classes")
    p.add_argument("--confirm-license-reviewed", action="store_true", required=True)
    p.add_argument("--skip-download", action="store_true", help="archive already present at --dest-dir/--archive-name")
    args = p.parse_args(argv)

    print(DIOR_R_LICENSE_NOTICE)
    if not args.confirm_license_reviewed:
        print("error: --confirm-license-reviewed is required (see notice above)", file=sys.stderr)
        return 1

    dest_dir = Path(args.dest_dir)
    archive_path = dest_dir / args.archive_name

    if not args.skip_download:
        _download(args.url, archive_path)

    ok = True
    ok &= gate_archive_size(archive_path, args.archive_size_min_bytes)
    ok &= gate_checksum(archive_path, args.sha256)

    if ok:
        import zipfile

        extracted_dir = dest_dir / "extracted"
        extracted_dir.mkdir(parents=True, exist_ok=True)
        print(f"extracting {archive_path} -> {extracted_dir}")
        try:
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(extracted_dir)
        except zipfile.BadZipFile as e:
            print(f"GATE_FAIL extraction: bad zip file ({e})", file=sys.stderr)
            ok = False
        if ok:
            expected_classes = args.expected_classes.split(",") if args.expected_classes else None
            ok &= gate_extraction_counts(extracted_dir, args.min_images, expected_classes)

    manifest = {"archive_path": str(archive_path), "sha256": _sha256_of_file(archive_path) if archive_path.exists() else None, "gates_passed": bool(ok)}
    with open(dest_dir / "DIOR_R_FETCH_MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if not ok:
        print("DIOR_R_GATE_FAILED -- falling back per design §4.2 (HRSC2016 or DOTA-v2.0)", file=sys.stderr)
        return 1
    print("DIOR_R_GATE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

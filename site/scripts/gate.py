#!/usr/bin/env python3
"""GATE-LOCAL checks for the RotCert Hugo companion (site/public)."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
PUBLIC = ROOT / "public"
STATIC = ROOT / "static"
ALLOWLIST = ROOT / "FIGURE_ALLOWLIST.txt"
DATA = ROOT / "data"

FORBIDDEN = [
    "BigEarthNet",
    "Matbench",
    "MVTec",
    "isprs",
    "jcp",
    "F1_debt",
    "graphicspath",
    "AutoDL",
    "autodl",
    "geospatial-fm",
    "materials-mlip",
    "inspect-gate",
    "asr-gate",
    "Launch →",
    r"\\cite",
    r"\\ref",
    "certify.g1",
    "certify.g2",
    "/home/",
    "paper_pr.pdf",
    "paper_ieeetran.pdf",
    "Pattern Recognition",
    "TGRS",
    "this paper",
    "manuscript",
    "submission",
    "journal",
    "Elsevier",
    "not accepted",
    "methods companion",
    "PDF kit",
    "arxiv",
    "companion",
]

# The checks above only see the rendered site. This branch is a public repo, so
# every tracked file is browsable on GitHub whether or not the site links it.
# These markers are journal-submission packaging and internal review process,
# which must not sit on the public tip at all.
REPO_FORBIDDEN = [
    "paper_pr.pdf",
    "paper_pr.tex",
    "paper_ieeetran",
    "cover_letter",
    "DESK-GATE",
    "DESK-CHECK",
    "Editorial Manager",
    "Pattern Recognition",
    "Elsevier",
    "desk-reject",
    "desk return",
    "submission_source",
    "RED-TEAM-REPORT",
    "VENUE-FIT",
]

# "TGRS" alone is a legitimate citation for the DIOR-R dataset paper
# (Cheng et al., TGRS 2022), so it is matched only in a venue-targeting sense.
REPO_FORBIDDEN_RE = [
    re.compile(r"TGRS(?!\s+\d{4})[^.\n]{0,40}\b(?:kit|odds|reviewer|scale|grid|submission)\b", re.I),
]

REPO_SKIP_DIRS = {".git", "public", "resources", "node_modules", "__pycache__", ".venv"}
REPO_TEXT_SUFFIXES = {
    ".md", ".txt", ".py", ".sh", ".tex", ".bib", ".toml", ".yaml", ".yml",
    ".cff", ".json", ".html", ".css", ".js",
}

ROUTES = [
    "index.html",
    "geometry/index.html",
    "g1/index.html",
    "g2/index.html",
    "evidence/index.html",
    "scenes/index.html",
    "limits/index.html",
    "reproduce/index.html",
    "cite/index.html",
]

NUMBERS = ["20/20", "2/15", "7/20", "0/1"]

COMMENT = re.compile(r"<!--.*?-->", re.S)
JSON_NAME = re.compile(r"\b[\w.-]+\.jsonl?\b", re.I)
PREFIX = "/rotcert/"


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def check_allowlist() -> None:
    listed = {
        line.strip()
        for line in ALLOWLIST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    found: set[str] = set()
    for path in STATIC.rglob("*"):
        if path.is_file():
            found.add(path.relative_to(STATIC).as_posix())
    extra = found - listed
    missing = listed - found
    if extra:
        fail(f"static files not in allowlist: {sorted(extra)}")
    if missing:
        fail(f"allowlist entries missing on disk: {sorted(missing)}")
    print(f"ok: allowlist ({len(found)} static files)")


def check_pdfs() -> None:
    pdfs = list(PUBLIC.rglob("paper_*.pdf")) + list(PUBLIC.rglob("*.pdf"))
    if pdfs:
        fail(f"PDF in public/: {pdfs}")
    print("ok: no PDFs in public/")


def strip_comments(text: str) -> str:
    return COMMENT.sub("", text)


def check_leaks() -> None:
    files = list(PUBLIC.rglob("*.html")) + list(PUBLIC.rglob("*.svg"))
    if not files:
        fail("no rendered HTML/SVG")
    for path in files:
        raw = path.read_text(encoding="utf-8", errors="replace")
        body = strip_comments(raw)
        lower = body.lower()
        for token in FORBIDDEN:
            if token.startswith(r"\\"):
                if token[1:] in body:
                    fail(f"{path}: forbidden {token}")
            elif token.lower() in lower:
                fail(f"{path}: forbidden {token}")
        if JSON_NAME.search(body):
            fail(f"{path}: json filename in rendered prose")
    print(f"ok: leak scan ({len(files)} files)")


def check_repo_surface() -> None:
    """Fail if submission/venue material reappears anywhere on the public tip."""
    problems: list[str] = []
    scanned = 0
    for path in REPO.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(REPO)
        if any(part in REPO_SKIP_DIRS for part in rel.parts):
            continue
        # A forbidden name in the path itself is a leak even for binaries.
        for token in REPO_FORBIDDEN:
            if token.lower() in rel.as_posix().lower():
                problems.append(f"{rel}: forbidden path token {token!r}")
        if path.suffix.lower() not in REPO_TEXT_SUFFIXES:
            continue
        if rel == Path("site/scripts/gate.py"):
            continue  # this file necessarily names what it blocks
        try:
            body = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        scanned += 1
        lower = body.lower()
        for token in REPO_FORBIDDEN:
            if token.lower() in lower:
                problems.append(f"{rel}: forbidden {token!r}")
        for pattern in REPO_FORBIDDEN_RE:
            m = pattern.search(body)
            if m:
                problems.append(f"{rel}: venue-targeting text {m.group(0)!r}")
    if problems:
        for p in sorted(set(problems)):
            print(f"  {p}", file=sys.stderr)
        fail(f"repo surface: {len(set(problems))} submission/venue leak(s) on the public tip")
    print(f"ok: repo surface ({scanned} text files)")


def check_routes_and_prefix() -> None:
    for rel in ROUTES:
        path = PUBLIC / rel
        if not path.is_file():
            fail(f"missing route {rel}")
        text = path.read_text(encoding="utf-8", errors="replace")
        if re.search(r"""(?:href|src)=["']?/(?:css|js|fonts)/""", text):
            fail(f"{rel}: unprefixed asset root")
        if re.search(r"url\(/fonts/", text):
            fail(f"{rel}: unprefixed font url")
        if re.search(r"href=/(?:[\s>]|$)", text) or 'href="/"' in text:
            fail(f"{rel}: unprefixed home href")
        if "/rotcert/rotcert/" in text:
            fail(f"{rel}: doubled path prefix")
        if PREFIX not in text:
            fail(f"{rel}: missing path prefix")
    print("ok: routes + prefix")


def check_numbers() -> None:
    home = (PUBLIC / "index.html").read_text(encoding="utf-8", errors="replace")
    for n in NUMBERS:
        if n not in home:
            fail(f"index.html missing glance number {n}")
    print("ok: glance numbers")


def extract_hash() -> None:
    required = [
        "glance.yaml",
        "g1_cells.yaml",
        "g2_cells.yaml",
        "dior_classes.yaml",
        "dota_actionable.yaml",
        "g2_planning.yaml",
    ]
    digest = hashlib.sha256()
    for name in required:
        path = DATA / name
        if not path.is_file():
            fail(f"missing extract {name}")
        digest.update(path.read_bytes())
    print(f"ok: extract-hash {digest.hexdigest()[:16]}")


def main() -> None:
    if not PUBLIC.is_dir():
        fail("site/public missing; run hugo first")
    check_allowlist()
    check_pdfs()
    check_leaks()
    check_repo_surface()
    check_routes_and_prefix()
    check_numbers()
    extract_hash()
    print("GATE-LOCAL extra checks passed")


if __name__ == "__main__":
    main()

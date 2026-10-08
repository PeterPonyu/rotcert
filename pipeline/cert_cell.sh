#!/bin/bash
# cert_cell.sh -- the generic per-cell certification wrapper (2026-07-13).
#
# Runs the exact CPU chain that produced each DIOR/DOTA cell, on ONE (detector,dataset)
# cell, emitting the same file set a DIOR cell has:
#
#   match  --dets --gt --iou-thr 0.5 --iou-metric rotated  -> matched.jsonl
#   calibrate --score gwd         --mondrian --alpha 0.10   -> cert_gwd.json
#   calibrate --score naive-coord --mondrian --alpha 0.10   -> cert_naive-coord.json
#   calibrate --score hull        --mondrian --alpha 0.10   -> cert_hull.json
#   recall    --mondrian --beta 0.20 --delta 0.05           -> recall.json
#   audit     --score gwd         --alpha 0.10              -> audit_gwd.json
#   audit     --score naive-coord --alpha 0.10              -> audit_naive.json
#   r20_generic.py matched.jsonl                            -> r20_coverage.json
#
# There is NO `route` subcommand: routing/refusal is built into calibrate (Mondrian
# per-class refusal), recall (LTT-HB power-floor refusal) and certify (per-box
# "stratum not calibrated" refusal). This wrapper deliberately does not call `certify`
# (per-box region emission) -- a cell's certificate is the calibrate+recall+audit+r20
# set above; certify is applied later to fresh detections.
#
# Content gates (never bare exit codes): every step's output is checked non-empty +
# valid JSON before the next step. Markers under OUT/markers/. Idempotent: an existing
# integral output is reused (so a resumed cell does not recompute the slow match).
#
# Byte-repro (verified 2026-07-13): with --dets/--gt for DIOR-ORCNN-fixedGT this
# reproduces dior_cert_results_2026-07-11/orcnn_fixedgt/ exactly -- matched.jsonl,
# recall.json, audit_{gwd,naive}.json and r20_coverage.json are BYTE-IDENTICAL; the
# three cert_*.json differ only in the (intentionally non-deterministic) provenance
# `timestamp_utc` field. `--verify DIR` re-runs that diff (timestamp-insensitive).
#
# Usage:
#   cert_cell.sh --dets D.jsonl --gt G.jsonl --out-dir OUT [--iou-thr 0.5]
#                [--iou-metric rotated] [--python PY] [--r20-script PATH]
#                [--matched M.jsonl]   # skip match, use an existing matched.jsonl
#                [--verify FROZEN_DIR] # after building OUT, diff it vs FROZEN_DIR
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROTCERT_ROOT="$(cd "$HERE/.." && pwd)"
export ROTCERT_ROOT
export PYTHONPATH="$ROTCERT_ROOT:${PYTHONPATH:-}"

DETS=""; GT=""; OUT=""; MATCHED_IN=""; VERIFY=""
IOU_THR=0.5; IOU_METRIC=rotated
PY="${ROTCERT_PY:-$ROTCERT_ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY=python3
R20="$HERE/r20_generic.py"

while [ $# -gt 0 ]; do
  case "$1" in
    --dets) DETS="$2"; shift 2;;
    --gt) GT="$2"; shift 2;;
    --out-dir) OUT="$2"; shift 2;;
    --matched) MATCHED_IN="$2"; shift 2;;
    --iou-thr) IOU_THR="$2"; shift 2;;
    --iou-metric) IOU_METRIC="$2"; shift 2;;
    --python) PY="$2"; shift 2;;
    --r20-script) R20="$2"; shift 2;;
    --verify) VERIFY="$2"; shift 2;;
    -h|--help) grep -E '^# ' "$0" | sed 's/^# //'; exit 0;;
    *) echo "cert_cell: unknown arg '$1'" >&2; exit 2;;
  esac
done

[ -n "$OUT" ] || { echo "cert_cell: --out-dir required" >&2; exit 2; }
MK="$OUT/markers"; mkdir -p "$OUT" "$MK"
LOG="$OUT/cert_cell.log"; exec > >(tee -a "$LOG") 2>&1
echo "=== cert_cell start $(date -Iseconds) out=$OUT ==="
FAILED=()
mark(){ printf '%s\n' "$2" > "$MK/$1.marker"; echo "MARKER $1=$2"; [ "$2" = OK ] || FAILED+=("$1"); return 0; }

json_gate(){ # FILE -> ok iff exists, non-empty, every line parses as JSON
  "$PY" - "$1" <<'PY'
import json,sys
p=sys.argv[1]
try:
    txt=open(p,encoding="utf-8").read()
except FileNotFoundError:
    print(f"JSON_GATE_FAIL {p} missing",file=sys.stderr); sys.exit(1)
if not txt.strip():
    print(f"JSON_GATE_FAIL {p} empty",file=sys.stderr); sys.exit(1)
try:
    if p.endswith(".jsonl"):
        n=sum(1 for ln in txt.splitlines() if ln.strip() and json.loads(ln))
    else:
        json.loads(txt); n=1
except Exception as e:
    print(f"JSON_GATE_FAIL {p}: {e}",file=sys.stderr); sys.exit(1)
print(f"JSON_GATE_OK {p} n={n}"); sys.exit(0)
PY
}

# ---- step 1: match (or adopt an existing matched.jsonl) ----
MATCHED="$OUT/matched.jsonl"
if [ -n "$MATCHED_IN" ]; then
  echo "== step 1: adopt existing matched.jsonl ($MATCHED_IN) =="
  [ "$MATCHED_IN" -ef "$MATCHED" ] 2>/dev/null || cp "$MATCHED_IN" "$MATCHED"
  if json_gate "$MATCHED"; then mark MATCH OK; else mark MATCH FAILED; fi
elif [ -s "$MATCHED" ] && json_gate "$MATCHED" >/dev/null 2>&1; then
  echo "== step 1: reuse existing $MATCHED =="; mark MATCH OK
else
  [ -n "$DETS" ] && [ -n "$GT" ] || { echo "cert_cell: need --dets and --gt (or --matched)" >&2; exit 2; }
  echo "== step 1: match =="
  "$PY" -m rotcert.cli match --dets "$DETS" --gt "$GT" \
    --iou-thr "$IOU_THR" --iou-metric "$IOU_METRIC" -o "$MATCHED" \
    || echo "warn: match rc!=0 (gate authoritative)"
  if json_gate "$MATCHED"; then mark MATCH OK; else mark MATCH FAILED; fi
fi
[ "$(cat "$MK/MATCH.marker" 2>/dev/null)" = OK ] || { echo "REFUSE: match gate failed"; echo "CERT_CELL_ABORT"; exit 1; }

# ---- step 2: calibrate (G1) for the three scores, per-class Mondrian, alpha=0.10 ----
echo "== step 2: calibrate (gwd, naive-coord, hull; --mondrian --alpha 0.10) =="
calibrate(){ # SCORE OUTFILE MARKER
  "$PY" -m rotcert.cli calibrate --matched "$MATCHED" --score "$1" --mondrian --alpha 0.10 -o "$2" \
    || echo "warn: calibrate $1 rc!=0 (gate authoritative)"
  if json_gate "$2"; then mark "$3" OK; else mark "$3" FAILED; fi
}
calibrate gwd         "$OUT/cert_gwd.json"          CAL_GWD
calibrate naive-coord "$OUT/cert_naive-coord.json"  CAL_NAIVE
calibrate hull        "$OUT/cert_hull.json"         CAL_HULL

# ---- step 3: recall (G2) certified FNR, per-class Mondrian ----
echo "== step 3: recall (--mondrian --beta 0.20 --delta 0.05) =="
"$PY" -m rotcert.cli recall --matched "$MATCHED" --mondrian --beta 0.20 --delta 0.05 -o "$OUT/recall.json" \
  || echo "warn: recall rc!=0 (gate authoritative)"
if json_gate "$OUT/recall.json"; then mark RECALL OK; else mark RECALL FAILED; fi

# ---- step 4: audit (V1/V2 regime tables) for gwd + naive-coord ----
echo "== step 4: audit (gwd, naive-coord; --alpha 0.10) =="
"$PY" -m rotcert.cli audit --matched "$MATCHED" --score gwd         --alpha 0.10 -o "$OUT/audit_gwd.json" \
  || echo "warn: audit gwd rc!=0 (gate authoritative)"
if json_gate "$OUT/audit_gwd.json"; then mark AUDIT_GWD OK; else mark AUDIT_GWD FAILED; fi
"$PY" -m rotcert.cli audit --matched "$MATCHED" --score naive-coord --alpha 0.10 -o "$OUT/audit_naive.json" \
  || echo "warn: audit naive rc!=0 (gate authoritative)"
if json_gate "$OUT/audit_naive.json"; then mark AUDIT_NAIVE OK; else mark AUDIT_NAIVE FAILED; fi

# ---- step 5: R=20 out-of-sample scene-split coverage headline ----
echo "== step 5: r20_generic.py (OOS R=20 coverage) =="
"$PY" "$R20" "$MATCHED" "$OUT/r20_coverage.json" || echo "warn: r20 rc!=0 (gate authoritative)"
if json_gate "$OUT/r20_coverage.json"; then mark R20 OK; else mark R20 FAILED; fi

# ---- optional: byte-repro verification vs a frozen cell dir ----
# Two fields are intentionally environment-dependent and excluded from the diff:
#   "timestamp_utc" -- provenance wall-clock stamp (cert_*.json)
#   "matched"       -- the input matched.jsonl path recorded by r20 (r20_coverage.json)
# Everything else must be byte-identical.
if [ -n "$VERIFY" ]; then
  echo "== verify: diff $OUT vs $VERIFY (timestamp_utc + matched-path insensitive) =="
  vfail=0
  for f in matched.jsonl cert_gwd.json cert_naive-coord.json cert_hull.json recall.json audit_gwd.json audit_naive.json r20_coverage.json; do
    if [ ! -f "$VERIFY/$f" ]; then echo "VERIFY $f: SKIP (absent in frozen)"; continue; fi
    if cmp -s "$OUT/$f" "$VERIFY/$f"; then echo "VERIFY $f: BYTE-IDENTICAL"; else
      if diff <(grep -vE '"(timestamp_utc|matched)"' "$VERIFY/$f") <(grep -vE '"(timestamp_utc|matched)"' "$OUT/$f") >/dev/null 2>&1; then
        echo "VERIFY $f: IDENTICAL modulo provenance path/timestamp"
      else echo "VERIFY $f: DIFFERS"; vfail=1; fi
    fi
  done
  [ "$vfail" -eq 0 ] && echo "VERIFY_ALL_OK" || echo "VERIFY_MISMATCH"
fi

echo "=== cert_cell end $(date -Iseconds) ==="
if [ "${#FAILED[@]}" -eq 0 ]; then echo "CERT_CELL_DONE out=$OUT"; else echo "CERT_CELL_PARTIAL failed=[${FAILED[*]}]"; exit 1; fi

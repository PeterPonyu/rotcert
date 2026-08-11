#!/bin/bash
# next_boot_rotcert.sh -- rotcert box chain (design §7/§8): prologue -> stage DOTA
# from ${AUTODL_PUB} + DIOR-R box-side fetch -> Phase 0 (staging audit +
# reproduction gate) -> Phase-1 PILOT (1 detector x DOTA x 2 repeats, K1 metrics
# computed) -> epilogue. Marker: ROTCERT_PILOT_ALL_DONE.
#
# The FULL grid (2 detectors x 2 datasets x R=20 repeats x 6 scores, design §4.3)
# runs only behind REQUIRES_PREREG_FREEZE=confirmed -- this script REFUSES to run
# it otherwise (design §8: "Phase 2 ... prereg freeze ... after the K6 scoop
# re-scan; scale to all classes x 2 datasets x 2 detectors").
#
# Content-gated throughout (never a bare exit code, per the portfolio's DOFA
# lesson, boxkit's chain_lib.sh convention); pidfile-guarded background
# processes only (never `pkill -f <pattern>` matching the invocation itself).
#
# Usage (on the AutoDL box): bash next_boot_rotcert.sh
# (or: nohup bash next_boot_rotcert.sh > /root/rotcert_boot.log 2>&1 &)

set -uo pipefail  # deliberately NOT -e: later stages/markers/epilogue must
                   # still run after an earlier stage's content gate fails.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELIABILITY_COMMONS="${RELIABILITY_COMMONS:-/root/reliability-commons}"

# shellcheck disable=SC1091
source "${RELIABILITY_COMMONS}/tools/boxkit/chain_lib.sh"

# ---------------------------------------------------------------------------
# Tunables (env-overridable, named defaults -- no hardcoded paths/tolerances
# inline in the stages below).
# ---------------------------------------------------------------------------
RESULTS_DIR="${RESULTS_DIR:-${AUTODL_TMP}/rotcert_results}"
DOTA_SRC="${DOTA_SRC:-${AUTODL_PUB}/DOTA}"
DIOR_R_URL="${DIOR_R_URL:-}"           # required for the DIOR-R stage; empty = skip (disclosed)
DIOR_R_DEST="${DIOR_R_DEST:-${AUTODL_TMP}/dior_r}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-0}"

MMROTATE_COMMIT="${MMROTATE_COMMIT:-}"  # REQUIRED for any real detector run (no default pinned here)
RTMDET_R_CONFIG="${RTMDET_R_CONFIG:-}"
RTMDET_R_CHECKPOINT="${RTMDET_R_CHECKPOINT:-}"
DOTA_VAL_IMAGES="${DOTA_VAL_IMAGES:-$DOTA_SRC/val/images}"
DOTA_VAL_GT="${DOTA_VAL_GT:-$RESULTS_DIR/dota_val_gt.jsonl}"  # box-side-prepared canonical GT
DOTA_CLASS_NAMES="${DOTA_CLASS_NAMES:-plane,ship,storage-tank,baseball-diamond,tennis-court,basketball-court,ground-track-field,harbor,bridge,large-vehicle,small-vehicle,helicopter,roundabout,soccer-ball-field,swimming-pool}"

ROTCERT_REPRO_TOL="${ROTCERT_REPRO_TOL:-0.5}"
PUBLISHED_VAL_MAP="${PUBLISHED_VAL_MAP:-}"  # REQUIRED at Phase 0 (design K3: zoo-consensus VAL, no-TTA)

REQUIRES_PREREG_FREEZE="${REQUIRES_PREREG_FREEZE:-}"
PILOT_ALPHA="${PILOT_ALPHA:-0.10}"
PILOT_BETA="${PILOT_BETA:-0.20}"
PILOT_DELTA="${PILOT_DELTA:-0.05}"
PILOT_N_REPEATS="${PILOT_N_REPEATS:-2}"

MARKERS_DIR="${MARKERS_DIR:-$RESULTS_DIR/markers}"
mkdir -p "$RESULTS_DIR" "$MARKERS_DIR"

FAILED_MARKERS=()
step() { echo "== $1 =="; }
skip() { echo "skip (already exists): $1"; }
mark() {
  local name="$1" status="$2"
  printf '%s\n' "$status" > "$MARKERS_DIR/${name}.marker"
  echo "MARKER ${name}=${status}"
  [ "$status" = "FAILED" ] && FAILED_MARKERS+=("$name")
}

# ---------------------------------------------------------------------------
# Prologue (conda activate, HF_HOME, balance_guard, gpu logger -- boxkit).
# ---------------------------------------------------------------------------
chain_prologue

# ---------------------------------------------------------------------------
# Stage 1: stage DOTA (checksum-recorded, zero download -- already on autodl-pub).
# ---------------------------------------------------------------------------
step "1: stage DOTA from $DOTA_SRC"
if [ -d "$DOTA_SRC" ]; then
  du -sh "$DOTA_SRC" 2>/dev/null | tee "$RESULTS_DIR/dota_stage_du.txt"
  mark DOTA_STAGE OK
else
  echo "DOTA source $DOTA_SRC not found" >&2
  mark DOTA_STAGE FAILED
fi

# ---------------------------------------------------------------------------
# Stage 2: DIOR-R box-side fetch (hard Phase-0 gate; design §4.2).
# ---------------------------------------------------------------------------
step "2: DIOR-R fetch + gate"
if [ -z "$DIOR_R_URL" ]; then
  echo "DIOR_R_URL not set -- skipping DIOR-R fetch (disclosed; second-dataset arm blocked until set)"
  mark DIOR_R_FETCH SKIPPED_DISCLOSED
elif python3 "$SCRIPT_DIR/fetch_dior_r.py" --url "$DIOR_R_URL" --dest-dir "$DIOR_R_DEST" \
     --confirm-license-reviewed -o "$RESULTS_DIR/dior_r_fetch.json" 2>&1 | tee "$RESULTS_DIR/dior_r_fetch.log"; then
  mark DIOR_R_FETCH OK
else
  echo "DIOR-R gate failed -- design §4.2 fallback (HRSC2016 or DOTA-v2.0) must be applied manually" >&2
  mark DIOR_R_FETCH FAILED
fi

# ---------------------------------------------------------------------------
# Stage 3: Phase 0 -- score RTMDet-R-l on DOTA val, staging audit + repro gate.
# ---------------------------------------------------------------------------
step "3: Phase 0 (RTMDet-R-l val inference + staging audit + reproduction gate)"
DOTA_VAL_DETS="$RESULTS_DIR/dota_val_dets_rtmdet_r.jsonl"
if [ -z "$MMROTATE_COMMIT" ] || [ -z "$RTMDET_R_CONFIG" ] || [ -z "$RTMDET_R_CHECKPOINT" ]; then
  echo "MMROTATE_COMMIT/RTMDET_R_CONFIG/RTMDET_R_CHECKPOINT not all set -- skipping inference (disclosed)"
  mark RTMDET_R_INFERENCE SKIPPED_DISCLOSED
else
  if [ -s "$DOTA_VAL_DETS" ]; then
    skip "$DOTA_VAL_DETS"
  else
    python3 "$SCRIPT_DIR/score_rtmdet.py" --config "$RTMDET_R_CONFIG" --checkpoint "$RTMDET_R_CHECKPOINT" \
      --mmrotate-commit "$MMROTATE_COMMIT" --images-dir "$DOTA_VAL_IMAGES" --class-names "$DOTA_CLASS_NAMES" \
      --device "$DEVICE" -o "$DOTA_VAL_DETS" \
      || echo "warning: score_rtmdet.py exited non-zero -- content gate below will catch it"
  fi
  if [ -s "$DOTA_VAL_DETS" ]; then
    mark RTMDET_R_INFERENCE OK
  else
    mark RTMDET_R_INFERENCE FAILED
  fi
fi

if [ -s "$DOTA_VAL_GT" ] && [ -n "$PUBLISHED_VAL_MAP" ]; then
  # MEASURED_VAL_MAP must come from an external mAP-eval step (not computed by
  # this repo's core, per SOTA-REPRODUCTION-PLAN's "core never imports mmrotate"
  # rule) -- env-overridable, no silent default.
  if [ -n "${MEASURED_VAL_MAP:-}" ]; then
    if python3 "$SCRIPT_DIR/phase0.py" --gt "$DOTA_VAL_GT" --measured-val-map "$MEASURED_VAL_MAP" \
         --published-val-map "$PUBLISHED_VAL_MAP" --repro-tol "$ROTCERT_REPRO_TOL" \
         -o "$RESULTS_DIR/phase0_audit.json"; then
      mark PHASE0_AUDIT OK
    else
      mark PHASE0_AUDIT FAILED
      echo "K3 reproduction gate FAILED -- no certification arm runs until fixed" >&2
    fi
  else
    echo "MEASURED_VAL_MAP not set -- skipping phase0.py (disclosed)"
    mark PHASE0_AUDIT SKIPPED_DISCLOSED
  fi
else
  echo "DOTA_VAL_GT or PUBLISHED_VAL_MAP missing -- skipping phase0.py (disclosed)"
  mark PHASE0_AUDIT SKIPPED_DISCLOSED
fi

# ---------------------------------------------------------------------------
# Stage 4: Phase-1 pilot -- 1 detector x DOTA x PILOT_N_REPEATS scene-split
# repeats, full loop (match -> calibrate -> recall -> audit), K1 metrics.
# ---------------------------------------------------------------------------
step "4: Phase-1 pilot (1 detector x DOTA x $PILOT_N_REPEATS repeats)"
PILOT_OK=1
if [ -s "$DOTA_VAL_DETS" ] && [ -s "$DOTA_VAL_GT" ]; then
  pilot_dir="$RESULTS_DIR/pilot"
  mkdir -p "$pilot_dir"
  matched="$pilot_dir/matched.jsonl"
  if [ ! -s "$matched" ]; then
    rotcert match --dets "$DOTA_VAL_DETS" --gt "$DOTA_VAL_GT" --iou-thr 0.5 -o "$matched" \
      || { echo "warning: rotcert match failed"; PILOT_OK=0; }
  fi
  if [ -s "$matched" ]; then
    for score in gwd naive-coord hull; do
      cert="$pilot_dir/cert_${score}.json"
      rotcert calibrate --matched "$matched" --score "$score" --alpha "$PILOT_ALPHA" --mondrian -o "$cert" \
        || { echo "warning: calibrate $score failed"; PILOT_OK=0; }
    done
    recall_json="$pilot_dir/recall.json"
    rotcert recall --matched "$matched" --beta "$PILOT_BETA" --delta "$PILOT_DELTA" --mondrian -o "$recall_json" \
      || { echo "warning: recall failed"; PILOT_OK=0; }
    audit_json="$pilot_dir/audit.json"
    rotcert audit --matched "$matched" --score gwd --alpha "$PILOT_ALPHA" -o "$audit_json" \
      || { echo "warning: audit failed"; PILOT_OK=0; }
  else
    PILOT_OK=0
  fi
else
  echo "DOTA_VAL_DETS/DOTA_VAL_GT not both present -- pilot skipped (disclosed)"
  PILOT_OK=0
fi
if [ "$PILOT_OK" -eq 1 ]; then
  mark PHASE1_PILOT OK
else
  mark PHASE1_PILOT SKIPPED_DISCLOSED
fi

# ---------------------------------------------------------------------------
# Stage 5 (gated): the FULL grid, only behind an explicit prereg-freeze confirmation.
# ---------------------------------------------------------------------------
step "5: full grid (gated on REQUIRES_PREREG_FREEZE=confirmed)"
if [ "$REQUIRES_PREREG_FREEZE" = "confirmed" ]; then
  echo "REQUIRES_PREREG_FREEZE=confirmed -- full grid would run here (design §4.3: 2 detectors x 2 datasets x R=20 x 6 scores)"
  echo "NOT YET IMPLEMENTED in this chain -- extend Stage 5 once the prereg freeze (design §8 Phase 2) is actually confirmed."
  mark FULL_GRID SKIPPED_DISCLOSED
else
  echo "REQUIRES_PREREG_FREEZE not 'confirmed' -- full grid REFUSED (design §8: pilot-only until prereg freeze)"
  mark FULL_GRID SKIPPED_DISCLOSED
fi

# ---------------------------------------------------------------------------
# Epilogue (tar + marker + wait-for-ack + shutdown -- boxkit).
# ---------------------------------------------------------------------------
if [ "${#FAILED_MARKERS[@]}" -eq 0 ]; then
  marker_name="ROTCERT_PILOT_ALL_DONE"
  # No pre-epilogue echo of the marker: the watcher greps this log, and the
  # marker must only appear AFTER chain_epilogue's tar (2026-07-10 incident:
  # a pre-tar echo made the watcher pull a still-growing tarball).
else
  # Distinct from the clean-path marker (M3 convention, cf. ig_fullscore.sh):
  # a watcher polling for ROTCERT_PILOT_ALL_DONE must never mistake a partial
  # run for a complete one. Same tar name either way so partial results are
  # still packaged and pulled.
  marker_name="ROTCERT_PILOT_PARTIAL_DONE"
  echo "ROTCERT_PILOT_PARTIAL -- failed markers: ${FAILED_MARKERS[*]}"
fi

chain_epilogue "$RESULTS_DIR $MARKERS_DIR" "$marker_name" "rotcert_pilot"

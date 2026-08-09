#!/bin/bash
# next_boot_rotcert_dior_infer.sh -- rotcert DIOR-R val-INFERENCE chain (design
# §4.1 A2, runs AFTER next_boot_rotcert_dior_train.sh finishes on the box).
# Produces the DIOR-R substrate the local certification arm consumes: per-detector
# detections-JSONL over the DIOR-R TEST split + a canonical GT JSONL. Marker:
# ROTCERT_DIOR_INFER_ALL_DONE.
#
# Gate order (each a CONTENT gate -- never a bare exit code, per the portfolio's
# DOFA lesson + boxkit chain_lib.sh convention):
#   1. WAIT-FOR-TRAINING gate -- refuses to start unless the training chain left
#      a per-detector checkpoint-integrity marker == OK in its markers dir. If only
#      ONE detector's marker is OK (training PARTIAL), proceed with that detector
#      and DISCLOSE the other as skipped; if NEITHER, refuse (nothing to score).
#      NB: chain_epilogue writes its ALL_DONE line to the LOG, not a markers-dir
#      file, so we key off the integrity MARKER FILES the training chain's mark()
#      wrote (ORCNN_DIOR_CKPT_INTEGRITY / RTMDET_R_DIOR_CKPT_INTEGRITY) -- those
#      are the authoritative "this checkpoint is present and integral" signals.
#   2. Inference x2 -- score_rtmdet.py (detector-agnostic; handles ORCNN + RTMDet)
#      with each in-house checkpoint over the DIOR-R TEST images. Content gate:
#      non-empty + every line JSON-parses + distinct-image coverage >= a floor.
#   3. GT canonicalization -- prepare_dior_gt.py over the OBB xmls restricted to the
#      TEST ImageSet. Content gate: n_gt_records > 0 AND degenerate boxes <= floor.
#   4. NO certification stages here (prereg pending) and NO AOPG anything -- the
#      certificate + reproduction gate run LOCALLY after the tar is pulled.
# Then epilogue: tar -> DISTINCT done/partial marker -> ack-wait -> shutdown (boxkit).
#
# DIOR-R TEST is the only held-out split (11,738 images): trainval (11,725) was
# consumed by next_boot_rotcert_dior_train.sh, so scoring it would be train-on-test.
#
# pidfile-guarded background processes only (never `pkill -f <pattern>` matching the
# invocation itself -- chain_lib.sh ABSOLUTE RULE).
#
# Usage (on the AutoDL box, after training completes): bash next_boot_rotcert_dior_infer.sh
# (or: nohup bash next_boot_rotcert_dior_infer.sh > /root/rotcert_dior_infer.log 2>&1 &)

set -uo pipefail  # deliberately NOT -e: later stages/markers/epilogue must still run
                   # after an earlier stage's content gate fails.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELIABILITY_COMMONS="${RELIABILITY_COMMONS:-/root/reliability-commons}"

# shellcheck disable=SC1091
source "${RELIABILITY_COMMONS}/tools/boxkit/chain_lib.sh"

# ---------------------------------------------------------------------------
# Tunables (env-overridable, named defaults -- NO hardcoded paths/counts inline
# in the stages below).
# ---------------------------------------------------------------------------
RESULTS_DIR="${RESULTS_DIR:-/root/autodl-tmp/rotcert_dior_infer_results}"

# --- training-chain handoff (its markers dir + work dir; must match the train chain) ---
TRAIN_RESULTS_DIR="${TRAIN_RESULTS_DIR:-/root/autodl-tmp/rotcert_dior_train_results}"
TRAIN_MARKERS_DIR="${TRAIN_MARKERS_DIR:-$TRAIN_RESULTS_DIR/markers}"
ORCNN_CKPT_MARKER="${ORCNN_CKPT_MARKER:-ORCNN_DIOR_CKPT_INTEGRITY}"
RTMDET_CKPT_MARKER="${RTMDET_CKPT_MARKER:-RTMDET_R_DIOR_CKPT_INTEGRITY}"

# --- DIOR-R staged layout (mirrors next_boot_rotcert_dior_train.sh) ---
DIOR_R_ROOT="${DIOR_R_ROOT:-/root/autodl-tmp/dior_r}"
DIOR_R_OBB_ANN_SUBDIR="${DIOR_R_OBB_ANN_SUBDIR:-Annotations/Oriented Bounding Boxes}"
DIOR_R_TEST_IMG_SUBDIR="${DIOR_R_TEST_IMG_SUBDIR:-JPEGImages-test}"
DIOR_R_IMAGESETS_SUBDIR="${DIOR_R_IMAGESETS_SUBDIR:-ImageSets/Main}"
DIOR_R_TEST_IMAGESET="${DIOR_R_TEST_IMAGESET:-$DIOR_R_ROOT/$DIOR_R_IMAGESETS_SUBDIR/test.txt}"
DIOR_R_WORK_DIR="${DIOR_R_WORK_DIR:-$DIOR_R_ROOT/work_dirs}"

# --- detectors: work-dir names + epochs match the training chain's train_stage() ---
ORCNN_NAME="${ORCNN_NAME:-orcnn_dior}"
RTMDET_NAME="${RTMDET_NAME:-rtmdet_r_dior}"
SEED="${SEED:-0}"
ORCNN_EPOCHS="${ORCNN_EPOCHS:-12}"     # 1x  (default checkpoint = epoch_${ORCNN_EPOCHS}.pth)
RTMDET_R_EPOCHS="${RTMDET_R_EPOCHS:-36}"  # 3x (default checkpoint = epoch_${RTMDET_R_EPOCHS}.pth)

# Checkpoints: env-overridable; default to the train chain's per-seed final epoch.
ORCNN_DIOR_CHECKPOINT="${ORCNN_DIOR_CHECKPOINT:-$DIOR_R_WORK_DIR/$ORCNN_NAME/seed_${SEED}/epoch_${ORCNN_EPOCHS}.pth}"
RTMDET_R_DIOR_CHECKPOINT="${RTMDET_R_DIOR_CHECKPOINT:-$DIOR_R_WORK_DIR/$RTMDET_NAME/seed_${SEED}/epoch_${RTMDET_R_EPOCHS}.pth}"

# Configs REQUIRED for a real inference run (empty default => disclosed skip, per donor).
ORCNN_DIOR_CONFIG="${ORCNN_DIOR_CONFIG:-}"
RTMDET_R_DIOR_CONFIG="${RTMDET_R_DIOR_CONFIG:-}"

# --- mm-stack (score_rtmdet.py pins the commit; our core never imports mmrotate) ---
MMROTATE_COMMIT="${MMROTATE_COMMIT:-}"   # REQUIRED for any real inference run
DEVICE="${DEVICE:-cuda:0}"
SCORE_THR="${SCORE_THR:-0.05}"           # pass-through to score_rtmdet.py

# --- DIOR-R 20 classes, index-aligned to the trained model's label order ---
# VERIFIED 2026-07-11 against the box's pinned mmrotate checkout
# (/root/mmrotate/mmrotate/datasets/dior.py DIORDataset.METAINFO): the
# expressway-* classes sit at indices 6-7 and dam at 8 (NOT the DIOR paper's
# alphabetical order), all lowercase. Detector label indices follow METAINFO,
# so a transposed order silently mislabels every detection.
DIOR_CLASS_NAMES="${DIOR_CLASS_NAMES:-airplane,airport,baseballfield,basketballcourt,bridge,chimney,expressway-service-area,expressway-toll-station,dam,golffield,groundtrackfield,harbor,overpass,ship,stadium,storagetank,tenniscourt,trainstation,vehicle,windmill}"

# --- content-gate floors (named defaults; env-overridable) ---
DIOR_TEST_MIN_COVER="${DIOR_TEST_MIN_COVER:-11000}"   # distinct images that must appear in dets (of 11,738)
DIOR_GT_MAX_DEGENERATE="${DIOR_GT_MAX_DEGENERATE:-0}" # degenerate GT boxes tolerated (full 800x800 images => 0)

# --- GT difficult policy (forwarded to prepare_dior_gt.py; frozen at prereg) ---
DIOR_DIFFICULT_POLICY="${DIOR_DIFFICULT_POLICY:-include}"
# DIOR-R <angle> values sampled on the box (2026-07-11) include -11/-9.6/-6.2
# (>2*pi) => DEGREES. Moot for the corner-tag schema this dataset actually uses
# (prepare_dior_gt's polygon path never reads <angle>), but the default must
# still be right for any 5-param stragglers.
DIOR_ANGLE_UNIT="${DIOR_ANGLE_UNIT:-degrees}"
export DIOR_DIFFICULT_POLICY DIOR_ANGLE_UNIT

MARKERS_DIR="${MARKERS_DIR:-$RESULTS_DIR/markers}"
mkdir -p "$RESULTS_DIR" "$MARKERS_DIR"

FAILED_MARKERS=()
RUN_PARTIAL=0   # forces the PARTIAL epilogue marker even when skips are "disclosed"
step() { echo "== $1 =="; }
skip() { echo "skip (already exists): $1"; }
mark() {
  local name="$1" status="$2"
  printf '%s\n' "$status" > "$MARKERS_DIR/${name}.marker"
  echo "MARKER ${name}=${status}"
  { [ "$status" = "FAILED" ] || [ "$status" = "REFUSED" ]; } && FAILED_MARKERS+=("$name")
  return 0
}

# read_marker NAME -> stdout the marker's content (trimmed), or empty if absent.
read_marker() {
  local f="$TRAIN_MARKERS_DIR/$1.marker"
  [ -f "$f" ] && tr -d '[:space:]' < "$f" || printf ''
}

# gate_jsonl_coverage FILE FLOOR -> exit 0 iff FILE is non-empty, every line parses
# as JSON, and the count of DISTINCT image_ids is >= FLOOR. Pure stdlib; prints a
# one-line summary. (No process-pattern / marker strings embedded -- grep-phantom
# lesson: only this chain's OWN done-marker must never appear before chain_epilogue.)
gate_jsonl_coverage() {
  python3 - "$1" "$2" <<'PY'
import json, sys
path, floor = sys.argv[1], int(sys.argv[2])
ids = set()
n = 0
try:
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"COVERAGE_FAIL: {path} line {ln} not JSON: {e}", file=sys.stderr)
                sys.exit(1)
            iid = rec.get("image_id")
            if iid is not None:
                ids.add(iid)
            n += 1
except FileNotFoundError:
    print(f"COVERAGE_FAIL: {path} missing", file=sys.stderr)
    sys.exit(1)
ok = n > 0 and len(ids) >= floor
print(f"COVERAGE {path}: records={n} distinct_images={len(ids)} floor={floor} passed={ok}")
sys.exit(0 if ok else 1)
PY
}

# gate_gt_provenance PROV_JSON MAX_DEGENERATE -> exit 0 iff n_gt_records > 0 AND
# n_degenerate_skipped <= MAX_DEGENERATE. Reads the sidecar prepare_dior_gt wrote.
gate_gt_provenance() {
  python3 - "$1" "$2" <<'PY'
import json, sys
prov_path, max_deg = sys.argv[1], int(sys.argv[2])
try:
    prov = json.load(open(prov_path, encoding="utf-8"))
except FileNotFoundError:
    print(f"GT_GATE_FAIL: {prov_path} missing", file=sys.stderr)
    sys.exit(1)
n = int(prov.get("n_gt_records", 0))
deg = int(prov.get("n_degenerate_skipped", 0))
ok = n > 0 and deg <= max_deg
print(f"GT_GATE {prov_path}: n_gt_records={n} degenerate={deg} max_degenerate={max_deg} passed={ok}")
sys.exit(0 if ok else 1)
PY
}

# infer_detector NAME CONFIG CHECKPOINT READY DETS_OUT MARKER
# Scores one detector over the DIOR-R test images; content-gates the JSONL.
infer_detector() {
  local name="$1" config="$2" checkpoint="$3" ready="$4" dets_out="$5" marker="$6"
  if [ "$ready" -ne 1 ]; then
    echo "$name: training checkpoint not ready (integrity marker not OK) -- inference skipped (disclosed)"
    mark "$marker" SKIPPED_DISCLOSED
    return
  fi
  if [ -z "$MMROTATE_COMMIT" ] || [ -z "$config" ] || [ -z "$DIOR_CLASS_NAMES" ]; then
    echo "$name: MMROTATE_COMMIT/config/DIOR_CLASS_NAMES not all set -- inference skipped (disclosed)"
    mark "$marker" SKIPPED_DISCLOSED
    return
  fi
  if [ ! -s "$checkpoint" ]; then
    echo "$name: checkpoint $checkpoint missing/empty -- inference skipped (disclosed)"
    mark "$marker" SKIPPED_DISCLOSED
    return
  fi
  if [ -s "$dets_out" ]; then
    skip "$dets_out"
  else
    python3 "$SCRIPT_DIR/score_rtmdet.py" --config "$config" --checkpoint "$checkpoint" \
      --mmrotate-commit "$MMROTATE_COMMIT" --images-dir "$TEST_IMAGES_DIR" \
      --class-names "$DIOR_CLASS_NAMES" --score-thr "$SCORE_THR" --device "$DEVICE" -o "$dets_out" \
      || echo "warning: score_rtmdet.py ($name) exited non-zero -- content gate below is authoritative"
  fi
  if gate_jsonl_coverage "$dets_out" "$DIOR_TEST_MIN_COVER" | tee -a "$RESULTS_DIR/${name}_coverage.txt"; then
    mark "$marker" OK
  else
    mark "$marker" FAILED
  fi
}

# ---------------------------------------------------------------------------
# Prologue (conda activate, HF_HOME, balance_guard, per-minute gpu logger -- boxkit).
# ---------------------------------------------------------------------------
chain_prologue

TEST_IMAGES_DIR="$DIOR_R_ROOT/$DIOR_R_TEST_IMG_SUBDIR"
OBB_ANN_DIR="$DIOR_R_ROOT/$DIOR_R_OBB_ANN_SUBDIR"

# ---------------------------------------------------------------------------
# Stage 1: wait-for-training gate (per-detector checkpoint-integrity markers).
# ---------------------------------------------------------------------------
step "1: wait-for-training gate ($TRAIN_MARKERS_DIR)"
ORCNN_READY=0
RTMDET_READY=0
orcnn_status="$(read_marker "$ORCNN_CKPT_MARKER")"
rtmdet_status="$(read_marker "$RTMDET_CKPT_MARKER")"
echo "training integrity markers: $ORCNN_CKPT_MARKER=${orcnn_status:-<absent>} $RTMDET_CKPT_MARKER=${rtmdet_status:-<absent>}"
[ "$orcnn_status" = "OK" ] && [ -s "$ORCNN_DIOR_CHECKPOINT" ] && ORCNN_READY=1
[ "$rtmdet_status" = "OK" ] && [ -s "$RTMDET_R_DIOR_CHECKPOINT" ] && RTMDET_READY=1

if [ "$ORCNN_READY" -eq 1 ] && [ "$RTMDET_READY" -eq 1 ]; then
  echo "both detectors' checkpoints present + integral -- proceeding with the full 2-detector inference"
  mark WAIT_FOR_TRAINING OK
elif [ "$ORCNN_READY" -eq 1 ] || [ "$RTMDET_READY" -eq 1 ]; then
  echo "only one detector's checkpoint is ready (training PARTIAL) -- proceeding with the ready one, other disclosed"
  mark WAIT_FOR_TRAINING PARTIAL_DISCLOSED
  RUN_PARTIAL=1
else
  echo "REFUSE: neither $ORCNN_CKPT_MARKER nor $RTMDET_CKPT_MARKER is OK with an integral checkpoint in $TRAIN_MARKERS_DIR." >&2
  echo "  The training chain has not produced a scorable DIOR-R checkpoint yet -- run it to completion first." >&2
  mark WAIT_FOR_TRAINING REFUSED
fi

# ---------------------------------------------------------------------------
# Stage 2: inference x2 over the DIOR-R TEST split (only the ready detectors).
# ---------------------------------------------------------------------------
step "2: DIOR-R test inference (images: $TEST_IMAGES_DIR)"
if [ ! -d "$TEST_IMAGES_DIR" ]; then
  echo "DIOR-R test images dir $TEST_IMAGES_DIR missing -- both inference arms skipped (disclosed)" >&2
  ORCNN_READY=0
  RTMDET_READY=0
fi
ORCNN_DETS="$RESULTS_DIR/dior_test_dets_orcnn.jsonl"
RTMDET_DETS="$RESULTS_DIR/dior_test_dets_rtmdet.jsonl"
infer_detector "orcnn" "$ORCNN_DIOR_CONFIG" "$ORCNN_DIOR_CHECKPOINT" "$ORCNN_READY" "$ORCNN_DETS" ORCNN_DIOR_INFER
infer_detector "rtmdet" "$RTMDET_R_DIOR_CONFIG" "$RTMDET_R_DIOR_CHECKPOINT" "$RTMDET_READY" "$RTMDET_DETS" RTMDET_DIOR_INFER

# ---------------------------------------------------------------------------
# Stage 3: DIOR-R test GT canonicalization (box-side prepare_dior_gt.py).
# ---------------------------------------------------------------------------
step "3: DIOR-R test GT canonicalization ($OBB_ANN_DIR)"
DIOR_TEST_GT="$RESULTS_DIR/dior_test_gt.jsonl"
if [ ! -d "$OBB_ANN_DIR" ]; then
  echo "DIOR-R OBB annotation dir $OBB_ANN_DIR missing -- GT prep skipped (disclosed)" >&2
  mark DIOR_TEST_GT SKIPPED_DISCLOSED
elif [ -s "$DIOR_TEST_GT" ] && [ -s "$DIOR_TEST_GT.provenance.json" ]; then
  skip "$DIOR_TEST_GT"
  if gate_gt_provenance "$DIOR_TEST_GT.provenance.json" "$DIOR_GT_MAX_DEGENERATE"; then
    mark DIOR_TEST_GT OK
  else
    mark DIOR_TEST_GT FAILED
  fi
else
  imageset_arg=()
  if [ -f "$DIOR_R_TEST_IMAGESET" ]; then
    imageset_arg=(--imageset-file "$DIOR_R_TEST_IMAGESET")
  else
    echo "warning: test ImageSet $DIOR_R_TEST_IMAGESET absent -- GT will cover ALL xmls under" >&2
    echo "  $OBB_ANN_DIR (trainval+test, ~23k), NOT the test split alone. Set DIOR_R_TEST_IMAGESET" >&2
    echo "  to the test.txt split list so the GT lines up with the test-only detections." >&2
  fi
  python3 "$SCRIPT_DIR/prepare_dior_gt.py" --annfiles-dir "$OBB_ANN_DIR" \
    "${imageset_arg[@]}" --difficult-policy "$DIOR_DIFFICULT_POLICY" \
    --angle-unit "$DIOR_ANGLE_UNIT" -o "$DIOR_TEST_GT" \
    2>&1 | tee "$RESULTS_DIR/dior_test_gt_prep.log" \
    || echo "warning: prepare_dior_gt.py exited non-zero (schema refusal or IO) -- content gate below is authoritative"
  if [ -s "$DIOR_TEST_GT" ] && gate_gt_provenance "$DIOR_TEST_GT.provenance.json" "$DIOR_GT_MAX_DEGENERATE"; then
    mark DIOR_TEST_GT OK
  else
    mark DIOR_TEST_GT FAILED
  fi
fi

# ---------------------------------------------------------------------------
# Stage 4: NO certification here. The Learn-then-Test certificate, the score
# audit, and the AOPG reproduction gate all run LOCALLY after this tar is pulled
# (prereg pending). This chain's sole job is the DIOR-R dets + GT substrate.
# ---------------------------------------------------------------------------
step "4: certification deferred to local post-pull (prereg pending; no AOPG here)"

# ---------------------------------------------------------------------------
# Epilogue (tar + DISTINCT done/partial marker + wait-for-ack + shutdown -- boxkit).
# ---------------------------------------------------------------------------
if [ "${#FAILED_MARKERS[@]}" -eq 0 ] && [ "$RUN_PARTIAL" -eq 0 ]; then
  marker_name="ROTCERT_DIOR_INFER_ALL_DONE"
  # No pre-epilogue echo of the marker: the watcher greps this log, and the marker
  # must only appear AFTER chain_epilogue's tar (2026-07-10 incident: a pre-tar echo
  # made the watcher pull a still-growing tarball).
else
  # Distinct from the clean-path marker: a watcher polling for ALL_DONE must never
  # mistake a partial/refused run (or a single-detector run) for the full 2-detector
  # deliverable. Same tar either way so partial artifacts still get pulled.
  marker_name="ROTCERT_DIOR_INFER_PARTIAL_DONE"
  echo "ROTCERT_DIOR_INFER_PARTIAL -- failed/refused markers: ${FAILED_MARKERS[*]:-none}; partial_training=${RUN_PARTIAL}"
fi

chain_epilogue "$RESULTS_DIR $MARKERS_DIR" "$marker_name" "rotcert_dior_infer"

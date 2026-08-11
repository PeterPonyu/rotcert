#!/bin/bash
# next_boot_rotcert_dior_train.sh -- rotcert DIOR-R in-house TRAINING chain
# (design §4.1 A2 addendum 2026-07-10): no license-clean DIOR-R-trained checkpoint
# exists anywhere (mmrotate zoo is DOTA/HRSC-only; AOPG publishes tables, no weights;
# LSKNet is CC-BY-NC), so BOTH detectors must be trained in-house on DIOR-R trainval
# via Apache-2.0 mmrotate dev-1.x (which has DIOR dataset support). This chain stands
# that pipeline up on the box. Marker: ROTCERT_DIOR_TRAIN_ALL_DONE.
#
# Gate order (each is a CONTENT gate -- never a bare exit code, per the portfolio's
# DOFA lesson + boxkit chain_lib.sh convention):
#   1. DIOR-R STAGED-DATA content gate  -- counts vs the staged layout (OBB xmls +
#      trainval/test jpgs + ImageSets), expected counts read from env (named
#      defaults), NEVER inline literals.
#   2. LICENSE-REVIEW gate              -- HARD refuse unless the human-review marker
#      file exists (mirrors fetch_dior_r.py's --confirm-license-reviewed pattern);
#      also governs whether in-house weights may be rehosted (addendum A7).
#   3. mm-stack import-probe gate       -- import probes, NEVER pip exit codes (house
#      lesson: pip can exit 0 with a broken import and vice versa).
#   4. Oriented R-CNN R-50 1x training  -- parameterized (config/epochs/batch/seeds
#      via env with named defaults), per-seed work_dir, checkpoint-integrity gate.
#   5. RTMDet-R-l 3x training           -- same shape, its own params.
#   6. AOPG-table reproduction gate     -- REQUIRES the published AOPG DIOR-R mAP
#      value frozen at prereg; guarded on REQUIRES_PREREG_FREEZE=confirmed (mirrors
#      next_boot_rotcert.sh's full-grid freeze gate).
# Then epilogue: tar -> distinct DONE/FAIL marker -> ack-wait -> shutdown (boxkit).
#
# Per-minute nvidia-smi logging is started by chain_prologue (boxkit gpu_util logger,
# 60s interval) -> $GPU_UTIL_LOG; this chain does not duplicate it.
#
# pidfile-guarded background processes only (never `pkill -f <pattern>` matching the
# invocation itself -- chain_lib.sh ABSOLUTE RULE).
#
# Usage (on the AutoDL box): bash next_boot_rotcert_dior_train.sh
# (or: nohup bash next_boot_rotcert_dior_train.sh > /root/rotcert_dior_train.log 2>&1 &)

set -uo pipefail  # deliberately NOT -e: later stages/markers/epilogue must still run
                   # after an earlier stage's content gate fails.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELIABILITY_COMMONS="${RELIABILITY_COMMONS:-/root/reliability-commons}"

# shellcheck disable=SC1091
source "${RELIABILITY_COMMONS}/tools/boxkit/chain_lib.sh"

# ---------------------------------------------------------------------------
# Tunables (env-overridable, named defaults -- NO hardcoded paths/counts/tols
# inline in the stages below).
# ---------------------------------------------------------------------------
RESULTS_DIR="${RESULTS_DIR:-${AUTODL_TMP}/rotcert_dior_train_results}"

# --- DIOR-R staged layout (design A2: staged from ModelScope to the data disk) ---
DIOR_R_ROOT="${DIOR_R_ROOT:-${AUTODL_TMP}/dior_r}"
DIOR_R_OBB_ANN_SUBDIR="${DIOR_R_OBB_ANN_SUBDIR:-Annotations/Oriented Bounding Boxes}"
DIOR_R_TRAINVAL_IMG_SUBDIR="${DIOR_R_TRAINVAL_IMG_SUBDIR:-JPEGImages-trainval}"
DIOR_R_TEST_IMG_SUBDIR="${DIOR_R_TEST_IMG_SUBDIR:-JPEGImages-test}"
DIOR_R_IMAGESETS_SUBDIR="${DIOR_R_IMAGESETS_SUBDIR:-ImageSets/Main}"
# Expected counts (DIOR-R: 23463 images = 11725 trainval + 11738 test, one OBB xml
# per image). Floors, env-overridable -- read here, never asserted as literals below.
DIOR_R_MIN_OBB_XMLS="${DIOR_R_MIN_OBB_XMLS:-23000}"
DIOR_R_MIN_TRAINVAL_JPGS="${DIOR_R_MIN_TRAINVAL_JPGS:-11000}"
DIOR_R_MIN_TEST_JPGS="${DIOR_R_MIN_TEST_JPGS:-11000}"
DIOR_R_IMG_EXT="${DIOR_R_IMG_EXT:-jpg}"

# --- LICENSE-REVIEW gate (HARD; addendum A7) -----------------------------------
# A human must read the DIOR-R distribution terms (incl. the derived-weights-rehost
# clause) and TOUCH this marker file before any training runs. Mirrors
# fetch_dior_r.py's --confirm-license-reviewed required flag.
DIOR_R_LICENSE_REVIEWED_MARKER="${DIOR_R_LICENSE_REVIEWED_MARKER:-$DIOR_R_ROOT/DIOR_R_LICENSE_REVIEWED}"

# --- mm-stack (training carries the pins now, not just inference; design A7) ----
MMROTATE_COMMIT="${MMROTATE_COMMIT:-}"           # REQUIRED for any real training run
MMROTATE_DIR="${MMROTATE_DIR:-/root/mmrotate}"
MMROTATE_TRAIN_SCRIPT="${MMROTATE_TRAIN_SCRIPT:-$MMROTATE_DIR/tools/train.py}"
DEVICE="${DEVICE:-cuda:0}"
EXTRA_CFG_OPTIONS="${EXTRA_CFG_OPTIONS:-}"        # appended verbatim to --cfg-options

# --- Oriented R-CNN R-50 1x stage (design A2: ~8-10 GPU-h) ----------------------
ORCNN_DIOR_CONFIG="${ORCNN_DIOR_CONFIG:-}"        # REQUIRED for the orcnn stage
ORCNN_EPOCHS="${ORCNN_EPOCHS:-12}"                # 1x
ORCNN_BATCH="${ORCNN_BATCH:-2}"                   # samples per gpu
ORCNN_SEEDS="${ORCNN_SEEDS:-0}"                   # comma-separated; A2 open seed policy (1 vs 3)

# --- RTMDet-R-l 3x stage (design A2: ~25-35 GPU-h at batch ~8 on 24GB) ----------
RTMDET_R_DIOR_CONFIG="${RTMDET_R_DIOR_CONFIG:-}"  # REQUIRED for the rtmdet stage
RTMDET_R_EPOCHS="${RTMDET_R_EPOCHS:-36}"          # 3x
RTMDET_R_BATCH="${RTMDET_R_BATCH:-8}"             # batch ~8 fits 24GB (A2)
RTMDET_R_SEEDS="${RTMDET_R_SEEDS:-0}"             # comma-separated; A2 open seed policy

DIOR_R_WORK_DIR="${DIOR_R_WORK_DIR:-$DIOR_R_ROOT/work_dirs}"
CKPT_MIN_BYTES="${CKPT_MIN_BYTES:-1000000}"       # legacy-pickle non-empty floor (integrity fallback)
export CKPT_MIN_BYTES

# --- AOPG-table reproduction gate (design A2/K3: target = AOPG DIOR-R table) -----
# The published AOPG mAPs and this run's MEASURED mAPs come from an external eval
# (tools/test.py) step, NOT computed here -- mirrors next_boot_rotcert.sh's
# MEASURED_VAL_MAP convention. No silent defaults; a missing value => disclosed skip.
DIOR_R_REPRO_TOL="${DIOR_R_REPRO_TOL:-0.5}"       # mAP points
AOPG_ORCNN_DIOR_MAP="${AOPG_ORCNN_DIOR_MAP:-}"    # published AOPG table number (REQUIRED at freeze)
AOPG_RTMDET_R_DIOR_MAP="${AOPG_RTMDET_R_DIOR_MAP:-}"
MEASURED_ORCNN_DIOR_MAP="${MEASURED_ORCNN_DIOR_MAP:-}"
MEASURED_RTMDET_R_DIOR_MAP="${MEASURED_RTMDET_R_DIOR_MAP:-}"

REQUIRES_PREREG_FREEZE="${REQUIRES_PREREG_FREEZE:-}"

MARKERS_DIR="${MARKERS_DIR:-$RESULTS_DIR/markers}"
mkdir -p "$RESULTS_DIR" "$MARKERS_DIR"

FAILED_MARKERS=()
step() { echo "== $1 =="; }
skip() { echo "skip (already exists): $1"; }
mark() {
  local name="$1" status="$2"
  printf '%s\n' "$status" > "$MARKERS_DIR/${name}.marker"
  echo "MARKER ${name}=${status}"
  [ "$status" = "FAILED" ] || [ "$status" = "REFUSED" ] && FAILED_MARKERS+=("$name")
}

# count_files DIR PATTERN -> number of matching regular files under DIR (recursive).
count_files() {
  find "$1" -type f -name "$2" 2>/dev/null | wc -l | tr -d ' '
}

# probe_ckpt_integrity CKPT -> exit 0 iff the checkpoint's zip CRCs verify (modern
# torch.save is a zip archive); falls back to a non-empty size floor for legacy
# pickle checkpoints. Pure stdlib -- runs in any environment.
probe_ckpt_integrity() {
  python3 - "$1" <<'PY'
import os, sys, zipfile
path = sys.argv[1]
if not os.path.exists(path):
    print(f"CKPT_INTEGRITY_FAIL: {path} missing", file=sys.stderr); sys.exit(1)
try:
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
    if bad is not None:
        print(f"CKPT_INTEGRITY_FAIL: CRC error in member {bad} of {path}", file=sys.stderr)
        sys.exit(1)
    print(f"CKPT_INTEGRITY_OK: zip CRC verified ({path})")
except zipfile.BadZipFile:
    floor = int(os.environ.get("CKPT_MIN_BYTES", "1000000"))
    sz = os.path.getsize(path)
    if sz < floor:
        print(f"CKPT_INTEGRITY_FAIL: legacy pickle {sz} < CKPT_MIN_BYTES={floor} ({path})", file=sys.stderr)
        sys.exit(1)
    print(f"CKPT_INTEGRITY_OK: legacy pickle, {sz} bytes >= {floor} (not a zip) ({path})")
PY
}

# repro_gate MEASURED PUBLISHED TOL -> exit 0 iff |measured-published| <= tol.
# Pure stdlib float compare (awk/bc float pitfalls avoided); writes nothing.
repro_gate() {
  python3 - "$1" "$2" "$3" <<'PY'
import sys
measured, published, tol = (float(sys.argv[i]) for i in (1, 2, 3))
gap = measured - published
ok = abs(gap) <= tol
print(f"AOPG_REPRO measured={measured} published={published} tol={tol} gap={gap:+.4f} passed={ok}")
sys.exit(0 if ok else 1)
PY
}

# train_stage NAME CONFIG EPOCHS BATCH SEEDS_CSV MARKER_TRAIN MARKER_CKPT
# Trains one detector across the seed set; content-gates each seed's final
# checkpoint via probe_ckpt_integrity. Guarded by GATES_OK (staged+license+mm).
train_stage() {
  local name="$1" config="$2" epochs="$3" batch="$4" seeds_csv="$5"
  local marker_train="$6" marker_ckpt="$7"
  local work_base="$DIOR_R_WORK_DIR/$name"

  if [ "$GATES_OK" -ne 1 ]; then
    echo "$name: upstream gates not all OK -- training skipped (disclosed)"
    mark "$marker_train" SKIPPED_DISCLOSED
    mark "$marker_ckpt" SKIPPED_DISCLOSED
    return
  fi
  if [ -z "$MMROTATE_COMMIT" ] || [ -z "$config" ]; then
    echo "$name: MMROTATE_COMMIT/${name}_CONFIG not both set -- training skipped (disclosed)"
    mark "$marker_train" SKIPPED_DISCLOSED
    mark "$marker_ckpt" SKIPPED_DISCLOSED
    return
  fi

  local train_ok=1 ckpt_ok=1 seed work ckpt
  for seed in ${seeds_csv//,/ }; do
    work="$work_base/seed_${seed}"
    ckpt="$work/epoch_${epochs}.pth"
    mkdir -p "$work"
    if [ -s "$ckpt" ]; then
      skip "$ckpt"
    else
      # mmrotate dev-1.x (mmengine) cfg keys [VERIFY against the pinned config at
      # Phase 0]: randomness.seed / train_cfg.max_epochs / train_dataloader.batch_size.
      # shellcheck disable=SC2086
      python3 "$MMROTATE_TRAIN_SCRIPT" "$config" --work-dir "$work" \
        --cfg-options "randomness.seed=${seed}" "train_cfg.max_epochs=${epochs}" \
        "train_dataloader.batch_size=${batch}" ${EXTRA_CFG_OPTIONS} \
        2>&1 | tee "$RESULTS_DIR/${name}_seed${seed}_train.log" \
        || echo "warning: $name seed $seed train exited non-zero -- integrity gate below is authoritative"
    fi
    if [ -s "$ckpt" ] && probe_ckpt_integrity "$ckpt"; then
      : # this seed's checkpoint is present and integral
    else
      echo "$name seed $seed: no integral checkpoint at $ckpt" >&2
      train_ok=0
      ckpt_ok=0
    fi
  done

  if [ "$train_ok" -eq 1 ]; then mark "$marker_train" OK; else mark "$marker_train" FAILED; fi
  if [ "$ckpt_ok" -eq 1 ]; then mark "$marker_ckpt" OK; else mark "$marker_ckpt" FAILED; fi
}

# aopg_repro_gate NAME MEASURED PUBLISHED MARKER
# Guarded on REQUIRES_PREREG_FREEZE (the AOPG table number must be frozen at prereg,
# same discipline as next_boot_rotcert.sh's full-grid gate). Disclosed-skip otherwise.
aopg_repro_gate() {
  local name="$1" measured="$2" published="$3" marker="$4"
  if [ "$REQUIRES_PREREG_FREEZE" != "confirmed" ]; then
    echo "$name: REQUIRES_PREREG_FREEZE not 'confirmed' -- AOPG reproduction gate deferred (disclosed; freeze the AOPG target first)"
    mark "$marker" SKIPPED_DISCLOSED
    return
  fi
  if [ -z "$measured" ] || [ -z "$published" ]; then
    echo "$name: MEASURED/AOPG published DIOR-R mAP not both set -- AOPG reproduction gate skipped (disclosed)"
    mark "$marker" SKIPPED_DISCLOSED
    return
  fi
  if repro_gate "$measured" "$published" "$DIOR_R_REPRO_TOL" | tee "$RESULTS_DIR/${name}_aopg_repro.txt"; then
    mark "$marker" OK
  else
    mark "$marker" FAILED
    echo "$name: AOPG reproduction gate FAILED -- no certification arm runs on this DIOR-R detector until fixed (K3)" >&2
  fi
}

# ---------------------------------------------------------------------------
# Prologue (conda activate, HF_HOME, balance_guard, per-minute gpu logger -- boxkit).
# ---------------------------------------------------------------------------
chain_prologue

GATES_OK=1  # cleared by any of the three hard upstream gates below

# ---------------------------------------------------------------------------
# Gate 1: DIOR-R staged-data content gate (counts vs the staged layout).
# ---------------------------------------------------------------------------
step "1: DIOR-R staged-data content gate ($DIOR_R_ROOT)"
obb_dir="$DIOR_R_ROOT/$DIOR_R_OBB_ANN_SUBDIR"
trainval_dir="$DIOR_R_ROOT/$DIOR_R_TRAINVAL_IMG_SUBDIR"
test_dir="$DIOR_R_ROOT/$DIOR_R_TEST_IMG_SUBDIR"
imagesets_dir="$DIOR_R_ROOT/$DIOR_R_IMAGESETS_SUBDIR"

n_xml=$(count_files "$obb_dir" "*.xml")
n_trainval=$(count_files "$trainval_dir" "*.$DIOR_R_IMG_EXT")
n_test=$(count_files "$test_dir" "*.$DIOR_R_IMG_EXT")
{
  echo "obb_xmls=$n_xml (floor $DIOR_R_MIN_OBB_XMLS)"
  echo "trainval_jpgs=$n_trainval (floor $DIOR_R_MIN_TRAINVAL_JPGS)"
  echo "test_jpgs=$n_test (floor $DIOR_R_MIN_TEST_JPGS)"
  echo "imagesets_dir=$imagesets_dir"
} | tee "$RESULTS_DIR/dior_r_staged_counts.txt"

staged_ok=1
[ "$n_xml" -ge "$DIOR_R_MIN_OBB_XMLS" ] || { echo "GATE_FAIL: OBB xmls $n_xml < $DIOR_R_MIN_OBB_XMLS" >&2; staged_ok=0; }
[ "$n_trainval" -ge "$DIOR_R_MIN_TRAINVAL_JPGS" ] || { echo "GATE_FAIL: trainval jpgs $n_trainval < $DIOR_R_MIN_TRAINVAL_JPGS" >&2; staged_ok=0; }
[ "$n_test" -ge "$DIOR_R_MIN_TEST_JPGS" ] || { echo "GATE_FAIL: test jpgs $n_test < $DIOR_R_MIN_TEST_JPGS" >&2; staged_ok=0; }
[ -d "$imagesets_dir" ] || { echo "GATE_FAIL: ImageSets dir $imagesets_dir missing" >&2; staged_ok=0; }
if [ "$staged_ok" -eq 1 ]; then
  mark DIOR_R_STAGED_CONTENT OK
else
  mark DIOR_R_STAGED_CONTENT FAILED
  GATES_OK=0
fi

# ---------------------------------------------------------------------------
# Gate 2: LICENSE-REVIEW gate (HARD refuse unless the human-review marker exists).
# ---------------------------------------------------------------------------
step "2: DIOR-R license-review gate ($DIOR_R_LICENSE_REVIEWED_MARKER)"
if [ -f "$DIOR_R_LICENSE_REVIEWED_MARKER" ]; then
  echo "license-review marker present -- a human confirmed the DIOR-R terms (incl. derived-weights rehost clause, addendum A7)"
  mark DIOR_R_LICENSE_REVIEW OK
else
  echo "REFUSE: $DIOR_R_LICENSE_REVIEWED_MARKER absent -- DIOR-R license terms NOT reviewed." >&2
  echo "  A human must read the DIOR-R distribution terms and 'touch' that marker before any training runs" >&2
  echo "  (mirrors fetch_dior_r.py --confirm-license-reviewed; the gate also governs weight rehosting)." >&2
  mark DIOR_R_LICENSE_REVIEW REFUSED
  GATES_OK=0
fi

# ---------------------------------------------------------------------------
# Gate 3: mm-stack import-probe gate (import probes, NEVER pip exit codes).
# ---------------------------------------------------------------------------
step "3: mm-stack import-probe gate"
if python3 -c "import torch, mmcv, mmdet, mmrotate" 2>>"$RESULTS_DIR/mm_stack_probe.log"; then
  echo "mm-stack imports OK (torch/mmcv/mmdet/mmrotate)"
  mark MM_STACK_PROBE OK
else
  echo "GATE_FAIL: mm-stack import probe failed (see $RESULTS_DIR/mm_stack_probe.log) -- training refused" >&2
  mark MM_STACK_PROBE FAILED
  GATES_OK=0
fi

# ---------------------------------------------------------------------------
# Stage 4: Oriented R-CNN R-50 1x training (design A2: ~8-10 GPU-h).
# ---------------------------------------------------------------------------
step "4: Oriented R-CNN R-50 1x training (epochs=$ORCNN_EPOCHS batch=$ORCNN_BATCH seeds=$ORCNN_SEEDS)"
train_stage "orcnn_dior" "$ORCNN_DIOR_CONFIG" "$ORCNN_EPOCHS" "$ORCNN_BATCH" "$ORCNN_SEEDS" \
  ORCNN_DIOR_TRAIN ORCNN_DIOR_CKPT_INTEGRITY

# ---------------------------------------------------------------------------
# Stage 5: RTMDet-R-l 3x training (design A2: ~25-35 GPU-h at batch ~8 on 24GB).
# ---------------------------------------------------------------------------
step "5: RTMDet-R-l 3x training (epochs=$RTMDET_R_EPOCHS batch=$RTMDET_R_BATCH seeds=$RTMDET_R_SEEDS)"
train_stage "rtmdet_r_dior" "$RTMDET_R_DIOR_CONFIG" "$RTMDET_R_EPOCHS" "$RTMDET_R_BATCH" "$RTMDET_R_SEEDS" \
  RTMDET_R_DIOR_TRAIN RTMDET_R_DIOR_CKPT_INTEGRITY

# ---------------------------------------------------------------------------
# Stage 6: AOPG-table reproduction gate (gated on REQUIRES_PREREG_FREEZE=confirmed).
# ---------------------------------------------------------------------------
step "6: AOPG DIOR-R reproduction gate (freeze-gated; tol=$DIOR_R_REPRO_TOL mAP)"
aopg_repro_gate "orcnn_dior" "$MEASURED_ORCNN_DIOR_MAP" "$AOPG_ORCNN_DIOR_MAP" AOPG_REPRO_ORCNN
aopg_repro_gate "rtmdet_r_dior" "$MEASURED_RTMDET_R_DIOR_MAP" "$AOPG_RTMDET_R_DIOR_MAP" AOPG_REPRO_RTMDET

# ---------------------------------------------------------------------------
# Epilogue (tar + DISTINCT done/fail marker + wait-for-ack + shutdown -- boxkit).
# ---------------------------------------------------------------------------
if [ "${#FAILED_MARKERS[@]}" -eq 0 ]; then
  marker_name="ROTCERT_DIOR_TRAIN_ALL_DONE"
  # No pre-epilogue echo of the marker: the watcher greps this log, and the
  # marker must only appear AFTER chain_epilogue's tar (2026-07-10 incident:
  # a pre-tar echo made the watcher pull a still-growing tarball).
else
  # Distinct from the clean-path marker (M3 convention): a watcher polling
  # for ALL_DONE must never mistake a partial/refused run for a complete
  # one. Same tar either way so partial artifacts still get pulled.
  marker_name="ROTCERT_DIOR_TRAIN_PARTIAL_DONE"
  echo "ROTCERT_DIOR_TRAIN_PARTIAL -- failed/refused markers: ${FAILED_MARKERS[*]}"
fi

chain_epilogue "$RESULTS_DIR $MARKERS_DIR $DIOR_R_WORK_DIR" "$marker_name" "rotcert_dior_train"

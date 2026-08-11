#!/bin/bash
# next_boot_dior_seeds.sh -- DIOR-R multi-seed arm one-command box runner (2026-07-14).
#
# The TGRS/ISPRS workload critic's recommended experimental addition: 2 EXTRA training
# seeds (1,2) for the two CORE in-house detectors on DIOR-R -- the one dataset where the
# coverage-matched efficiency claim survives. The existing cells are seed 0 (already
# trained + certified, dior_cert_results_2026-07-11/{orcnn,rtmdet}). This chain adds
# seeds {1,2} so the seed-variance (q_hat / coverage spread) story generalises from HRSC
# to the DIOR-R claim it actually defends.
#
# Runs on the CURRENT rotcert box (19514) AFTER its Config-B chain finishes S2A-Net
# tonight: mm-stack (/root/MMSTACK_OK), DIOR-R staged data ($DIOR_R_ROOT), and the two
# DIOR configs are ALREADY in place there. This script ASSUMES that environment (it does
# NOT re-stage data, rebuild mm-stack, or re-emit the frozen GT -- the new seed cells
# reuse the SAME frozen dior_test_gt.jsonl / coverage-matched grid as seed 0).
#
# Seed mechanism (identical to hrsc_run.sh + next_boot_rotcert_dior_train.sh): mmrotate
# dev-1.x tools/train.py, per-seed work_dir seed_${s}, seed injected via
#   --cfg-options randomness.seed=${s} train_cfg.max_epochs=${ep} train_dataloader.batch_size=${b}
# Inference: score_rtmdet.py (detector-agnostic; the SAME path the seed-0 DIOR cells +
# the Config-B new cells used) -> per-seed detections JSONL on the DIOR-R TEST split.
#
# Stages (each content-gated -- assert checkpoint files exist + inference JSONs non-empty,
# NEVER bare exit codes -- with its own per-stage sentinel so --resume skips completed
# stages):
#   Stage 0  env / mm-stack gate            (STAGE0_MMSTACK marker + import probe)
#   Stage 1  Oriented R-CNN DIOR seeds 1,2  (1x=12ep, ~8 GPU-h/seed) train + infer test
#   Stage 2  RTMDet-R DIOR seeds 1,2        (3x=36ep, ~6-8 GPU-h/seed) train + infer test
#            -- OPTIONAL via SKIP_RTMDET=1 (see COST NOTE below)
#   Stage 3  tar results + DIOR_SEEDS_ALL_DONE sentinel (chain_epilogue)
#
# COST NOTE (RTMDet 3x): the 3x schedule is 36 epochs, but RTMDet-R-l is a ONE-STAGE
# detector at ~10 min/epoch on DIOR-R (batch 8), so a seed is ~6-8 GPU-h -- NOT the ~24
# GPU-h a naive "3x => 3x cost" reading would suggest (two-stage ORCNN 1x is ~40 min/epoch
# x 12ep ~= 8 GPU-h; the per-epoch rates differ by ~4x). Empirically measured on the box's
# seed-0 run (COMPUTE-PLAN-2026-07-13.md line 224: "RTMDet-R-l 3x on DIOR-R (36 ep, batch
# 8): ~6-8 GPU-h, ~10 min/epoch"). Because RTMDet seeds are therefore CHEAP and the critic
# asked for BOTH core detectors, SKIP_RTMDET defaults to 0 (run both). Set SKIP_RTMDET=1
# for the ORCNN-only variant (~16 GPU-h instead of ~28-32).
#
# Certification of the new cells (cert_cell.sh per seed cell) + the q_hat/coverage spread
# extension are the LOCAL, CPU, post-pull follow-up -- see DIOR-SEEDS-STAGING-2026-07-14.md.
# This chain only makes the GPU substrate. No auto-shutdown (teardown is the boxkit API's).
#
# --dry-run validates every referenced sub-script/config/path locally, prints the plan
# (incl. est. GPU-hours per stage), and exits 0 iff all LOCAL preflight checks pass --
# WITHOUT any GPU, box, or chain_lib dependency. Box-only paths are reported, not failed.
#
# Usage on the box:  bash next_boot_dior_seeds.sh          (add --resume to skip done stages)
# Usage on the box:  SKIP_RTMDET=1 bash next_boot_dior_seeds.sh   (ORCNN-only, ~16 GPU-h)
# Usage locally:     bash next_boot_dior_seeds.sh --dry-run
set -uo pipefail   # NOT -e: later stages/epilogue must run after an earlier gate fails.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # .../orchestration
ROTCERT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RELIABILITY_COMMONS="${RELIABILITY_COMMONS:-/root/reliability-commons}"
export PYTHONPATH="$ROTCERT_ROOT:${PYTHONPATH:-}"

DRY=0; RESUME=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1;;
    --resume)  RESUME=1;;
    *) echo "unknown arg '$a' (accepts --dry-run, --resume)" >&2; exit 2;;
  esac
done

# ---- tunables (env-overridable, named defaults) ----
MMROTATE_COMMIT="${MMROTATE_COMMIT:-3ff004eb21ea040455b5585db229edba4037f1bf}"
MMR="${MMR:-/root/mmrotate}"
DEVICE="${DEVICE:-cuda:0}"
SCORE_THR="${SCORE_THR:-0.05}"
SKIP_RTMDET="${SKIP_RTMDET:-0}"            # 1 -> ORCNN-only arm (see COST NOTE)
export MMROTATE_COMMIT

# --- DIOR-R staged layout (mirrors next_boot_rotcert_dior_{train,infer}.sh / configB) ---
DIOR_R_ROOT="${DIOR_R_ROOT:-${AUTODL_TMP}/dior_r}"
DIOR_R_TEST_IMG_SUBDIR="${DIOR_R_TEST_IMG_SUBDIR:-JPEGImages-test}"
DIOR_R_IMAGESETS_SUBDIR="${DIOR_R_IMAGESETS_SUBDIR:-ImageSets/Main}"
DIOR_R_TEST_IMAGESET="${DIOR_R_TEST_IMAGESET:-$DIOR_R_ROOT/$DIOR_R_IMAGESETS_SUBDIR/test.txt}"
DIOR_R_WORK_DIR="${DIOR_R_WORK_DIR:-$DIOR_R_ROOT/work_dirs}"
DIOR_TEST_MIN_COVER="${DIOR_TEST_MIN_COVER:-11000}"    # distinct test images that must appear in dets
DIOR_DIFFICULT_POLICY="${DIOR_DIFFICULT_POLICY:-include}"
DIOR_ANGLE_UNIT="${DIOR_ANGLE_UNIT:-degrees}"
export DIOR_DIFFICULT_POLICY DIOR_ANGLE_UNIT
# DIOR-R 20 classes, index-aligned to the trained model's METAINFO label order (VERIFIED
# 2026-07-11). Identical to next_boot_rotcert_dior_infer.sh / configB.
DIOR_CLASS_NAMES="${DIOR_CLASS_NAMES:-airplane,airport,baseballfield,basketballcourt,bridge,chimney,expressway-service-area,expressway-toll-station,dam,golffield,groundtrackfield,harbor,overpass,ship,stadium,storagetank,tenniscourt,trainstation,vehicle,windmill}"

# --- the two core detectors (names/dirs/hyperparams identical to the seed-0 chain) ---
ORCNN_CFG_SRC="$SCRIPT_DIR/configs_dior/oriented-rcnn-le90_r50_fpn_1x_dior.py"
RTMDET_CFG_SRC="$SCRIPT_DIR/configs_dior/rotated_rtmdet_l-3x-dior.py"
ORCNN_NAME="${ORCNN_NAME:-orcnn_dior}"
RTMDET_NAME="${RTMDET_NAME:-rtmdet_r_dior}"
ORCNN_MMR_DIR="${ORCNN_MMR_DIR:-configs/oriented_rcnn}"
RTMDET_MMR_DIR="${RTMDET_MMR_DIR:-configs/rotated_rtmdet}"
ORCNN_EPOCHS="${ORCNN_EPOCHS:-12}"          # 1x
RTMDET_EPOCHS="${RTMDET_EPOCHS:-36}"        # 3x
ORCNN_BATCH="${ORCNN_BATCH:-2}"
RTMDET_BATCH="${RTMDET_BATCH:-8}"           # batch ~8 fits 24GB (A2)
# NEW seeds only -- seed 0 already exists (existing cell). Critic asked for 2 extra seeds.
ORCNN_SEEDS="${ORCNN_SEEDS:-1,2}"
RTMDET_SEEDS="${RTMDET_SEEDS:-1,2}"

# --- results / markers ---
RESULTS_DIOR_SEEDS="${RESULTS_DIOR_SEEDS:-${AUTODL_TMP}/rotcert_dior_seeds_results}"
MARKERS_DIR="${MARKERS_DIR:-$RESULTS_DIOR_SEEDS/../dior_seeds_markers}"
[ "$DRY" -eq 1 ] || mkdir -p "$RESULTS_DIOR_SEEDS" "$MARKERS_DIR"

# --- sub-scripts / configs referenced ---
SCORE="$SCRIPT_DIR/score_rtmdet.py"
CERT_CELL="$SCRIPT_DIR/cert_cell.sh"
R20="$SCRIPT_DIR/r20_generic.py"
TRAIN_PY="$MMR/tools/train.py"

FAILED=()
mark(){ [ "$DRY" -eq 1 ] && { echo "DRY MARKER $1=$2"; return 0; }
        printf '%s\n' "$2" > "$MARKERS_DIR/$1.marker"; echo "MARKER $1=$2"; [ "$2" = OK ] || FAILED+=("$1"); return 0; }
read_marker(){ [ -f "$MARKERS_DIR/$1.marker" ] && tr -d '[:space:]' < "$MARKERS_DIR/$1.marker" || printf ''; }
stage_done(){ [ "$RESUME" -eq 1 ] && [ "$(read_marker "$1")" = OK ]; }   # rc0 iff resume & already OK

# ---- content gates (pure stdlib; identical semantics to configB / hrsc_run.sh) ----
ckpt_gate(){ [ -s "$1" ] && python3 - "$1" <<'PY'
import sys,zipfile,os
p=sys.argv[1]
try:
    z=zipfile.ZipFile(p); sys.exit(0 if z.testzip() is None else 1)
except zipfile.BadZipFile:
    sys.exit(0 if os.path.getsize(p)>1_000_000 else 1)
PY
}
jsonl_cover_gate(){ # FILE FLOOR -> ok iff non-empty, all lines JSON, distinct image_ids>=floor
  python3 - "$1" "$2" <<'PY'
import json,sys
p,floor=sys.argv[1],int(sys.argv[2]); ids=set(); n=0
try:
    for line in open(p,encoding="utf-8"):
        line=line.strip()
        if not line: continue
        r=json.loads(line); n+=1
        if r.get("image_id") is not None: ids.add(r["image_id"])
except FileNotFoundError:
    print(f"COVER_FAIL {p} missing",file=sys.stderr); sys.exit(1)
ok=n>0 and len(ids)>=floor
print(f"COVER {p}: records={n} distinct_images={len(ids)} floor={floor} passed={ok}")
sys.exit(0 if ok else 1)
PY
}

# ---- preflight: local, no side effects (runs in both modes) ----
preflight(){
  local ok=1 f
  echo "== preflight: sub-scripts, configs =="
  for f in "$SCORE" "$CERT_CELL" "$R20" "$ORCNN_CFG_SRC" "$RTMDET_CFG_SRC"; do
    if [ -f "$f" ]; then echo "OK  $f"; else echo "MISSING $f"; ok=0; fi
  done
  echo "== preflight: bash -n on shell runners =="
  for f in "$CERT_CELL" "${BASH_SOURCE[0]}"; do
    if bash -n "$f" 2>/dev/null; then echo "OK  syntax $f"; else echo "SYNTAX-FAIL $f"; ok=0; fi
  done
  echo "== preflight: DIOR configs py_compile + num_classes=20 + angle le90 =="
  for f in "$ORCNN_CFG_SRC" "$RTMDET_CFG_SRC"; do
    if python3 -m py_compile "$f" 2>/dev/null; then echo "OK  py_compile $(basename "$f")"; else echo "PYCOMPILE-FAIL $f"; ok=0; fi
    if grep -q 'num_classes=20' "$f" && ! grep -q 'num_classes=15' "$f"; then
      echo "OK  $(basename "$f") num_classes=20 (no residual 15)"
    else echo "BAD num_classes in $f"; ok=0; fi
    if grep -q "angle_version = 'le90'" "$f"; then echo "OK  $(basename "$f") angle le90"; else echo "BAD angle in $f"; ok=0; fi
  done
  echo "== preflight: seed args are the NEW seeds (must not include 0) =="
  for s in ${ORCNN_SEEDS//,/ } ${RTMDET_SEEDS//,/ }; do
    [ "$s" = 0 ] && { echo "BAD seed 0 requested (seed-0 cell already exists)"; ok=0; } || echo "OK  new seed $s"
  done
  echo "== preflight: box-only paths (reported, not failed in dry mode) =="
  for f in "$MMR" "$TRAIN_PY" "$DIOR_R_ROOT" "$DIOR_R_ROOT/$DIOR_R_TEST_IMG_SUBDIR" "$DIOR_R_TEST_IMAGESET" \
           "$DIOR_R_WORK_DIR" "$RELIABILITY_COMMONS/tools/boxkit/chain_lib.sh"; do
    if [ -e "$f" ]; then echo "present: $f"; else echo "absent (box-only, expected locally): $f"; fi
  done
  return $([ "$ok" -eq 1 ] && echo 0 || echo 1)
}

# ---- per-seed DIOR train+infer helper (mirrors configB dior_detector; loops seeds) ----
# Deploys the vendored config into its canonical $MMR/configs/<detector>/ (so the config's
# relative _base_ chain resolves), trains 1x/3x from cwd=$MMR (base dior.py's relative
# data_root 'data/DIOR/' resolves against the box symlink), content-gates the checkpoint,
# then infers the DIOR-R test split via score_rtmdet.py and content-gates the JSONL.
dior_seed(){ # NAME CFG_SRC MMR_DIR BATCH EPOCHS SEED MK_TRAIN MK_INFER
  local name="$1" cfg_src="$2" mmr_dir="$3" batch="$4" epochs="$5" seed="$6" mk_train="$7" mk_infer="$8"
  local work="$DIOR_R_WORK_DIR/${name}/seed_${seed}"
  local ckpt="$work/epoch_${epochs}.pth"
  local dets="$RESULTS_DIOR_SEEDS/dior_test_dets_${name}_seed${seed}.jsonl"
  local boxcfg="$MMR/$mmr_dir/$(basename "$cfg_src")"

  if stage_done "$mk_infer"; then echo "resume: $name seed $seed already OK -- skip"; return; fi
  mkdir -p "$MMR/$mmr_dir" "$work"
  cp "$cfg_src" "$MMR/$mmr_dir/"
  # ensure the relative data_root resolves (data/DIOR -> $DIOR_R_ROOT)
  [ -e "$MMR/data/DIOR" ] || { mkdir -p "$MMR/data"; ln -sfn "$DIOR_R_ROOT" "$MMR/data/DIOR"; }

  if [ -s "$ckpt" ] && ckpt_gate "$ckpt"; then
    echo "skip $name seed $seed train (integral ckpt exists: $ckpt)"; mark "$mk_train" OK
  else
    ( cd "$MMR" && python3 "$TRAIN_PY" "$boxcfg" --work-dir "$work" \
        --cfg-options randomness.seed=${seed} train_cfg.max_epochs=${epochs} \
        train_dataloader.batch_size=${batch} \
        default_hooks.checkpoint.max_keep_ckpts=1 default_hooks.checkpoint.save_last=True ) \
      2>&1 | tee "$RESULTS_DIOR_SEEDS/${name}_seed${seed}_train.log" || echo "warn: $name seed $seed train rc!=0 (gate authoritative)"
    if [ -s "$ckpt" ] && ckpt_gate "$ckpt"; then mark "$mk_train" OK; else mark "$mk_train" FAILED; fi
  fi

  if [ "$(read_marker "$mk_train")" != OK ]; then
    echo "REFUSE $name seed $seed infer: no integral checkpoint"; mark "$mk_infer" SKIPPED_TRAIN; return
  fi
  if [ -s "$dets" ] && jsonl_cover_gate "$dets" "$DIOR_TEST_MIN_COVER" >/dev/null 2>&1; then
    echo "skip $name seed $seed infer (dets exist): $dets"; mark "$mk_infer" OK; return
  fi
  python3 "$SCORE" --config "$boxcfg" --checkpoint "$ckpt" --mmrotate-commit "$MMROTATE_COMMIT" \
    --images-dir "$DIOR_R_ROOT/$DIOR_R_TEST_IMG_SUBDIR" --class-names "$DIOR_CLASS_NAMES" \
    --score-thr "$SCORE_THR" --device "$DEVICE" -o "$dets" \
    || echo "warn: score $name seed $seed rc!=0 (gate authoritative)"
  if jsonl_cover_gate "$dets" "$DIOR_TEST_MIN_COVER" | tee -a "$RESULTS_DIOR_SEEDS/${name}_seed${seed}_coverage.txt"; then
    mark "$mk_infer" OK; cp "$boxcfg" "$RESULTS_DIOR_SEEDS/" 2>/dev/null
  else mark "$mk_infer" FAILED; fi
}

# ================= DRY-RUN =================
if [ "$DRY" -eq 1 ]; then
  echo "=== next_boot_dior_seeds (DRY-RUN) $(date -Iseconds) ==="
  if preflight; then
    echo "PLAN stage0: gate /root/MMSTACK_OK + import probe (torch,mmcv,mmdet,mmrotate) [~0 GPU-h]"
    echo "PLAN stage1: Oriented R-CNN DIOR seeds {$ORCNN_SEEDS} (1x ${ORCNN_EPOCHS}ep batch ${ORCNN_BATCH}) train + infer test [~8 GPU-h/seed => ~16 GPU-h]"
    if [ "$SKIP_RTMDET" -eq 1 ]; then
      echo "PLAN stage2: SKIPPED (SKIP_RTMDET=1) -- RTMDet-R DIOR seeds NOT trained"
    else
      echo "PLAN stage2: RTMDet-R DIOR seeds {$RTMDET_SEEDS} (3x ${RTMDET_EPOCHS}ep batch ${RTMDET_BATCH}) train + infer test [~6-8 GPU-h/seed => ~12-16 GPU-h; 3x is CHEAP, ~10 min/ep one-stage, NOT ~24]"
    fi
    echo "PLAN stage3: chain_epilogue tars [$RESULTS_DIOR_SEEDS $MARKERS_DIR] -> marker DIOR_SEEDS_ALL_DONE"
    if [ "$SKIP_RTMDET" -eq 1 ]; then
      echo "PLAN total est: ~16 GPU-h (ORCNN-only, seeds {$ORCNN_SEEDS}); local cert: cert_cell.sh per new cell + q_hat/coverage spread extension (DIOR-SEEDS-STAGING-2026-07-14.md)"
    else
      echo "PLAN total est: ~28-32 GPU-h (2 ORCNN seeds x ~8 + 2 RTMDet seeds x ~6-8); local cert: cert_cell.sh per new cell + q_hat/coverage spread extension (DIOR-SEEDS-STAGING-2026-07-14.md)"
    fi
    echo "DIOR_SEEDS_DRYRUN_OK"; exit 0
  else
    echo "DIOR_SEEDS_DRYRUN_FAIL"; exit 1
  fi
fi

# ================= REAL RUN (box) =================
# shellcheck disable=SC1091
source "$RELIABILITY_COMMONS/tools/boxkit/chain_lib.sh"
chain_prologue

# --- env self-activation (copied verbatim from next_boot_rotcert_configB.sh) --
# The prologue path can leave python3 pointing at an env with no mmcv/cv2 -> stage-0 probe
# FAILS and gate-skips every training stage. Activate + set paths explicitly:
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate "${MMROT_ENV:-mmrot}" 2>/dev/null \
  || { echo "REFUSE: cannot activate conda env '${MMROT_ENV:-mmrot}'"; exit 1; }
export PYTHONPATH="/root/mmrotate:$RELIABILITY_COMMONS/tools/rotcert${PYTHONPATH:+:$PYTHONPATH}"
preflight || echo "preflight reported issues -- gates below are authoritative"

# ---- stage 0: env / mm-stack gate ----
echo "== stage 0: mm-stack gate =="
if stage_done STAGE0_MMSTACK; then echo "resume: stage0 OK -- skip"; else
  if [ ! -f /root/MMSTACK_OK ]; then echo "WARN: /root/MMSTACK_OK absent"; fi
  if python3 -c "import torch,mmcv,mmdet,mmrotate" 2>/dev/null; then mark STAGE0_MMSTACK OK
  else echo "REFUSE: mm-stack import probe failed"; mark STAGE0_MMSTACK FAILED; fi
fi
MMSTACK_OK=0; [ "$(read_marker STAGE0_MMSTACK)" = OK ] && MMSTACK_OK=1

# ---- stage 1: Oriented R-CNN DIOR seeds 1,2 (train 1x + infer test) ----
echo "== stage 1: Oriented R-CNN DIOR seeds {$ORCNN_SEEDS} (train 1x + infer test) [~8 GPU-h/seed] =="
if [ "$MMSTACK_OK" -eq 1 ]; then
  s1_ok=1
  for s in ${ORCNN_SEEDS//,/ }; do
    dior_seed "$ORCNN_NAME" "$ORCNN_CFG_SRC" "$ORCNN_MMR_DIR" "$ORCNN_BATCH" "$ORCNN_EPOCHS" "$s" \
              "ORCNN_TRAIN_S$s" "ORCNN_INFER_S$s"
    [ "$(read_marker "ORCNN_INFER_S$s")" = OK ] || s1_ok=0
  done
  [ "$s1_ok" -eq 1 ] && mark STAGE1_ORCNN OK || mark STAGE1_ORCNN FAILED
else
  echo "REFUSE stage1: mm-stack gate not OK"; mark STAGE1_ORCNN SKIPPED_GATE
fi

# ---- stage 2: RTMDet-R DIOR seeds 1,2 (train 3x + infer test) -- optional ----
if [ "$SKIP_RTMDET" -eq 1 ]; then
  echo "== stage 2: SKIPPED (SKIP_RTMDET=1) -- ORCNN-only arm =="
  mark STAGE2_RTMDET SKIPPED_SKIP_RTMDET
elif [ "$MMSTACK_OK" -eq 1 ]; then
  echo "== stage 2: RTMDet-R DIOR seeds {$RTMDET_SEEDS} (train 3x + infer test) [~6-8 GPU-h/seed] =="
  s2_ok=1
  for s in ${RTMDET_SEEDS//,/ }; do
    dior_seed "$RTMDET_NAME" "$RTMDET_CFG_SRC" "$RTMDET_MMR_DIR" "$RTMDET_BATCH" "$RTMDET_EPOCHS" "$s" \
              "RTMDET_TRAIN_S$s" "RTMDET_INFER_S$s"
    [ "$(read_marker "RTMDET_INFER_S$s")" = OK ] || s2_ok=0
  done
  [ "$s2_ok" -eq 1 ] && mark STAGE2_RTMDET OK || mark STAGE2_RTMDET FAILED
else
  echo "REFUSE stage2: mm-stack gate not OK"; mark STAGE2_RTMDET SKIPPED_GATE
fi

# ---- stage 3: tar + pull ----
# SKIPPED_SKIP_RTMDET is an intentional skip, not a failure -> don't taint ALL_DONE.
NONSKIP_FAILED=()
for m in "${FAILED[@]:-}"; do [ -n "$m" ] && [ "$m" != STAGE2_RTMDET ] && NONSKIP_FAILED+=("$m"); done
if [ "$SKIP_RTMDET" -eq 1 ] && [ "${#NONSKIP_FAILED[@]}" -eq 0 ]; then marker_name="DIOR_SEEDS_ALL_DONE"
elif [ "${#FAILED[@]}" -eq 0 ]; then marker_name="DIOR_SEEDS_ALL_DONE"
else marker_name="DIOR_SEEDS_PARTIAL_DONE"; echo "DIOR SEEDS PARTIAL -- failed: ${FAILED[*]}"; fi
chain_epilogue "$RESULTS_DIOR_SEEDS $MARKERS_DIR" "$marker_name" "rotcert_dior_seeds"

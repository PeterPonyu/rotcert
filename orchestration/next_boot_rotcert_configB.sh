#!/bin/bash
# next_boot_rotcert_configB.sh -- Config-B one-command box runner (2026-07-13).
#
# Config B = Config A PLUS two depth arms (compute plan sec.2 "Config B"):
#   (i)  cross-family detector breadth on the HARD dataset: train RoI Transformer AND
#        S2A-Net on DIOR-R in-house (1x schedule, ~8 GPU-h each), then infer the DIOR-R
#        held-out TEST split so the new cells join the existing DIOR grid.
#   (ii) cheapest training-variance demonstration: 3-seed (0,1,2) HRSC training for the
#        two existing detectors (Oriented R-CNN 1x, RTMDet-R 3x), ~3 GPU-h total (HRSC
#        is tiny, ~436 train imgs).
#
# NOTE on S2A-Net admissibility: the DOTA zoo EXCLUDED S2A-Net from its inference arm
# because the released S2A-Net *checkpoint* is v0.1.0-era and incompatible with mmrotate
# dev-1.x. In-house TRAINING with a dev-1.x config (configs_dior/s2anet-le135_...dior.py)
# has NO such incompatibility -- we build + train the model from the dev-1.x config and
# never load a v0.1.0 checkpoint. Same for RoI Transformer.
#
# "infer val": the two existing DIOR-R cells were certified on the DIOR-R held-out TEST
# split (trainval is consumed by training -> scoring it would be train-on-test). For the
# new RoI-Trans/S2A-Net cells to join the SAME coverage-matched grid (same scenes, same
# matcher, same GT), they are likewise scored on the DIOR-R TEST split. The GT is the
# frozen dior_test_gt.jsonl (already pulled); this chain also re-emits it for a
# self-contained tar.
#
# Stages (each content-gated -- assert checkpoint files exist + inference JSONs non-empty,
# NEVER bare exit codes -- with its own per-stage sentinel so --resume skips completed
# stages):
#   Stage 0  env / mm-stack gate       (MMSTACK_OK sentinel + import probe)
#   Stage 1  DIOR-R staged-data verify (counts vs staged layout) + test GT emit
#   Stage 2  train RoI Transformer DIOR (1x) + infer DIOR-R test
#   Stage 3  train S2A-Net DIOR (1x)        + infer DIOR-R test
#   Stage 4  HRSC 3-seed x 2 detectors: unzip + hrsc_run.sh (ORCNN_SEEDS/RTMDET_SEEDS)
#   Stage 5  tar results + ALL_DONE sentinel (chain_epilogue)
#
# Certification of the new cells (cert_cell.sh per cell) and the coverage-matched regime
# extension are the LOCAL, CPU, post-pull follow-up -- this chain only makes the GPU
# substrate.
#
# --dry-run validates every referenced sub-script/config/path locally, prints the plan
# (incl. est. GPU-hours per stage), and exits 0 iff all preflight checks pass -- WITHOUT
# any GPU, download, unzip, box, or chain_lib dependency. Run it locally before booting.
#
# Usage on the box:  bash next_boot_rotcert_configB.sh   (add --resume to skip done stages)
# Usage locally:     bash next_boot_rotcert_configB.sh --dry-run
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
export MMROTATE_COMMIT

# --- DIOR-R staged layout (mirrors next_boot_rotcert_dior_{train,infer}.sh) ---
DIOR_R_ROOT="${DIOR_R_ROOT:-${AUTODL_TMP}/dior_r}"
DIOR_R_OBB_ANN_SUBDIR="${DIOR_R_OBB_ANN_SUBDIR:-Annotations/Oriented Bounding Boxes}"
DIOR_R_TRAINVAL_IMG_SUBDIR="${DIOR_R_TRAINVAL_IMG_SUBDIR:-JPEGImages-trainval}"
DIOR_R_TEST_IMG_SUBDIR="${DIOR_R_TEST_IMG_SUBDIR:-JPEGImages-test}"
DIOR_R_IMAGESETS_SUBDIR="${DIOR_R_IMAGESETS_SUBDIR:-ImageSets/Main}"
DIOR_R_TEST_IMAGESET="${DIOR_R_TEST_IMAGESET:-$DIOR_R_ROOT/$DIOR_R_IMAGESETS_SUBDIR/test.txt}"
DIOR_R_WORK_DIR="${DIOR_R_WORK_DIR:-$DIOR_R_ROOT/work_dirs}"
DIOR_R_MIN_OBB_XMLS="${DIOR_R_MIN_OBB_XMLS:-23000}"
DIOR_R_MIN_TRAINVAL_JPGS="${DIOR_R_MIN_TRAINVAL_JPGS:-11000}"
DIOR_R_MIN_TEST_JPGS="${DIOR_R_MIN_TEST_JPGS:-11000}"
DIOR_R_IMG_EXT="${DIOR_R_IMG_EXT:-jpg}"
DIOR_TEST_MIN_COVER="${DIOR_TEST_MIN_COVER:-11000}"    # distinct images that must appear in dets (of 11,738)
DIOR_GT_MAX_DEGENERATE="${DIOR_GT_MAX_DEGENERATE:-0}"
DIOR_DIFFICULT_POLICY="${DIOR_DIFFICULT_POLICY:-include}"
DIOR_ANGLE_UNIT="${DIOR_ANGLE_UNIT:-degrees}"
export DIOR_DIFFICULT_POLICY DIOR_ANGLE_UNIT
# DIOR-R 20 classes, index-aligned to the trained model's METAINFO label order (VERIFIED
# 2026-07-11 against the box's pinned mmrotate DIORDataset.METAINFO -- a transposed order
# silently mislabels every detection). Identical to next_boot_rotcert_dior_infer.sh.
DIOR_CLASS_NAMES="${DIOR_CLASS_NAMES:-airplane,airport,baseballfield,basketballcourt,bridge,chimney,expressway-service-area,expressway-toll-station,dam,golffield,groundtrackfield,harbor,overpass,ship,stadium,storagetank,tenniscourt,trainstation,vehicle,windmill}"

# DIOR detector configs (this deliverable) + train hyperparams (1x for both).
ROI_TRANS_CFG_SRC="$SCRIPT_DIR/configs_dior/roi-trans-le90_r50_fpn_1x_dior.py"
S2ANET_CFG_SRC="$SCRIPT_DIR/configs_dior/s2anet-le135_r50_fpn_1x_dior.py"
ROI_TRANS_MMR_DIR="${ROI_TRANS_MMR_DIR:-configs/roi_trans}"
S2ANET_MMR_DIR="${S2ANET_MMR_DIR:-configs/s2anet}"
DIOR_EPOCHS="${DIOR_EPOCHS:-12}"                       # 1x
ROI_TRANS_BATCH="${ROI_TRANS_BATCH:-2}"               # two-stage @ 800px on 24GB
S2ANET_BATCH="${S2ANET_BATCH:-2}"                     # one-stage; 2 is the schedule_1x/dota base
DIOR_SEED="${DIOR_SEED:-0}"

# --- HRSC 3-seed arm (delegates to hrsc_run.sh) ---
HRSC_MS_ZIP="${HRSC_MS_ZIP:-${AUTODL_TMP}/HRSC2016-MS.zip}"
HRSC_ROOT="${HRSC_ROOT:-${AUTODL_TMP}/HRSC2016-MS}"
HRSC_MS_MD5="${HRSC_MS_MD5:-167501c0de0d6015a109a4abddd77fb1}"       # verified local prefetch 2026-07-13
export HRSC_ROOT
export ORCNN_SEEDS="${ORCNN_SEEDS:-0,1,2}"            # Config B: 3-seed training-variance
export RTMDET_SEEDS="${RTMDET_SEEDS:-0,1,2}"
RESULTS_HRSC="${RESULTS_HRSC:-${AUTODL_TMP}/hrsc_rotcert_results}"

# --- results / markers ---
RESULTS_DIOR="${RESULTS_DIOR:-${AUTODL_TMP}/rotcert_configB_dior_results}"
MARKERS_DIR="${MARKERS_DIR:-$RESULTS_DIOR/../configB_markers}"
[ "$DRY" -eq 1 ] || mkdir -p "$RESULTS_DIOR" "$MARKERS_DIR"

# --- sub-scripts / configs referenced ---
HRSC_RUN="$SCRIPT_DIR/hrsc_run.sh"
SCORE="$SCRIPT_DIR/score_rtmdet.py"
PREP_DIOR_GT="$SCRIPT_DIR/prepare_dior_gt.py"
CERT_CELL="$SCRIPT_DIR/cert_cell.sh"
R20="$SCRIPT_DIR/r20_generic.py"
HRSC_CFGS=("$SCRIPT_DIR/configs_hrsc/oriented-rcnn-le90_r50_fpn_1x_hrsc.py"
           "$SCRIPT_DIR/configs_hrsc/rotated-rtmdet_l-3x-hrsc.py")
TRAIN_PY="$MMR/tools/train.py"

FAILED=()
mark(){ [ "$DRY" -eq 1 ] && { echo "DRY MARKER $1=$2"; return 0; }
        printf '%s\n' "$2" > "$MARKERS_DIR/$1.marker"; echo "MARKER $1=$2"; [ "$2" = OK ] || FAILED+=("$1"); return 0; }
read_marker(){ [ -f "$MARKERS_DIR/$1.marker" ] && tr -d '[:space:]' < "$MARKERS_DIR/$1.marker" || printf ''; }
stage_done(){ [ "$RESUME" -eq 1 ] && [ "$(read_marker "$1")" = OK ]; }   # rc0 iff resume & already OK
sentinel(){ grep -q "$2" "$1" 2>/dev/null; }

# ---- content gates (pure stdlib; identical semantics to hrsc_run.sh) ----
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
  for f in "$HRSC_RUN" "$SCORE" "$PREP_DIOR_GT" "$CERT_CELL" "$R20" \
           "$ROI_TRANS_CFG_SRC" "$S2ANET_CFG_SRC" "${HRSC_CFGS[@]}"; do
    if [ -f "$f" ]; then echo "OK  $f"; else echo "MISSING $f"; ok=0; fi
  done
  echo "== preflight: bash -n on shell runners =="
  for f in "$HRSC_RUN" "$CERT_CELL" "${BASH_SOURCE[0]}"; do
    if bash -n "$f" 2>/dev/null; then echo "OK  syntax $f"; else echo "SYNTAX-FAIL $f"; ok=0; fi
  done
  echo "== preflight: new DIOR configs py_compile + num_classes=20 + le-version =="
  for f in "$ROI_TRANS_CFG_SRC" "$S2ANET_CFG_SRC"; do
    if python3 -m py_compile "$f" 2>/dev/null; then echo "OK  py_compile $(basename "$f")"; else echo "PYCOMPILE-FAIL $f"; ok=0; fi
    if grep -q 'num_classes=20' "$f" && ! grep -q 'num_classes=15' "$f"; then
      echo "OK  $(basename "$f") num_classes=20 (no residual 15)"
    else echo "BAD num_classes in $f"; ok=0; fi
  done
  grep -q "angle_version = 'le90'"  "$ROI_TRANS_CFG_SRC" && echo "OK  roi-trans angle le90" || { echo "BAD roi-trans angle"; ok=0; }
  grep -q "angle_version = 'le135'" "$S2ANET_CFG_SRC"   && echo "OK  s2anet angle le135" || { echo "BAD s2anet angle"; ok=0; }
  echo "== preflight: box-only paths (reported, not failed in dry mode) =="
  for f in "$MMR" "$TRAIN_PY" "$DIOR_R_ROOT" "$DIOR_R_ROOT/$DIOR_R_TEST_IMG_SUBDIR" "$DIOR_R_TEST_IMAGESET" \
           "$HRSC_MS_ZIP" "$HRSC_ROOT" "$RELIABILITY_COMMONS/tools/boxkit/chain_lib.sh"; do
    if [ -e "$f" ]; then echo "present: $f"; else echo "absent (box-only, expected locally): $f"; fi
  done
  return $([ "$ok" -eq 1 ] && echo 0 || echo 1)
}

# ---- DIOR train+infer helper (RoI-Trans / S2A-Net) ----
# Deploys the vendored config into $MMR, trains 1x (cwd=$MMR so base dior.py's relative
# data_root 'data/DIOR/' resolves against the box symlink), content-gates the checkpoint,
# then infers the DIOR-R test split via score_rtmdet.py and content-gates the JSONL.
dior_detector(){ # NAME CFG_SRC MMR_DIR BATCH TRAIN_MARKER INFER_MARKER
  local name="$1" cfg_src="$2" mmr_dir="$3" batch="$4" mk_train="$5" mk_infer="$6"
  local work="$DIOR_R_WORK_DIR/${name}_dior/seed_${DIOR_SEED}"
  local ckpt="$work/epoch_${DIOR_EPOCHS}.pth"
  local dets="$RESULTS_DIOR/dior_test_dets_${name}.jsonl"
  local boxcfg="$MMR/$mmr_dir/$(basename "$cfg_src")"

  if stage_done "$mk_infer"; then echo "resume: $name DIOR already OK -- skip"; return; fi
  mkdir -p "$MMR/$mmr_dir" "$work"
  cp "$cfg_src" "$MMR/$mmr_dir/"
  # ensure the relative data_root resolves (data/DIOR -> $DIOR_R_ROOT)
  [ -e "$MMR/data/DIOR" ] || { mkdir -p "$MMR/data"; ln -sfn "$DIOR_R_ROOT" "$MMR/data/DIOR"; }

  if [ -s "$ckpt" ] && ckpt_gate "$ckpt"; then
    echo "skip $name train (integral ckpt exists: $ckpt)"; mark "$mk_train" OK
  else
    ( cd "$MMR" && python3 "$TRAIN_PY" "$boxcfg" --work-dir "$work" \
        --cfg-options randomness.seed=${DIOR_SEED} train_cfg.max_epochs=${DIOR_EPOCHS} \
        train_dataloader.batch_size=${batch} \
        default_hooks.checkpoint.max_keep_ckpts=1 default_hooks.checkpoint.save_last=True ) \
      2>&1 | tee "$RESULTS_DIOR/${name}_train.log" || echo "warn: $name train rc!=0 (gate authoritative)"
    if [ -s "$ckpt" ] && ckpt_gate "$ckpt"; then mark "$mk_train" OK; else mark "$mk_train" FAILED; fi
  fi

  if [ "$(read_marker "$mk_train")" != OK ]; then
    echo "REFUSE $name infer: no integral checkpoint"; mark "$mk_infer" SKIPPED_TRAIN; return
  fi
  if [ -s "$dets" ] && jsonl_cover_gate "$dets" "$DIOR_TEST_MIN_COVER" >/dev/null 2>&1; then
    echo "skip $name infer (dets exist): $dets"; mark "$mk_infer" OK; return
  fi
  python3 "$SCORE" --config "$boxcfg" --checkpoint "$ckpt" --mmrotate-commit "$MMROTATE_COMMIT" \
    --images-dir "$DIOR_R_ROOT/$DIOR_R_TEST_IMG_SUBDIR" --class-names "$DIOR_CLASS_NAMES" \
    --score-thr "$SCORE_THR" --device "$DEVICE" -o "$dets" \
    || echo "warn: score $name rc!=0 (gate authoritative)"
  if jsonl_cover_gate "$dets" "$DIOR_TEST_MIN_COVER" | tee -a "$RESULTS_DIOR/${name}_coverage.txt"; then
    mark "$mk_infer" OK; cp "$boxcfg" "$RESULTS_DIOR/" 2>/dev/null
  else mark "$mk_infer" FAILED; fi
}

# ================= DRY-RUN =================
if [ "$DRY" -eq 1 ]; then
  echo "=== next_boot_rotcert_configB (DRY-RUN) $(date -Iseconds) ==="
  if preflight; then
    echo "PLAN stage0: gate /root/MMSTACK_OK + import probe (torch,mmcv,mmdet,mmrotate)"
    echo "PLAN stage1: verify DIOR-R staged data ($DIOR_R_ROOT: >=$DIOR_R_MIN_OBB_XMLS OBB xmls, >=$DIOR_R_MIN_TRAINVAL_JPGS trainval, >=$DIOR_R_MIN_TEST_JPGS test) + emit DIOR test GT [~0 GPU-h, CPU]"
    echo "PLAN stage2: deploy roi-trans DIOR cfg -> $MMR/$ROI_TRANS_MMR_DIR; train 1x (${DIOR_EPOCHS}ep batch ${ROI_TRANS_BATCH} seed ${DIOR_SEED}) + infer DIOR-R test [~8 GPU-h]"
    echo "PLAN stage3: deploy s2anet DIOR cfg -> $MMR/$S2ANET_MMR_DIR; train 1x (${DIOR_EPOCHS}ep batch ${S2ANET_BATCH} seed ${DIOR_SEED}) + infer DIOR-R test [~8 GPU-h]"
    echo "PLAN stage4: HRSC unzip ($HRSC_MS_ZIP) + hrsc_run.sh ORCNN_SEEDS=$ORCNN_SEEDS RTMDET_SEEDS=$RTMDET_SEEDS -> sentinel HRSC_RUN_ALL_DONE [~3 GPU-h total for 3 seeds x 2 detectors]"
    echo "PLAN stage5: chain_epilogue tars [$RESULTS_DIOR $RESULTS_HRSC $MARKERS_DIR] -> marker ROTCERT_CONFIGB_ALL_DONE"
    echo "PLAN total est: ~19 GPU-h (8 RoI-Trans + 8 S2A-Net DIOR + ~3 HRSC 3-seed); local cert: cert_cell.sh per new cell then coverage-matched regime extension"
    echo "CONFIGB_DRYRUN_OK"; exit 0
  else
    echo "CONFIGB_DRYRUN_FAIL"; exit 1
  fi
fi

# ================= REAL RUN (box) =================
# shellcheck disable=SC1091
source "$RELIABILITY_COMMONS/tools/boxkit/chain_lib.sh"
chain_prologue

# --- env self-activation (2026-07-14 fix; UPDATE-74 lessons (d)/(e) applied) --
# The first launch relied on the CALLER's conda env; the prologue path left
# python3 pointing at an env with no mmcv/cv2 -> stage-0 probe FAILED and
# prepare_dior_gt skipped all 124,445 objects ("No module named 'cv2'"),
# gate-skipping BOTH DIOR training stages. Activate + set paths explicitly:
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

# ---- stage 1: DIOR-R staged-data verify + test GT emit ----
echo "== stage 1: DIOR-R staged-data verify + test GT =="
if stage_done STAGE1_DIOR_DATA; then echo "resume: stage1 OK -- skip"; else
  obb_dir="$DIOR_R_ROOT/$DIOR_R_OBB_ANN_SUBDIR"
  n_xml=$(find "$obb_dir" -type f -name '*.xml' 2>/dev/null | wc -l | tr -d ' ')
  n_trainval=$(find "$DIOR_R_ROOT/$DIOR_R_TRAINVAL_IMG_SUBDIR" -type f -name "*.$DIOR_R_IMG_EXT" 2>/dev/null | wc -l | tr -d ' ')
  n_test=$(find "$DIOR_R_ROOT/$DIOR_R_TEST_IMG_SUBDIR" -type f -name "*.$DIOR_R_IMG_EXT" 2>/dev/null | wc -l | tr -d ' ')
  echo "obb_xmls=$n_xml (floor $DIOR_R_MIN_OBB_XMLS) trainval=$n_trainval (floor $DIOR_R_MIN_TRAINVAL_JPGS) test=$n_test (floor $DIOR_R_MIN_TEST_JPGS)"
  data_ok=1
  [ "$n_xml" -ge "$DIOR_R_MIN_OBB_XMLS" ] || { echo "GATE_FAIL OBB xmls"; data_ok=0; }
  [ "$n_trainval" -ge "$DIOR_R_MIN_TRAINVAL_JPGS" ] || { echo "GATE_FAIL trainval jpgs"; data_ok=0; }
  [ "$n_test" -ge "$DIOR_R_MIN_TEST_JPGS" ] || { echo "GATE_FAIL test jpgs"; data_ok=0; }
  [ "$data_ok" -eq 1 ] && mark STAGE1_DIOR_DATA OK || mark STAGE1_DIOR_DATA FAILED
  # emit DIOR-R test GT (deterministic; reproduces the frozen dior_test_gt.jsonl) for a self-contained tar
  DIOR_TEST_GT="$RESULTS_DIOR/dior_test_gt.jsonl"
  if [ "$data_ok" -eq 1 ] && [ ! -s "$DIOR_TEST_GT" ]; then
    imageset_arg=(); [ -f "$DIOR_R_TEST_IMAGESET" ] && imageset_arg=(--imageset-file "$DIOR_R_TEST_IMAGESET")
    python3 "$PREP_DIOR_GT" --annfiles-dir "$obb_dir" "${imageset_arg[@]}" \
      --difficult-policy "$DIOR_DIFFICULT_POLICY" --angle-unit "$DIOR_ANGLE_UNIT" -o "$DIOR_TEST_GT" \
      2>&1 | tee "$RESULTS_DIOR/dior_test_gt_prep.log" || echo "warn: dior gt prep rc!=0"
  fi
  [ -s "$DIOR_TEST_GT" ] && mark DIOR_TEST_GT OK || mark DIOR_TEST_GT FAILED
fi

# ---- stage 2: RoI Transformer DIOR train + infer ----
echo "== stage 2: RoI Transformer DIOR (train 1x + infer test) [~8 GPU-h] =="
if [ "$MMSTACK_OK" -eq 1 ] && [ "$(read_marker STAGE1_DIOR_DATA)" = OK ]; then
  dior_detector roi_trans "$ROI_TRANS_CFG_SRC" "$ROI_TRANS_MMR_DIR" "$ROI_TRANS_BATCH" ROI_TRANS_TRAIN ROI_TRANS_INFER
  [ "$(read_marker ROI_TRANS_INFER)" = OK ] && mark STAGE2_ROI_TRANS OK || mark STAGE2_ROI_TRANS FAILED
else
  echo "REFUSE stage2: mm-stack or DIOR data gate not OK"; mark STAGE2_ROI_TRANS SKIPPED_GATE
fi

# ---- stage 3: S2A-Net DIOR train + infer ----
echo "== stage 3: S2A-Net DIOR (train 1x + infer test) [~8 GPU-h] =="
if [ "$MMSTACK_OK" -eq 1 ] && [ "$(read_marker STAGE1_DIOR_DATA)" = OK ]; then
  dior_detector s2anet "$S2ANET_CFG_SRC" "$S2ANET_MMR_DIR" "$S2ANET_BATCH" S2ANET_TRAIN S2ANET_INFER
  [ "$(read_marker S2ANET_INFER)" = OK ] && mark STAGE3_S2ANET OK || mark STAGE3_S2ANET FAILED
else
  echo "REFUSE stage3: mm-stack or DIOR data gate not OK"; mark STAGE3_S2ANET SKIPPED_GATE
fi

# ---- stage 4: HRSC 3-seed x 2 detectors (unzip + hrsc_run.sh) ----
echo "== stage 4: HRSC 3-seed (ORCNN_SEEDS=$ORCNN_SEEDS RTMDET_SEEDS=$RTMDET_SEEDS) [~3 GPU-h] =="
if stage_done STAGE4_HRSC; then echo "resume: stage4 OK -- skip"; else
  if [ ! -f "$HRSC_ROOT/ImageSets/source/train.txt" ]; then
    if [ -s "$HRSC_MS_ZIP" ]; then
      if command -v md5sum >/dev/null && [ -n "$HRSC_MS_MD5" ]; then
        got=$(md5sum "$HRSC_MS_ZIP" | cut -d' ' -f1)
        [ "$got" = "$HRSC_MS_MD5" ] && echo "HRSC zip md5 OK ($got)" || echo "WARN HRSC zip md5 $got != $HRSC_MS_MD5"
      fi
      mkdir -p "$HRSC_ROOT"; unzip -q -o "$HRSC_MS_ZIP" -d "$HRSC_ROOT" || echo "warn: unzip rc!=0 (gate below authoritative)"
    else
      echo "HRSC zip absent at $HRSC_MS_ZIP and root not populated"
    fi
  fi
  nann=$(ls "$HRSC_ROOT/Annotations"/*.xml 2>/dev/null | wc -l)
  if [ -f "$HRSC_ROOT/ImageSets/source/train.txt" ] && [ "$nann" -ge 1000 ]; then
    HRSC_LOG=/root/hrsc_run.log
    bash "$HRSC_RUN" || echo "warn: hrsc_run rc!=0 (sentinel/gate authoritative)"
    if sentinel "$HRSC_LOG" HRSC_RUN_ALL_DONE; then mark STAGE4_HRSC OK
    elif sentinel "$HRSC_LOG" HRSC_RUN_PARTIAL_DONE; then mark STAGE4_HRSC PARTIAL
    else mark STAGE4_HRSC FAILED; fi
  else
    echo "REFUSE hrsc_run: HRSC unzip gate not OK"; mark STAGE4_HRSC SKIPPED_UNZIP
  fi
fi

# ---- stage 5: tar + pull ----
if [ "${#FAILED[@]}" -eq 0 ]; then marker_name="ROTCERT_CONFIGB_ALL_DONE"
else marker_name="ROTCERT_CONFIGB_PARTIAL_DONE"; echo "CONFIGB PARTIAL -- failed: ${FAILED[*]}"; fi
chain_epilogue "$RESULTS_DIOR $RESULTS_HRSC $MARKERS_DIR" "$marker_name" "rotcert_configB"

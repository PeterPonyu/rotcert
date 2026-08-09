#!/bin/bash
# hrsc_run.sh -- rotcert HRSC2016-MS 3rd-dataset arm, box-side train+infer chain
# (2026-07-13). Runs on the AutoDL box after: (a) mm-stack built (/root/MMSTACK_OK),
# (b) HRSC2016-MS unzipped at $HRSC_ROOT, (c) reliability-commons pushed. Produces the
# HRSC substrate the LOCAL certification arm consumes: per-detector detections-JSONL over
# the HRSC (canonical source) TEST split + a canonical single-class 'ship' GT JSONL.
#
# Content gates (never bare exit codes): convert HBB-crosscheck rc + counts; smoke
# finite-loss + checkpoint; per-detector checkpoint integrity; dets-JSONL coverage;
# GT provenance. Disk-frugal: max_keep_ckpts=1, no visualization (30GB system disk only).
# Marker: HRSC_RUN_ALL_DONE (log line, after the tar). No auto-shutdown (teardown is via
# the boxkit API from the local side).
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh; conda activate mmrot
export PYTHONPATH=/root/reliability-commons/tools/rotcert:${PYTHONPATH:-}
LOG=/root/hrsc_run.log
exec > >(tee -a "$LOG") 2>&1
echo "=== hrsc_run start $(date -Iseconds) ==="

MMROTATE_COMMIT=3ff004eb21ea040455b5585db229edba4037f1bf
HRSC_ROOT="${HRSC_ROOT:-/root/autodl-tmp/HRSC2016-MS}"
DOTA="${DOTA:-/root/autodl-tmp/hrsc_dota}"
MMR="${MMR:-/root/mmrotate}"
ORCH=/root/reliability-commons/tools/rotcert/orchestration
RES=/root/autodl-tmp/hrsc_rotcert_results
WORK="$DOTA/work_dirs"
MK="$RES/markers"
mkdir -p "$RES" "$MK" "$WORK"
ORCNN_EPOCHS="${ORCNN_EPOCHS:-12}"
RTMDET_EPOCHS="${RTMDET_EPOCHS:-36}"
ORCNN_SEEDS="${ORCNN_SEEDS:-0}"     # comma-separated; Config B multi-seed (e.g. 0,1,2)
RTMDET_SEEDS="${RTMDET_SEEDS:-0}"   # comma-separated; tiny dataset -> cheapest detector-seed variance
SMOKE_OK=0                          # defined up front so a convert-refuse cleanly blocks GPU stages
FAILED=()
mark(){ printf '%s\n' "$2" > "$MK/$1.marker"; echo "MARKER $1=$2"; { [ "$2" = FAILED ] || [ "$2" = REFUSED ]; } && FAILED+=("$1"); return 0; }

# ---- gate helpers (pure stdlib) ----
finite_loss_gate(){ # LOGFILE -> ok iff >=1 finite loss line and NO 'nan'/'inf' loss
  python - "$1" <<'PY'
import re,sys
p=sys.argv[1]; seen=0; bad=0
try:
    for ln in open(p,encoding="utf-8",errors="ignore"):
        m=re.search(r"loss:\s*([0-9eE.+-]+|nan|inf)",ln)
        if m:
            v=m.group(1); seen+=1
            if v in ("nan","inf") or (v.replace('.','',1).replace('-','',1).isdigit() and float(v)!=float(v)):
                bad+=1
except FileNotFoundError:
    print("FINITE_LOSS_FAIL: log missing",file=sys.stderr); sys.exit(1)
ok = seen>0 and bad==0
print(f"FINITE_LOSS seen={seen} bad={bad} passed={ok}")
sys.exit(0 if ok else 1)
PY
}
jsonl_cover_gate(){ # FILE FLOOR -> ok iff non-empty, all lines JSON, distinct image_ids>=floor
  python - "$1" "$2" <<'PY'
import json,sys
p,floor=sys.argv[1],int(sys.argv[2]); ids=set(); n=0
try:
    for ln,line in enumerate(open(p,encoding="utf-8"),1):
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
ckpt_gate(){ [ -s "$1" ] && python - "$1" <<'PY'
import sys,zipfile,os
p=sys.argv[1]
try:
    z=zipfile.ZipFile(p); bad=z.testzip()
    sys.exit(0 if bad is None else 1)
except zipfile.BadZipFile:
    sys.exit(0 if os.path.getsize(p)>1_000_000 else 1)
PY
}

# ---- gate 0: mm-stack present ----
if [ ! -f /root/MMSTACK_OK ]; then echo "REFUSE: /root/MMSTACK_OK absent"; mark MMSTACK REFUSED; fi
python -c "import torch,mmcv,mmdet,mmrotate" 2>/dev/null && mark MMSTACK OK || { mark MMSTACK FAILED; }

# ---- stage 1: convert HRSC-MS -> DOTA (canonical source splits) ----
echo "== stage 1: convert (source train=436 / test=453) =="
# prepare_hrsc.py returns rc=2 on the angle-convention refuse (median corner-HBB-vs-XML-HBB
# IoU below floor -> a sign/transpose bug); we surface that as REFUSED (distinct from a
# generic FAILED) and hard-block every GPU stage below. Correct convention scores ~0.81;
# verified locally 2026-07-13 (train 0.8242 / test 0.8146). CPU, before any GPU spend.
conv_ok=1; conv_refused=0
for split in train test; do
  if [ -d "$DOTA/$split/images" ] && [ "$(ls "$DOTA/$split/images" 2>/dev/null | wc -l)" -ge 400 ]; then
    echo "skip convert $split (exists)"
  else
    python "$ORCH/prepare_hrsc.py" --hrsc-root "$HRSC_ROOT" \
      --imageset-file "$HRSC_ROOT/ImageSets/source/$split.txt" --out-dir "$DOTA/$split"
    rc=$?
    if [ "$rc" -eq 2 ]; then echo "ANGLE-CONVENTION REFUSE on $split (prepare_hrsc rc=2)"; conv_refused=1; conv_ok=0;
    elif [ "$rc" -ne 0 ]; then conv_ok=0; fi
  fi
done
ntr=$(ls "$DOTA/train/images"/*.png 2>/dev/null | wc -l)
nte=$(ls "$DOTA/test/images"/*.png 2>/dev/null | wc -l)
echo "converted: train_imgs=$ntr test_imgs=$nte"
if [ "$conv_refused" -eq 1 ]; then mark CONVERT REFUSED
elif [ "$conv_ok" -eq 1 ] && [ "$ntr" -ge 400 ] && [ "$nte" -ge 400 ]; then mark CONVERT OK
else mark CONVERT FAILED; fi
CONVERT_OK=0; [ "$(cat "$MK/CONVERT.marker" 2>/dev/null)" = OK ] && CONVERT_OK=1

# ---- stage 2: deploy configs + prefetch pretrained backbones ----
echo "== stage 2: deploy configs =="
mkdir -p "$MMR/configs/oriented_rcnn" "$MMR/configs/rotated_rtmdet"
cp "$ORCH/configs_hrsc/oriented-rcnn-le90_r50_fpn_1x_hrsc.py" "$MMR/configs/oriented_rcnn/"
cp "$ORCH/configs_hrsc/rotated-rtmdet_l-3x-hrsc.py" "$MMR/configs/rotated_rtmdet/"
ORCNN_CFG="$MMR/configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_hrsc.py"
RTMDET_CFG="$MMR/configs/rotated_rtmdet/rotated-rtmdet_l-3x-hrsc.py"
# prefetch backbones to the torch hub cache (plain curl + proxy fallback; no heredocs).
CKD=/root/.cache/torch/hub/checkpoints
mkdir -p "$CKD"
fetch_ckpt(){ # URL DEST
  local url="$1" dest="$2"
  [ -s "$dest" ] && { echo "cached: $dest"; return 0; }
  curl -fsSL --max-time 180 "$url" -o "$dest" && [ -s "$dest" ] && { echo "fetched direct: $dest"; return 0; }
  ( source /etc/network_turbo 2>/dev/null; curl -fsSL --max-time 180 "$url" -o "$dest" )
  [ -s "$dest" ] && echo "fetched via proxy: $dest" || echo "WARN: could not prefetch $url (train may fetch it itself)"
}
fetch_ckpt "https://download.pytorch.org/models/resnet50-0676ba61.pth" "$CKD/resnet50-0676ba61.pth"
fetch_ckpt "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-l_8xb256-rsb-a1-600e_in1k-6a760974.pth" "$CKD/cspnext-l_8xb256-rsb-a1-600e_in1k-6a760974.pth"

COMMON_OPTS="default_hooks.checkpoint.max_keep_ckpts=1 default_hooks.checkpoint.save_last=True"

# ---- stage 3: smoke (1 epoch each; finite-loss + checkpoint gate) ----
run_train(){ # NAME CONFIG EPOCHS WORKDIR MARKER_PREFIX [SEED]
  local name="$1" cfg="$2" ep="$3" wd="$4" mp="$5" seed="${6:-0}"
  local ck="$wd/epoch_${ep}.pth"
  if [ -s "$ck" ] && ckpt_gate "$ck"; then echo "skip $name (ckpt exists)"; mark "${mp}_TRAIN" OK; mark "${mp}_CKPT" OK; return; fi
  # shellcheck disable=SC2086
  python "$MMR/tools/train.py" "$cfg" --work-dir "$wd" \
    --cfg-options randomness.seed=${seed} train_cfg.max_epochs=${ep} $COMMON_OPTS \
    2>&1 | tee "$RES/${name}_train.log" || echo "warn: $name train rc!=0 (gate authoritative)"
  if finite_loss_gate "$RES/${name}_train.log"; then mark "${mp}_LOSS" OK; else mark "${mp}_LOSS" FAILED; fi
  if [ -s "$ck" ] && ckpt_gate "$ck"; then mark "${mp}_TRAIN" OK; mark "${mp}_CKPT" OK; else mark "${mp}_TRAIN" FAILED; mark "${mp}_CKPT" FAILED; fi
}

echo "== stage 3: smoke (1 epoch each) =="
if [ "$CONVERT_OK" -eq 1 ]; then
  run_train orcnn_smoke  "$ORCNN_CFG"  1 "$RES/smoke_orcnn"  SMOKE_ORCNN 0
  run_train rtmdet_smoke "$RTMDET_CFG" 1 "$RES/smoke_rtmdet" SMOKE_RTMDET 0
  SMOKE_OK=1
  [ "$(cat "$MK/SMOKE_ORCNN_CKPT.marker" 2>/dev/null)" = OK ] || SMOKE_OK=0
  [ "$(cat "$MK/SMOKE_RTMDET_CKPT.marker" 2>/dev/null)" = OK ] || SMOKE_OK=0
  rm -rf "$RES/smoke_orcnn" "$RES/smoke_rtmdet"   # reclaim disk; full runs are fresh
else
  echo "REFUSE smoke: convert gate not OK ($(cat "$MK/CONVERT.marker" 2>/dev/null))"
  mark SMOKE_ORCNN_CKPT SKIPPED_CONVERT; mark SMOKE_RTMDET_CKPT SKIPPED_CONVERT; SMOKE_OK=0
fi

# ---- stage 4: full train (per-seed; Config B multi-seed via ORCNN_SEEDS/RTMDET_SEEDS) ----
if [ "$SMOKE_OK" -eq 1 ]; then
  echo "== stage 4: full train (ORCNN ${ORCNN_EPOCHS}ep seeds=$ORCNN_SEEDS; RTMDet ${RTMDET_EPOCHS}ep seeds=$RTMDET_SEEDS) =="
  for s in ${ORCNN_SEEDS//,/ }; do
    run_train orcnn "$ORCNN_CFG" "$ORCNN_EPOCHS" "$WORK/orcnn_hrsc/seed_$s" "ORCNN_S$s" "$s"
  done
  for s in ${RTMDET_SEEDS//,/ }; do
    run_train rtmdet "$RTMDET_CFG" "$RTMDET_EPOCHS" "$WORK/rtmdet_hrsc/seed_$s" "RTMDET_S$s" "$s"
  done
else
  echo "REFUSE full train: smoke failed"; mark ORCNN_TRAIN SKIPPED_SMOKE_FAIL; mark RTMDET_TRAIN SKIPPED_SMOKE_FAIL
fi

# ---- stage 5: infer on HRSC test split (per-seed; primary seed keeps the canonical filename) ----
echo "== stage 5: infer on test (${nte} imgs) =="
infer(){ # NAME CFG CKPT OUT MARKER
  local name="$1" cfg="$2" ck="$3" out="$4" mp="$5"
  if [ ! -s "$ck" ] || ! ckpt_gate "$ck"; then echo "$name: no integral ckpt -> infer skip"; mark "$mp" SKIPPED_DISCLOSED; return; fi
  python "$ORCH/score_rtmdet.py" --config "$cfg" --checkpoint "$ck" \
    --mmrotate-commit "$MMROTATE_COMMIT" --images-dir "$DOTA/test/images" \
    --class-names ship --score-thr 0.05 --device cuda:0 -o "$out" \
    || echo "warn: score $name rc!=0 (gate authoritative)"
  if jsonl_cover_gate "$out" 400 | tee -a "$RES/${name}_coverage.txt"; then mark "$mp" OK; else mark "$mp" FAILED; fi
}
ORCNN_S0="${ORCNN_SEEDS%%,*}"; RTMDET_S0="${RTMDET_SEEDS%%,*}"   # first seed = primary (canonical filename)
for s in ${ORCNN_SEEDS//,/ }; do
  out="$RES/hrsc_test_dets_orcnn.jsonl"; [ "$s" = "$ORCNN_S0" ] || out="$RES/hrsc_test_dets_orcnn_seed${s}.jsonl"
  infer "orcnn_seed$s"  "$ORCNN_CFG"  "$WORK/orcnn_hrsc/seed_$s/epoch_${ORCNN_EPOCHS}.pth"   "$out" "ORCNN_INFER_S$s"
done
for s in ${RTMDET_SEEDS//,/ }; do
  out="$RES/hrsc_test_dets_rtmdet.jsonl"; [ "$s" = "$RTMDET_S0" ] || out="$RES/hrsc_test_dets_rtmdet_seed${s}.jsonl"
  infer "rtmdet_seed$s" "$RTMDET_CFG" "$WORK/rtmdet_hrsc/seed_$s/epoch_${RTMDET_EPOCHS}.pth" "$out" "RTMDET_INFER_S$s"
done

# ---- stage 6: GT canonicalization (single-class ship) ----
echo "== stage 6: GT prep =="
python "$ORCH/prepare_dota_gt.py" --annfiles-dir "$DOTA/test/annfiles" -o "$RES/hrsc_test_gt.jsonl" \
  2>&1 | tee "$RES/hrsc_gt_prep.log" || echo "warn: gt prep rc!=0"
if [ -s "$RES/hrsc_test_gt.jsonl" ] && [ -s "$RES/hrsc_test_gt.jsonl.provenance.json" ]; then mark GT OK; else mark GT FAILED; fi

# ---- stage 7: tar (train logs + configs + dets + gt + provenance) ----
echo "== stage 7: tar =="
cp "$ORCNN_CFG" "$RTMDET_CFG" "$RES/" 2>/dev/null
cp "$DOTA/train/prepare_hrsc.provenance.json" "$RES/prepare_hrsc_train.provenance.json" 2>/dev/null
cp "$DOTA/test/prepare_hrsc.provenance.json" "$RES/prepare_hrsc_test.provenance.json" 2>/dev/null
# keep only final checkpoints in the tar (disk/pull frugal), one per trained seed
mkdir -p "$RES/checkpoints"
for s in ${ORCNN_SEEDS//,/ }; do
  cp "$WORK/orcnn_hrsc/seed_$s/epoch_${ORCNN_EPOCHS}.pth" "$RES/checkpoints/orcnn_seed${s}_epoch_${ORCNN_EPOCHS}.pth" 2>/dev/null
done
for s in ${RTMDET_SEEDS//,/ }; do
  cp "$WORK/rtmdet_hrsc/seed_$s/epoch_${RTMDET_EPOCHS}.pth" "$RES/checkpoints/rtmdet_seed${s}_epoch_${RTMDET_EPOCHS}.pth" 2>/dev/null
done
DEST=/root/autodl-tmp/hrsc_rotcert.tar.gz
tar --warning=no-file-changed -czf "$DEST" -C /root/autodl-tmp hrsc_rotcert_results 2>>"$LOG"
if [ -s "$DEST" ]; then
  echo "tarred -> $DEST ($(stat -c%s "$DEST") bytes)"
  if [ "${#FAILED[@]}" -eq 0 ]; then echo "HRSC_RUN_ALL_DONE"; else echo "HRSC_RUN_PARTIAL_DONE failed=[${FAILED[*]}]"; fi
else
  echo "HRSC_RUN_TAR_FAILED"
fi
echo "=== hrsc_run end $(date -Iseconds) ==="

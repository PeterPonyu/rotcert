#!/bin/bash
# orcnn_dota_infer.sh -- box-side one-shot (box-reuse round, 2026-07-11):
# score the PUBLISHED mmrotate Oriented R-CNN DOTA checkpoint over the SAME
# DOTA val images the pilot used, producing the fourth cell of the TGRS
# 2x2 detector-x-dataset grid (RTMDet-DOTA pilot / ORCNN-DIOR / RTMDet-DIOR
# already exist). Inference-only; no training, no certification (local).
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh && conda activate "${CHAIN_CONDA_ENV:-mm}"

export CHAIN_LOG="${CHAIN_LOG:-/root/orcnn_dota_infer.log}"
source /root/reliability-commons/tools/boxkit/chain_lib.sh

RESULTS_DIR="${RESULTS_DIR:-/root/autodl-tmp/orcnn_dota_results}"
MARKERS_DIR="$RESULTS_DIR/markers"
mkdir -p "$RESULTS_DIR" "$MARKERS_DIR"

CKPT_URL="${CKPT_URL:-https://download.openmmlab.com/mmrotate/v0.1.0/oriented_rcnn/oriented_rcnn_r50_fpn_fp16_1x_dota_le90/oriented_rcnn_r50_fpn_fp16_1x_dota_le90-57c88621.pth}"
CKPT="${CKPT:-/root/autodl-tmp/orcnn_dota_published.pth}"
CONFIG="${CONFIG:-/root/mmrotate/configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py}"
IMAGES="${IMAGES:-/root/autodl-tmp/dota/val/images}"   # SAME dir as the pilot (grid comparability)
CLASSES="${CLASSES:-plane,ship,storage-tank,baseball-diamond,tennis-court,basketball-court,ground-track-field,harbor,bridge,large-vehicle,small-vehicle,helicopter,roundabout,soccer-ball-field,swimming-pool}"
MMROTATE_COMMIT="${MMROTATE_COMMIT:-3ff004e}"
DETS="$RESULTS_DIR/dota_val_dets_orcnn.jsonl"
MIN_IMAGES="${MIN_IMAGES:-400}"   # DOTA val = 458 full images

fail=0
mark() { printf '%s\n' "$2" > "$MARKERS_DIR/$1.marker"; echo "MARKER $1=$2"; [ "$2" = "FAILED" ] && fail=1; return 0; }

# 1. Published checkpoint (wget -c resumable; content gate on size, >=100MB)
if [ ! -s "$CKPT" ] || [ "$(stat -c%s "$CKPT")" -lt 100000000 ]; then
  wget -c -q -O "$CKPT" "$CKPT_URL" || true
fi
if [ -s "$CKPT" ] && [ "$(stat -c%s "$CKPT")" -ge 100000000 ]; then
  mark ORCNN_DOTA_CKPT OK
else
  mark ORCNN_DOTA_CKPT FAILED
fi

# 2. Inference over the pilot's val images (skip-if-exists, content-gated)
if [ "$fail" -eq 0 ]; then
  if [ -s "$DETS" ]; then
    echo "skip (exists): $DETS"
  else
    python3 /root/reliability-commons/tools/rotcert/orchestration/score_rtmdet.py \
      --config "$CONFIG" --checkpoint "$CKPT" --mmrotate-commit "$MMROTATE_COMMIT" \
      --images-dir "$IMAGES" --class-names "$CLASSES" --device cuda:0 -o "$DETS" \
      || echo "warning: score exited non-zero -- gate below authoritative"
  fi
  n_img=$(python3 - "$DETS" <<'PY'
import json, sys
ids=set()
try:
    for line in open(sys.argv[1]):
        line=line.strip()
        if line: ids.add(json.loads(line).get("image_id"))
except FileNotFoundError:
    pass
print(len(ids))
PY
)
  echo "distinct images scored: $n_img (floor $MIN_IMAGES)"
  if [ "$n_img" -ge "$MIN_IMAGES" ]; then mark ORCNN_DOTA_INFER OK; else mark ORCNN_DOTA_INFER FAILED; fi
else
  mark ORCNN_DOTA_INFER SKIPPED_DISCLOSED
fi

if [ "$fail" -eq 0 ]; then
  marker_name="ORCNN_DOTA_INFER_ALL_DONE"
  # No pre-epilogue echo of the marker (2026-07-10 watcher-race rule).
else
  marker_name="ORCNN_DOTA_INFER_PARTIAL_DONE"
  echo "ORCNN_DOTA_INFER_PARTIAL -- see markers"
fi
chain_epilogue "$RESULTS_DIR" "$marker_name" "orcnn_dota_infer"

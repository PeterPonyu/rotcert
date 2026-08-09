#!/bin/bash
# dior_infer_trigger.sh -- box-side: launches the DIOR-R inference chain once
# the TRAINING chain has fully finished. Completion is detected by the removal
# of chain_prologue's /root/chain_running.pid (chain_epilogue deletes it after
# tar + marker + ack-wait) -- NOT by the integrity-marker FILES, which already
# exist on this box with stale content from pre-NaN-fix relaunches and would
# fire a file-existence wait immediately (2026-07-11 review note).
set -u

PIDFILE="${TRAIN_CHAIN_PIDFILE:-/root/chain_running.pid}"
INFER_LOG="${INFER_LOG:-/root/rotcert_dior_infer.log}"
POLL_S="${POLL_S:-300}"

echo "$(date -Iseconds) dior_infer_trigger: waiting for $PIDFILE removal (training chain epilogue)" >> "$INFER_LOG"
while [ -f "$PIDFILE" ]; do
  sleep "$POLL_S"
done
echo "$(date -Iseconds) dior_infer_trigger: training chain finished -- launching inference chain" >> "$INFER_LOG"

export CHAIN_CONDA_ENV="${CHAIN_CONDA_ENV:-mm}"
export CHAIN_LOG="$INFER_LOG"
export MMROTATE_COMMIT="${MMROTATE_COMMIT:-3ff004e}"
export ORCNN_DIOR_CONFIG="${ORCNN_DIOR_CONFIG:-/root/mmrotate/configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dior.py}"
export RTMDET_R_DIOR_CONFIG="${RTMDET_R_DIOR_CONFIG:-/root/mmrotate/configs/rotated_rtmdet/rotated_rtmdet_l-3x-dior.py}"

bash /root/reliability-commons/tools/rotcert/orchestration/next_boot_rotcert_dior_infer.sh >> "$INFER_LOG" 2>&1

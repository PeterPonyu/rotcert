#!/bin/bash
# rtmdet_resume_runner.sh -- box-side one-shot: resume RTMDet-R-l DIOR-R
# training from epoch_12 after the pad_size_divisor=32 config fix (2026-07-11
# epoch-12 val crash: CSPNeXt-PAFPN "Expected size 50 but got size 49" -- DIOR
# test images are not uniformly 800x800 and nothing padded them to /32).
# On integral epoch_36: refresh the training chain's RTMDet markers to OK,
# re-tar the training results (the box tar is stale at epoch_12), then run the
# inference chain against a FRESH log, /root/rotcert_dior_infer2.log -- the
# first run's log already contains ROTCERT_DIOR_INFER_PARTIAL_DONE, which
# would instantly trip a watcher polling that file.
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh && conda activate mm

WORK="${WORK:-/root/autodl-tmp/dior_r/work_dirs/rtmdet_r_dior/seed_0}"
CFG="${CFG:-/root/mmrotate/configs/rotated_rtmdet/rotated_rtmdet_l-3x-dior.py}"
RES="${RES:-/root/autodl-tmp/rotcert_dior_train_results}"
CKPT="$WORK/epoch_36.pth"

if [ ! -s "$CKPT" ]; then
  # The dataset base config uses the RELATIVE data_root 'data/DIOR/', resolved
  # against /root/mmrotate (where the training chain runs and the data symlink
  # lives) -- running from anywhere else dies on ImageSets/Main/train.txt.
  cd /root/mmrotate || exit 1
  # Same cfg-options as the training chain's train_stage (batch MUST match for
  # the resumed schedule to line up); --resume auto-loads $WORK/last_checkpoint.
  python3 /root/mmrotate/tools/train.py "$CFG" --work-dir "$WORK" --resume \
    --cfg-options randomness.seed=0 train_cfg.max_epochs=36 \
    train_dataloader.batch_size=8 default_hooks.checkpoint.max_keep_ckpts=3 \
    2>&1 | tee "$RES/rtmdet_r_dior_seed0_resume.log" \
    || echo "warning: resume train exited non-zero -- integrity gate below is authoritative"
fi

if [ -s "$CKPT" ] && python3 -c "import torch; torch.load('$CKPT', map_location='cpu', weights_only=False)" 2>/dev/null; then
  printf 'OK\n' > "$RES/markers/RTMDET_R_DIOR_TRAIN.marker"
  printf 'OK\n' > "$RES/markers/RTMDET_R_DIOR_CKPT_INTEGRITY.marker"
  echo "$(date -Iseconds) rtmdet resume: epoch_36 integral -- markers refreshed OK"
  tar --warning=no-file-changed -czf /root/autodl-tmp/rotcert_dior_train.tar.gz \
    "$RES" /root/autodl-tmp/dior_r/work_dirs 2>/dev/null
  echo "$(date -Iseconds) rtmdet resume: training tar refreshed"

  export CHAIN_CONDA_ENV=mm
  export CHAIN_LOG=/root/rotcert_dior_infer2.log
  export MMROTATE_COMMIT="${MMROTATE_COMMIT:-3ff004e}"
  export ORCNN_DIOR_CONFIG="${ORCNN_DIOR_CONFIG:-/root/mmrotate/configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dior.py}"
  export RTMDET_R_DIOR_CONFIG="$CFG"
  bash /root/reliability-commons/tools/rotcert/orchestration/next_boot_rotcert_dior_infer.sh >> /root/rotcert_dior_infer2.log 2>&1
else
  echo "$(date -Iseconds) RTMDET RESUME FAILED -- no integral epoch_36; markers left FAILED" >&2
fi

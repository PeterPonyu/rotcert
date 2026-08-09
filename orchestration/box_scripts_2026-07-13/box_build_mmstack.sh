#!/bin/bash
# box_build_mmstack.sh -- build the mmrotate mm-stack in a fresh py3.10 conda env
# (base is py3.12 -> no prebuilt mmcv 2.x wheels; no nvcc on box -> prebuilt wheels only).
# Env name: mmrot. Pins chosen for prebuilt-wheel availability on cu121/torch2.1.0/cp310,
# satisfying mmrotate dev-1.x (@3ff004e) + mmdet 3.3.0 + mmcv 2.1.0 + mmengine 0.10.x.
set -uo pipefail
LOG=/root/mmstack_build.log
exec > >(tee -a "$LOG") 2>&1
echo "=== mmstack build start $(date -Iseconds) ==="
source /root/miniconda3/etc/profile.d/conda.sh

ALIYUN="-i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com"
MMR=/root/mmrotate
COMMIT=3ff004eb21ea040455b5585db229edba4037f1bf

# 1. env
if ! conda env list | grep -q '/envs/mmrot$'; then
  echo "-- creating conda env mmrot (python 3.10)"
  conda create -n mmrot python=3.10 -y || { echo "MMSTACK_BUILD_FAIL create-env"; exit 1; }
fi
conda activate mmrot
python --version

# 2. torch (default pypi linux wheel for 2.1.0 is the cu121 build)
if ! python -c "import torch" 2>/dev/null; then
  echo "-- installing torch 2.1.0 + torchvision 0.16.0"
  pip install $ALIYUN torch==2.1.0 torchvision==0.16.0 || { echo "MMSTACK_BUILD_FAIL torch"; exit 1; }
fi
python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available(),'dev',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')" || { echo "MMSTACK_BUILD_FAIL torch-probe"; exit 1; }

# 3. mm stack (openmim + mmengine + mmcv prebuilt wheel + mmdet)
# LESSON (b) 2026-07-13: pin numpy<2 BEFORE mmcv/mmdet/pycocotools are installed --
# numpy 2.x breaks the ABI of the C-extensions those wheels were compiled against
# (mmcv, pycocotools), so a numpy-2 env imports-but-crashes at first op. Enforced here
# so the turnkey build never regresses to the numpy-2 default. Idempotent.
python -c "import numpy,sys; sys.exit(0 if numpy.__version__[0]=='1' else 1)" 2>/dev/null \
  || pip install $ALIYUN "numpy<2" || { echo "MMSTACK_BUILD_FAIL numpy-pin"; exit 1; }
python -c "import mim" 2>/dev/null || pip install $ALIYUN -U openmim
python -c "import mmengine" 2>/dev/null || pip install $ALIYUN "mmengine==0.10.5"
if ! python -c "import mmcv" 2>/dev/null; then
  echo "-- installing mmcv 2.1.0 (prebuilt cu121/torch2.1.0 wheel)"
  pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html || { echo "MMSTACK_BUILD_FAIL mmcv"; exit 1; }
fi
python -c "import mmdet" 2>/dev/null || pip install $ALIYUN "mmdet==3.3.0" || { echo "MMSTACK_BUILD_FAIL mmdet"; exit 1; }

# 4. mmrotate @ pinned commit (turbo proxy first, then ghproxy, then direct)
if [ ! -d "$MMR/.git" ]; then
  echo "-- cloning mmrotate"
  ( source /etc/network_turbo 2>/dev/null; git clone https://github.com/open-mmlab/mmrotate.git "$MMR" ) \
    || git clone https://ghproxy.com/https://github.com/open-mmlab/mmrotate.git "$MMR" \
    || git clone https://github.com/open-mmlab/mmrotate.git "$MMR" \
    || { echo "MMSTACK_BUILD_FAIL clone"; exit 1; }
fi
cd "$MMR"
( source /etc/network_turbo 2>/dev/null; git fetch --all 2>/dev/null )
git checkout "$COMMIT" 2>/dev/null || git checkout dev-1.x 2>/dev/null || echo "warn: checkout fell through"
echo "-- mmrotate HEAD: $(git rev-parse HEAD)"
pip install $ALIYUN -e . || { echo "MMSTACK_BUILD_FAIL mmrotate-e"; exit 1; }

# LESSON (a) 2026-07-13: mmrotate's OWN version gate (mmrotate/__init__.py) asserts
# mmdet <= mmdet_maximum_version (a '3.2.0'-ish bound), which the installed mmdet 3.3.0
# fails -> `import mmrotate` raises before any op. There is NO prebuilt mmcv wheel <2.1.0
# for cu121/torch2.1.0/cp310 and the box has no nvcc to build one, so DOWNGRADING mmdet
# is not viable; patch the gate to admit 3.3.0 instead (mmrotate's mmcv bound is already
# satisfied). Idempotent + verified below; runs BEFORE the probe (which imports mmrotate).
MMR_INIT="$MMR/mmrotate/__init__.py"
if [ -f "$MMR_INIT" ]; then
  sed -i -E "s/(mmdet_maximum_version\s*=\s*')[0-9.]+(')/\13.3.0\2/" "$MMR_INIT"
  gate=$(python - "$MMR_INIT" <<'PY'
import re,sys
t=open(sys.argv[1]).read()
m=re.search(r"mmdet_maximum_version\s*=\s*'([0-9.]+)'",t)
print(m.group(1) if m else "NONE")
PY
)
  echo "-- mmrotate mmdet_maximum_version patched to: $gate"
  [ "$gate" = "3.3.0" ] || echo "WARN: mmdet_maximum_version gate is '$gate' (expected 3.3.0); import may still assert"
else
  echo "WARN: $MMR_INIT absent -- cannot patch mmdet version gate"
fi

# 5. probe: imports + versions + CUDA rotated-nms op
python - <<'PY'
import torch, mmcv, mmdet, mmrotate
print("IMPORTS_OK torch", torch.__version__, "mmcv", mmcv.__version__, "mmdet", mmdet.__version__, "mmrotate", mmrotate.__version__)
from mmcv.ops import nms_rotated
b = torch.tensor([[50.,50.,20.,10.,0.1],[51.,51.,20.,10.,0.2]], dtype=torch.float32).cuda()
s = torch.tensor([0.9,0.8], dtype=torch.float32).cuda()
d,k = nms_rotated(b, s, 0.1)
print("NMS_ROTATED_CUDA_OK", tuple(d.shape), "kept", int(k.numel()))
PY
rc=$?
if [ "$rc" -eq 0 ]; then
  touch /root/MMSTACK_OK
  echo "MMSTACK_OK written"
  echo "MMSTACK_BUILD_DONE $(date -Iseconds)"
else
  echo "MMSTACK_BUILD_FAIL probe rc=$rc"
  exit 1
fi

# --- FOLDED-IN FIXES 2026-07-13 (originally a manual no-GPU-mode recovery; now applied
#     inline above so the Config-B turnkey build never needs the manual step) ---
# Original pins (mmdet==3.3.0 + a hardcoded mmrotate mmdet_maximum_version='3.2.0')
# failed mmrotate's own version-gate assertion. There is NO prebuilt mmcv wheel
# <2.1.0 for cu121/torch2.1.0/cp310 (only 2.1.0/2.2.0 exist), and the box has no
# nvcc to build mmcv from source -- so downgrading mmcv is not viable. Fixes now IN
# the script: (a) after `pip install -e .`, patch mmrotate/__init__.py
# mmdet_maximum_version -> '3.3.0' (step 4 tail; runs before the import probe);
# (b) pin numpy<2 before the mm-stack installs (step 3 head; numpy 2.x breaks the
# ABI of mmcv/pycocotools C-extensions compiled against numpy<2).
# The manual fix was verified with a functional smoke test (not just import): built an
# actual RTMDet model from configs/rotated_rtmdet/rotated_rtmdet_l-100e-aug-dota.py via
# mmengine.Config + mmdet.registry.MODELS.build().

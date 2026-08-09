#!/bin/bash
# next_boot_rotcert_configA.sh -- Config-A one-command box runner (2026-07-13).
#
# Produces the substrate for the Config-A grid jump (COMPUTE-PLAN-2026-07-13.md):
#   Stage 1  HRSC2016-MS 3rd-dataset arm  -- hrsc_run.sh: convert (angle-convention
#            refuse-gate) -> smoke (1-epoch finite-loss) -> full train (2 detectors,
#            ORCNN_SEEDS/RTMDET_SEEDS) -> infer test -> GT prep -> tar. Sentinel:
#            HRSC_RUN_ALL_DONE.
#   Stage 2  DOTA detector-zoo inference  -- configs_dota_zoo/dota_zoo_infer.sh: four
#            released Apache-2.0 zoo checkpoints (RoI Transformer, S2A-Net, Oriented
#            RepPoints, Gliding Vertex) over the existing DOTA val crops. Sentinel:
#            DOTA_ZOO_INFER_ALL_DONE.
#   Stage 3  tar + pull (chain_epilogue). Marker: ROTCERT_CONFIGA_ALL_DONE.
#
# Certification of the new cells (cert_cell.sh) and the regime-conditional efficiency
# analysis are the LOCAL, CPU, post-pull Tier-1 follow-up -- this chain only makes the
# GPU substrate. Content-gated throughout (never bare exit codes); each sub-runner has
# its own smoke gate and sentinel, which this chain checks.
#
# --dry-run validates every sub-script, config, manifest and intended path WITHOUT any
# GPU, download, unzip, box, or chain_lib dependency, prints the plan, and exits 0 iff
# all preflight checks pass. Intended to run locally before the box is ever booted.
#
# Usage on the box:  bash next_boot_rotcert_configA.sh
# Usage locally:     bash next_boot_rotcert_configA.sh --dry-run
set -uo pipefail   # NOT -e: later stages/epilogue must run after an earlier gate fails.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # .../orchestration
ROTCERT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RELIABILITY_COMMONS="${RELIABILITY_COMMONS:-/root/reliability-commons}"

DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

# ---- tunables (env-overridable, named defaults) ----
MMROTATE_COMMIT="${MMROTATE_COMMIT:-3ff004eb21ea040455b5585db229edba4037f1bf}"
HRSC_MS_ZIP="${HRSC_MS_ZIP:-/root/autodl-tmp/HRSC2016-MS.zip}"
HRSC_ROOT="${HRSC_ROOT:-/root/autodl-tmp/HRSC2016-MS}"
HRSC_MS_MD5="${HRSC_MS_MD5:-167501c0de0d6015a109a4abddd77fb1}"       # verified local prefetch 2026-07-13
export HRSC_ROOT MMROTATE_COMMIT
export ORCNN_SEEDS="${ORCNN_SEEDS:-0}"                               # Config B: e.g. 0,1,2
export RTMDET_SEEDS="${RTMDET_SEEDS:-0}"
RESULTS_HRSC="${RESULTS_HRSC:-/root/autodl-tmp/hrsc_rotcert_results}"
RESULTS_ZOO="${RESULTS_ZOO:-/root/autodl-tmp/dota_zoo_results}"
DOTA_VAL_IMAGES="${DOTA_VAL_IMAGES:-/root/autodl-tmp/dota_split/val/images}"  # box_dota_crops.sh output: literal-gated 5297 ss_val crops, NOT the 458 raw val images
MMR="${MMR:-/root/mmrotate}"

HRSC_RUN="$SCRIPT_DIR/hrsc_run.sh"
ZOO_RUN="$SCRIPT_DIR/configs_dota_zoo/dota_zoo_infer.sh"
PREP_HRSC="$SCRIPT_DIR/prepare_hrsc.py"
SCORE="$SCRIPT_DIR/score_rtmdet.py"
CERT_CELL="$SCRIPT_DIR/cert_cell.sh"
R20="$SCRIPT_DIR/r20_generic.py"
MANIFEST="$SCRIPT_DIR/configs_dota_zoo/dota_zoo_manifest.json"
HRSC_CFGS=("$SCRIPT_DIR/configs_hrsc/oriented-rcnn-le90_r50_fpn_1x_hrsc.py"
           "$SCRIPT_DIR/configs_hrsc/rotated-rtmdet_l-3x-hrsc.py")

MARKERS_DIR="${MARKERS_DIR:-$RESULTS_HRSC/../configA_markers}"
[ "$DRY" -eq 1 ] || mkdir -p "$MARKERS_DIR"
FAILED=()
mark(){ [ "$DRY" -eq 1 ] && { echo "DRY MARKER $1=$2"; return 0; }
        printf '%s\n' "$2" > "$MARKERS_DIR/$1.marker"; echo "MARKER $1=$2"; [ "$2" = OK ] || FAILED+=("$1"); return 0; }
sentinel(){ grep -q "$2" "$1" 2>/dev/null; }   # LOGFILE SENTINEL -> rc 0 iff present

# ---- preflight: local, no side effects (runs in both modes) ----
preflight(){
  local ok=1
  echo "== preflight: sub-scripts, configs, manifest =="
  for f in "$HRSC_RUN" "$ZOO_RUN" "$PREP_HRSC" "$SCORE" "$CERT_CELL" "$R20" "$MANIFEST" "${HRSC_CFGS[@]}"; do
    if [ -f "$f" ]; then echo "OK  $f"; else echo "MISSING $f"; ok=0; fi
  done
  echo "== preflight: bash -n on shell runners =="
  for s in "$HRSC_RUN" "$ZOO_RUN" "$CERT_CELL"; do
    if bash -n "$s" 2>/dev/null; then echo "OK  syntax $s"; else echo "SYNTAX-FAIL $s"; ok=0; fi
  done
  echo "== preflight: manifest parses + 4 detectors =="
  local n
  n=$(python3 -c "import json;print(len(json.load(open('$MANIFEST'))['detectors']))" 2>/dev/null || echo 0)
  if [ "$n" -eq 4 ]; then echo "OK  manifest lists 4 detectors"; else echo "BAD manifest detector count=$n"; ok=0; fi
  echo "== preflight: DOTA zoo infer own dry-run =="
  if ZOO_PY=python3 MMR="$MMR" IMAGES="$DOTA_VAL_IMAGES" RESULTS_DIR="$RESULTS_ZOO" bash "$ZOO_RUN" --dry-run >/tmp/_zoo_dry.$$ 2>&1 && grep -q DOTA_ZOO_DRYRUN_OK /tmp/_zoo_dry.$$; then
    echo "OK  dota_zoo_infer --dry-run -> DOTA_ZOO_DRYRUN_OK"
  else echo "FAIL dota_zoo_infer --dry-run"; sed 's/^/    /' /tmp/_zoo_dry.$$; ok=0; fi
  rm -f /tmp/_zoo_dry.$$
  echo "== preflight: box-only paths (reported, not failed in dry mode) =="
  for p in "$HRSC_MS_ZIP" "$HRSC_ROOT" "$DOTA_VAL_IMAGES" "$MMR/configs" "$RELIABILITY_COMMONS/tools/boxkit/chain_lib.sh"; do
    if [ -e "$p" ]; then echo "present: $p"; else echo "absent (box-only, expected locally): $p"; fi
  done
  return $([ "$ok" -eq 1 ] && echo 0 || echo 1)
}

if [ "$DRY" -eq 1 ]; then
  echo "=== next_boot_rotcert_configA (DRY-RUN) $(date -Iseconds) ==="
  if preflight; then
    echo "PLAN stage1: [unzip $HRSC_MS_ZIP -> $HRSC_ROOT if absent; md5 gate $HRSC_MS_MD5] then bash $HRSC_RUN (ORCNN_SEEDS=$ORCNN_SEEDS RTMDET_SEEDS=$RTMDET_SEEDS) -> sentinel HRSC_RUN_ALL_DONE"
    echo "PLAN stage2: bash $ZOO_RUN -> sentinel DOTA_ZOO_INFER_ALL_DONE"
    echo "PLAN stage3: chain_epilogue tars [$RESULTS_HRSC $RESULTS_ZOO] -> marker ROTCERT_CONFIGA_ALL_DONE"
    echo "CONFIGA_DRYRUN_OK"; exit 0
  else
    echo "CONFIGA_DRYRUN_FAIL"; exit 1
  fi
fi

# ================= REAL RUN (box) =================
# shellcheck disable=SC1091
source "$RELIABILITY_COMMONS/tools/boxkit/chain_lib.sh"
chain_prologue
preflight || echo "preflight reported issues -- gates below are authoritative"

# ---- stage 1: HRSC unzip (content-gated) + hrsc_run.sh ----
echo "== stage 1: HRSC2016-MS unzip + train/infer =="
if [ ! -f "$HRSC_ROOT/ImageSets/source/train.txt" ]; then
  if [ -s "$HRSC_MS_ZIP" ]; then
    if command -v md5sum >/dev/null && [ -n "$HRSC_MS_MD5" ]; then
      got=$(md5sum "$HRSC_MS_ZIP" | cut -d' ' -f1)
      [ "$got" = "$HRSC_MS_MD5" ] && echo "HRSC zip md5 OK ($got)" || echo "WARN HRSC zip md5 $got != $HRSC_MS_MD5"
    fi
    mkdir -p "$HRSC_ROOT"; unzip -q -o "$HRSC_MS_ZIP" -d "$HRSC_ROOT" || echo "warn: unzip rc!=0 (gate below authoritative)"
    # HRSC2016-MS.zip extracts AllImages/ Annotations/ ImageSets/ at the zip root
  else
    echo "HRSC zip absent at $HRSC_MS_ZIP and root not populated"
  fi
fi
nann=$(ls "$HRSC_ROOT/Annotations"/*.xml 2>/dev/null | wc -l)
if [ -f "$HRSC_ROOT/ImageSets/source/train.txt" ] && [ "$nann" -ge 1000 ]; then mark HRSC_UNZIP OK; else mark HRSC_UNZIP FAILED; fi

HRSC_LOG=/root/hrsc_run.log
if [ "$(cat "$MARKERS_DIR/HRSC_UNZIP.marker" 2>/dev/null)" = OK ]; then
  bash "$HRSC_RUN" || echo "warn: hrsc_run rc!=0 (sentinel/gate authoritative)"
  if sentinel "$HRSC_LOG" HRSC_RUN_ALL_DONE; then mark HRSC_RUN OK
  elif sentinel "$HRSC_LOG" HRSC_RUN_PARTIAL_DONE; then mark HRSC_RUN PARTIAL
  else mark HRSC_RUN FAILED; fi
else
  echo "REFUSE hrsc_run: unzip gate not OK"; mark HRSC_RUN SKIPPED_UNZIP
fi

# ---- stage 2: DOTA zoo inference ----
echo "== stage 2: DOTA detector-zoo inference (4 detectors) =="
ZOO_LOG="$RESULTS_ZOO/dota_zoo_infer.log"; mkdir -p "$RESULTS_ZOO"
RESULTS_DIR="$RESULTS_ZOO" MMR="$MMR" IMAGES="$DOTA_VAL_IMAGES" MMROTATE_COMMIT="$MMROTATE_COMMIT" \
  bash "$ZOO_RUN" 2>&1 | tee "$ZOO_LOG"
if sentinel "$ZOO_LOG" DOTA_ZOO_INFER_ALL_DONE; then mark DOTA_ZOO OK
elif sentinel "$ZOO_LOG" DOTA_ZOO_INFER_PARTIAL; then mark DOTA_ZOO PARTIAL
else mark DOTA_ZOO FAILED; fi

# ---- stage 3: tar + pull ----
if [ "${#FAILED[@]}" -eq 0 ]; then marker_name="ROTCERT_CONFIGA_ALL_DONE"
else marker_name="ROTCERT_CONFIGA_PARTIAL_DONE"; echo "CONFIGA PARTIAL -- failed: ${FAILED[*]}"; fi
chain_epilogue "$RESULTS_HRSC $RESULTS_ZOO $MARKERS_DIR" "$marker_name" "rotcert_configA"

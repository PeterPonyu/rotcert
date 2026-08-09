#!/bin/bash
# dota_zoo_infer.sh -- Config-A DOTA-inference arm (2026-07-13). Scores four released
# Apache-2.0 mmrotate DOTA-v1.0 zoo checkpoints (RoI Transformer, S2A-Net, Oriented
# RepPoints, Gliding Vertex) over the SAME DOTA val crops the two existing DOTA cells
# used, via the detector-agnostic orchestration/score_rtmdet.py. Inference only; the
# per-detector detections JSONL is then certified locally by cert_cell.sh.
#
# Single source of truth: configs_dota_zoo/dota_zoo_manifest.json (config filename,
# checkpoint URL, expected mAP, license, angle convention). The vendored leaf config is
# copied into $MMR/configs/<detector>/ before inference so its ../_base_ chain resolves
# against the installed mmrotate tree at the pinned commit.
#
# --dry-run validates the manifest, the vendored configs, the mmrotate config dirs, the
# DOTA val images dir and the score script WITHOUT any GPU, download or inference, then
# prints each planned score_rtmdet.py command. Exit 0 iff every check passes.
set -uo pipefail

# Box-side only: activate the conda env with the mm-stack (mmcv/mmdet/mmrotate).
# --dry-run must still work with none of that installed (see module docstring),
# so only activate for a real run, and don't fail if conda itself is absent
# (e.g. local dry-run invocations off-box).
if [ "${1:-}" != "--dry-run" ] && [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh; conda activate "${MM_CONDA_ENV:-base}"
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"           # .../orchestration/configs_dota_zoo
ORCH="$(cd "$HERE/.." && pwd)"                                  # .../orchestration
ROTCERT_ROOT="$(cd "$ORCH/.." && pwd)"                          # .../tools/rotcert (rotcert package root)
export PYTHONPATH="$ROTCERT_ROOT:${PYTHONPATH:-}"               # score_rtmdet.py needs `import rotcert.gwd`
MANIFEST="$HERE/dota_zoo_manifest.json"

DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
PY="${ZOO_PY:-python3}"
MMR="${MMR:-/root/mmrotate}"
RESULTS_DIR="${RESULTS_DIR:-/root/autodl-tmp/dota_zoo_results}"
IMAGES="${IMAGES:-/root/autodl-tmp/dota/val/images}"           # SAME dir as existing DOTA cells (grid comparability)
CKPT_CACHE="${CKPT_CACHE:-/root/autodl-tmp/dota_zoo_ckpts}"
MMROTATE_COMMIT="${MMROTATE_COMMIT:-3ff004eb21ea040455b5585db229edba4037f1bf}"
CLASSES="${CLASSES:-plane,ship,storage-tank,baseball-diamond,tennis-court,basketball-court,ground-track-field,harbor,bridge,large-vehicle,small-vehicle,helicopter,roundabout,soccer-ball-field,swimming-pool}"
MIN_IMAGES="${MIN_IMAGES:-400}"
SCORE="$ORCH/score_rtmdet.py"

MK="$RESULTS_DIR/markers"; [ "$DRY" -eq 1 ] || mkdir -p "$RESULTS_DIR" "$MK" "$CKPT_CACHE"
FAILED=()
mark(){ [ "$DRY" -eq 1 ] && { echo "DRY MARKER $1=$2"; return 0; }
        printf '%s\n' "$2" > "$MK/$1.marker"; echo "MARKER $1=$2"; [ "$2" = OK ] || FAILED+=("$1"); return 0; }

# Emit one TSV row per detector from the manifest: slug \t config \t mmr_dir \t url \t map
rows(){
  "$PY" - "$MANIFEST" <<'PY'
import json,sys,os
m=json.load(open(sys.argv[1]))
slug={"RoI Transformer":"roi_trans","S2A-Net":"s2anet","Oriented RepPoints":"oriented_reppoints","Gliding Vertex":"gliding_vertex"}
for d in m["detectors"]:
    s=slug[d["detector"]]
    mdir=d["mmrotate_config_dir_on_box"].strip("/")
    print("\t".join([s, d["config"], mdir, d["checkpoint_url"], str(d["expected_map_dotav1"])]))
PY
}

echo "=== dota_zoo_infer $( [ "$DRY" -eq 1 ] && echo '(DRY-RUN)' ) $(date -Iseconds) ==="

# ---- preflight (runs in both modes) ----
pf_ok=1
[ -f "$MANIFEST" ] && "$PY" -c "import json;json.load(open('$MANIFEST'))" 2>/dev/null && echo "OK manifest parses" || { echo "MISSING/BAD manifest: $MANIFEST"; pf_ok=0; }
[ -f "$SCORE" ] && echo "OK score script: $SCORE" || { echo "MISSING score script: $SCORE"; pf_ok=0; }
if [ "$DRY" -eq 1 ]; then
  # In dry mode on the local side, $MMR/$IMAGES may not exist (they are box paths). Report, do not fail on them.
  [ -d "$MMR/configs" ] && echo "OK mmrotate configs tree: $MMR/configs" || echo "NOTE (box-only) mmrotate tree absent locally: $MMR/configs"
  [ -d "$IMAGES" ] && echo "OK DOTA val images: $IMAGES" || echo "NOTE (box-only) DOTA val images absent locally: $IMAGES"
else
  [ -d "$MMR/configs" ] || { echo "MISSING mmrotate tree: $MMR/configs"; pf_ok=0; }
  [ -d "$IMAGES" ] || { echo "MISSING DOTA val images: $IMAGES"; pf_ok=0; }
fi

# ---- per-detector loop ----
while IFS=$'\t' read -r slug cfg mdir url emap; do
  [ -n "$slug" ] || continue
  vend="$HERE/$cfg"
  dets="$RESULTS_DIR/dota_val_dets_${slug}.jsonl"
  ck="$CKPT_CACHE/${slug}.pth"
  echo "--- $slug (cfg=$cfg mAP=$emap) ---"
  if [ ! -f "$vend" ]; then echo "MISSING vendored config: $vend"; mark "${slug}_CFG" FAILED; pf_ok=0; continue; fi
  echo "OK vendored config: $vend"
  boxcfg="$MMR/$mdir/$cfg"
  if [ "$DRY" -eq 1 ]; then
    echo "PLAN: cp $vend $MMR/$mdir/"
    echo "PLAN: wget -c -O $ck $url"
    echo "PLAN: $PY $SCORE --config $boxcfg --checkpoint $ck --mmrotate-commit $MMROTATE_COMMIT --images-dir $IMAGES --class-names <15> --device cuda:0 -o $dets"
    mark "${slug}_PLANNED" OK
    continue
  fi
  # real run
  mkdir -p "$MMR/$mdir"; cp "$vend" "$MMR/$mdir/"
  if [ ! -s "$ck" ] || [ "$(stat -c%s "$ck")" -lt 100000000 ]; then
    wget -c -q -O "$ck" "$url" || echo "warn: wget $slug rc!=0 (gate below authoritative)"
  fi
  if [ ! -s "$ck" ] || [ "$(stat -c%s "$ck")" -lt 100000000 ]; then mark "${slug}_CKPT" FAILED; continue; fi
  mark "${slug}_CKPT" OK
  if [ -s "$dets" ]; then echo "skip (exists): $dets"; else
    "$PY" "$SCORE" --config "$boxcfg" --checkpoint "$ck" --mmrotate-commit "$MMROTATE_COMMIT" \
      --images-dir "$IMAGES" --class-names "$CLASSES" --score-thr 0.05 --device cuda:0 -o "$dets" \
      || echo "warn: score $slug rc!=0 (gate authoritative)"
  fi
  nimg=$("$PY" - "$dets" <<'PY'
import json,sys
ids=set()
try:
    for ln in open(sys.argv[1]):
        ln=ln.strip()
        if ln: ids.add(json.loads(ln).get("image_id"))
except FileNotFoundError: pass
print(len(ids))
PY
)
  echo "distinct images scored: $nimg (floor $MIN_IMAGES)"
  if [ "$nimg" -ge "$MIN_IMAGES" ]; then mark "${slug}_INFER" OK; else mark "${slug}_INFER" FAILED; fi
done < <(rows)

if [ "$DRY" -eq 1 ]; then
  [ "$pf_ok" -eq 1 ] && { echo "DOTA_ZOO_DRYRUN_OK"; exit 0; } || { echo "DOTA_ZOO_DRYRUN_FAIL"; exit 1; }
fi
echo "=== dota_zoo_infer end $(date -Iseconds) ==="
if [ "${#FAILED[@]}" -eq 0 ] && [ "$pf_ok" -eq 1 ]; then echo "DOTA_ZOO_INFER_ALL_DONE"; else echo "DOTA_ZOO_INFER_PARTIAL failed=[${FAILED[*]}]"; exit 1; fi

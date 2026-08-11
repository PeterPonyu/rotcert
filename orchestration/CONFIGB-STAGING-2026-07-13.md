# rotcert Config-B box-run staging note (2026-07-13)

**Scope.** LOCAL staging only. No GPU was run, no AutoDL/boxkit API touched, no frozen
result dir or manuscript modified. This note records what was built for the Config-B
box run (COMPUTE-PLAN-2026-07-13.md sec.2 "Config B"), what was verified locally, and
what remains box-day-only.

Config B = Config A **plus** two depth arms:
1. **DIOR-R cross-family breadth** — train RoI Transformer and S2A-Net on DIOR-R in-house
   (1x schedule, ~8 GPU-h each), then infer the DIOR-R held-out **test** split.
2. **HRSC training-variance** — 3-seed (0,1,2) HRSC training for the two existing
   detectors (Oriented R-CNN 1x, RTMDet-R 3x), ~3 GPU-h total.

---

## Files created / edited (all under `orchestration/`)

**Created**
- `configs_dior/roi-trans-le90_r50_fpn_1x_dior.py` — RoI Transformer DIOR-R training config.
- `configs_dior/s2anet-le135_r50_fpn_1x_dior.py` — S2A-Net DIOR-R training config.
- `next_boot_rotcert_configB.sh` — turnkey box runner (stages 0–5, content-gated,
  per-stage sentinels, `--dry-run`, `--resume`). `chmod +x`.
- `CONFIGB-STAGING-2026-07-13.md` — this note.

**Edited**
- `box_scripts_2026-07-13/box_build_mmstack.sh` — folded the two hard-won Config-A box
  lessons INTO the build (previously only described in a trailing comment, applied by
  hand during the Config-A run):
  - **(a)** after `pip install -e .`, patch `mmrotate/__init__.py`
    `mmdet_maximum_version -> '3.3.0'` (idempotent sed + verify), so `import mmrotate`
    stops asserting against the installed mmdet 3.3.0. Runs **before** the import probe.
  - **(b)** pin `numpy<2` before the mm-stack installs (numpy 2.x breaks the ABI of the
    mmcv/pycocotools C-extensions). Idempotent.
  - Trailing comment updated to say the fixes are now inline, not manual.
  - NOT changed (working Config-A pins; changing them is unneeded risk): torch
    2.1.0/cu121, mmcv 2.1.0 cu121 prebuilt, mmdet 3.3.0, mmengine 0.10.5, mmrotate
    @3ff004e. (The task's reference pin list names cu118 / mmengine 0.10.7; the
    committed Config-A script that produced `MMSTACK_OK` uses cu121 / 0.10.5 — left as-is.)

---

## Intentional config diffs vs the two parents (each stated in the config header too)

Both DIOR configs **merge** a model block from the DOTA-zoo config with the DIOR-R
dataset/schedule/optimizer conventions. Verified locally: each model block is
**byte-identical to its DOTA-zoo parent modulo `num_classes`**, and each
optimizer+pipeline+dataloader block is **byte-identical to the DIOR ORCNN parent**.

### `roi-trans-le90_r50_fpn_1x_dior.py`
- vs DOTA `configs_dota_zoo/roi-trans-le90_r50_fpn_1x_dota.py`:
  - `_base_` dataset `dota.py -> dior.py`.
  - both cascade `bbox_head[*].num_classes` `15 -> 20`.
  - trailing `optim_wrapper=dict(optimizer=dict(lr=0.005))` (SGD) **replaced** by AdamW
    lr=1e-4 + grad-clip (`_delete_=True`) — the DIOR-R disclosed stability recipe.
  - `train_pipeline` + `train_dataloader` restated with
    `mmdet.FilterAnnotations(min_gt_bbox_wh=(1e-2,1e-2))` after Resize (zero-area-GT NaN guard).
- vs DIOR `configs_dior/oriented-rcnn-le90_r50_fpn_1x_dior.py`: model block is RoI
  Transformer (CascadeRCNN 2-stage), not Oriented R-CNN. `angle_version` stays `le90`
  (both parents le90).

### `s2anet-le135_r50_fpn_1x_dior.py`
- vs DOTA `configs_dota_zoo/s2anet-le135_r50_fpn_1x_dota.py`:
  - `_base_` dataset `dota.py -> dior.py`.
  - `bbox_head_init.num_classes` and `bbox_head_refine[0].num_classes` `15 -> 20`.
  - **added** an AdamW `optim_wrapper` (the DOTA S2A-Net parent has none → would inherit
    schedule_1x SGD, which NaNs on DIOR-R).
  - `train_pipeline` + `train_dataloader` restated with `FilterAnnotations` (as above).
- vs DIOR ORCNN parent: model block is S2A-Net (RefineSingleStageDetector), and
  `angle_version` stays **le135** — S2A-Net's native coder/anchor convention, a
  per-detector field, not a dataset field. Inert downstream: `score_rtmdet.py`
  canonicalizes every prediction to le90 before writing the detections JSONL.

**S2A-Net admissibility.** The DOTA zoo excluded S2A-Net from its *inference* arm because
the released S2A-Net *checkpoint* is v0.1.0-era and incompatible with mmrotate dev-1.x.
In-house *training* with the dev-1.x config builds and trains the model from that config
and never loads a v0.1.0 checkpoint, so there is no incompatibility here (stated in the
config header and the box-script header).

---

## Local verification done (no GPU)

- `bash -n next_boot_rotcert_configB.sh` → OK.
- `next_boot_rotcert_configB.sh --dry-run` → `CONFIGB_DRYRUN_OK`, exit 0 (full plan +
  per-stage GPU-hour estimates printed; box-only paths reported, not failed).
- `python3 -m py_compile` on both DIOR configs → OK.
- grep-verified: both DIOR configs have `num_classes=20` and **no residual `num_classes=15`**;
  RoI-Trans `le90`, S2A-Net `le135`.
- Model blocks byte-identical to DOTA-zoo parents modulo `num_classes` (diff-confirmed).
- AdamW `optim_wrapper` + `train_pipeline` + `train_dataloader` byte-identical to the
  DIOR ORCNN parent (diff-confirmed).
- `DIOR_CLASS_NAMES` string byte-identical to `next_boot_rotcert_dior_infer.sh`
  (METAINFO label order — a transpose would mislabel every detection).
- `box_build_mmstack.sh` `bash -n` OK; the mmdet-gate sed tested on a stub `__init__.py`.
- No new dependencies added. No debug tokens left.

---

## Remains box-day-only (cannot run locally)

- mm-stack build (`box_build_mmstack.sh`) → `/root/MMSTACK_OK`; DIOR-R + HRSC-MS data on
  the data disk; the `data/DIOR -> $DIOR_R_ROOT` symlink under `$MMR` (the script creates
  it defensively; DIOR base config uses the relative `data_root='data/DIOR/'`, resolved
  with `cwd=$MMR`).
- Actual training (RoI-Trans ~8 GPU-h, S2A-Net ~8 GPU-h, HRSC 3-seed ~3 GPU-h) and
  inference. mm-config validity beyond `py_compile` (registry build) is only checkable on
  the box with mmengine/mmdet/mmrotate installed.
- HRSC-MS unzip + `hrsc_run.sh` (its own convert angle-convention refuse-gate, smoke
  finite-loss gate, and `HRSC_RUN_ALL_DONE` sentinel are authoritative).

**Turnkey behavior.** Content-gated (checkpoint zip-CRC + JSONL distinct-image coverage
floors, never bare exit codes). `--resume` re-runs the script and skips any stage whose
`STAGE*` marker is already `OK`; within a stage, an existing integral checkpoint / covered
JSONL is reused. Clean marker `ROTCERT_CONFIGB_ALL_DONE`; any gate failure →
`ROTCERT_CONFIGB_PARTIAL_DONE` (same tar either way, so partial substrate still pulls).

---

## Expected outputs and where the LOCAL certify step picks up

Box tar (via `chain_epilogue`) carries:
- DIOR-R arm (`$RESULTS_DIOR = ${AUTODL_TMP}/rotcert_configB_dior_results/`):
  `dior_test_dets_roi_trans.jsonl`, `dior_test_dets_s2anet.jsonl`, `dior_test_gt.jsonl`
  (+ provenance), the two deployed configs, train logs, coverage txts.
- HRSC arm (`$RESULTS_HRSC = ${AUTODL_TMP}/hrsc_rotcert_results/`, produced by
  `hrsc_run.sh`): per-seed `hrsc_test_dets_{orcnn,rtmdet}[_seed{1,2}].jsonl`,
  `hrsc_test_gt.jsonl`, per-seed final checkpoints, configs, provenance.
- `configB_markers/` (all per-stage + per-detector markers).

**Local certify pickup (CPU, post-pull; unchanged tools):**
1. Per new cell, run `cert_cell.sh --dets <dets.jsonl> --gt <gt.jsonl> --out-dir <cell>`
   (match → calibrate gwd/naive-coord/hull → recall → audit gwd/naive → r20). New cells:
   - `roi_trans` DIOR-R  — gt = `dior_test_gt.jsonl` (this arm's, == the frozen
     `dior_infer_results_2026-07-11` GT; either is valid, deterministic).
   - `s2anet` DIOR-R     — same DIOR-R test GT.
   - HRSC seed-0 cells for ORCNN + RTMDet — gt = `hrsc_test_gt.jsonl`; the seed-1/2 dets
     are the training-variance replicates for the same GT.
2. Then the **coverage-matched regime extension** (COMPUTE-PLAN sec.5): append each new
   cell's `matched.jsonl` to the coverage-matched runner's `CELLS` list and re-run (CPU),
   including the HRSC 3-seed replicates for the detector-training-variance row. This is
   where the elongation hypothesis (GWD wins at matched coverage on HRSC's extreme
   aspect ratios) is tested for the DIOR-R cross-family and HRSC-seed cells.

`cert_cell.sh` and the coverage-matched runner are unchanged by this staging (referenced,
not modified).

---

## Risks / watch-items for box day

- **S2A-Net DIOR-R stability.** AdamW + FilterAnnotations were proven for the ORCNN/RTMDet
  DIOR arms; S2A-Net (FocalLoss one-stage) is new on DIOR-R. hrsc_run-style smoke isn't
  wired for the DIOR arm here — watch the first ~1k iters of `s2anet_train.log` for finite
  loss; if it NaNs, the lever is lr (drop 1e-4 → 5e-5) or add a smoke pre-gate.
- **RoI Transformer / S2A-Net batch on 24 GB.** Defaulted to batch 2 (safe, matches the
  schedule_1x/DOTA base). Env-tunable via `ROI_TRANS_BATCH` / `S2ANET_BATCH` if VRAM allows.
- **DIOR val-crash analog.** The RTMDet DIOR arm needed `pad_size_divisor=32` (non-uniform
  800px test images). RoI-Trans/S2A-Net both already carry `pad_size_divisor=32` in their
  data_preprocessor, so the CSPNeXt-style crash shouldn't recur — but the S2A-Net FPN
  `add_extra_convs='on_input'` path is untested on DIOR test sizes; watch inference.
- **`data/DIOR` symlink.** The script creates `$MMR/data/DIOR -> $DIOR_R_ROOT` only if
  absent; if a stale symlink points elsewhere, training dies on `ImageSets/Main/train.txt`.
  Confirm it resolves before a long run.
- **"infer val" wording.** The task said "infer val"; implemented as DIOR-R **test** split
  (the only held-out split; trainval is training data) to keep the new cells on the same
  grid as the existing DIOR cells. Flagged here in case a val-split product was intended.
- **box_build_mmstack.sh gate value.** The sed sets `mmdet_maximum_version='3.3.0'`
  (mmrotate's assertion is `<=`, so 3.3.0 passes). If a future mmdet pin > 3.3.0 is used,
  bump the sed target too.

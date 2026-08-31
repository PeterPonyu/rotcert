# rotcert data manifest

Source of record: the frozen RotCert protocol specification (revised with the
2026-07-10 review note) and the SOTA reproduction plan §1/§3/§5 (the
RTMDet-R-l anchor, mmrotate staleness, container spec).

## Datasets

**DOTA-v1.0** (Xia et al., CVPR'18, arXiv:1711.10398 [VERIFY]). 15 classes, 2806
images, ~188k instances. train/val public GT, test GT server-only. This tool
calibrates and evaluates on **val only** (design §4.2); the test server is touched at
most once per PAPER, for the final reported mAP, never for certification.

- Local mirror (read-only, box-side): `${AUTODL_PUB}/DOTA` (~30 GB, zero
  download).
- mmrotate split-crops (1024x1024, 200px overlap) add roughly another ~2x on disk.
- **v1.5 is a Phase-3 ablation arm ONLY** — it is a superset of v1.0 and cannot serve
  as the independent second-dataset arm (design §4.2, binding).

**DIOR-R** (oriented DIOR, Cheng et al., TGRS 2022 [VERIFY]). 20 classes, ~23k images
800x800, ~190k oriented instances. **NOT in autodl-pub** — box-side download item,
gated by `orchestration/fetch_dior_r.py`'s size/checksum/license/count audit (design
§4.2, a HARD Phase-0 gate). If any of those gates fails (download blocked, license
non-permissive, counts mismatch), the second dataset falls back to **HRSC2016** or
**DOTA-v2.0** — never DOTA-v1.5 (barred, see above).

- **Staged state (2026-07-10):** DIOR-R is **fully staged from ModelScope** to the box
  data disk at `${AUTODL_TMP}/dior_r/` (project state): 23463 images = 11725 trainval
  + 11738 test (`JPEGImages-trainval` / `JPEGImages-test`), one OBB xml per image under
  `Annotations/Oriented Bounding Boxes/`, and split lists under `ImageSets/Main/`. Still
  owed at Phase 0: the size/checksum/per-class-count audit + the human **license review**.
- **Checkpoint gap (2026-07-10 survey, design §4.1 A2 addendum):** **no license-clean
  DIOR-R-trained checkpoint exists anywhere** — the mmrotate zoo is DOTA/HRSC-only, AOPG
  publishes DIOR-R tables but no downloadable weights, and LSKNet (CC-BY-NC) ships no
  DIOR-R weights either. So the DIOR-R arm is **inference + in-house training**: both
  detectors are trained on DIOR-R trainval via Apache-2.0 mmrotate dev-1.x
  (`orchestration/next_boot_rotcert_dior_train.sh`; ~45-60 GPU-h, revised total budget
  ~90-110 GPU-h). The DIOR-R **reproduction gate (K3)** targets the **AOPG published
  DIOR-R table** (`jbwang1997/AOPG`, Apache-2.0), NOT a zoo val mAP.
- **License gate (HARD; addendum A7):** the DIOR-R terms review is a Phase-0 gate for
  *use* AND now governs *rehosting* of the in-house-trained weights. `fetch_dior_r.py`'s
  `--confirm-license-reviewed` flag and `next_boot_rotcert_dior_train.sh`'s
  `DIOR_R_LICENSE_REVIEWED_MARKER` file gate both enforce it; if redistribution is barred
  or ambiguous, the release is **result-JSONs-only** (no checkpoint hosting).
- **Training-seed policy — OPEN prereg decision:** 1 seed with disclosure (cheap) vs 3
  seeds (~2x DIOR-R training GPU-h). The training chain parameterizes it
  (`ORCNN_SEEDS`/`RTMDET_R_SEEDS`, default "0"); settled at prereg freeze, which also
  reopens §4.1's frozen-checkpoint disclosure for the DIOR-R arm.

## The scene-level discipline (binding, C1 fix)

**The unit of exchangeability, splitting, and resampling is the SOURCE SCENE (the
original DOTA image), never the crop.** DOTA is tiled into overlapping 1024x1024
crops, so one physical object can appear in >= 2 crops; a crop-level split would place
the same object in both calibration and evaluation, silently inflating coverage
(design §4.2, CRITICAL). Every crop derived from one scene is assigned to the same
split part (`rotcert.splits.three_way_scene_split`); DIOR-R images are already
single-tile, so scene == image there.

Scene-id extraction from DOTA crop filenames follows the mmrotate split-crop naming
convention: `P0006__1024__0___0.png` -> scene id `P0006` (image-id prefix, then the
crop window/offset suffix separated by double underscores). This is a NAMING
CONVENTION, not a mathematical fact — `rotcert.splits.DEFAULT_SCENE_ID_REGEX` is
configurable; verify against the actual staged filenames at Phase 0 before trusting
the default on a dataset variant not yet checked.

`rotcert.io.populate_scene_ids` fills `scene_id` from `image_id` for any record
missing it; `rotcert.splits.assert_scene_level_splits` REFUSES (raises) rather than
silently falling back to crop-level splitting when `scene_id` is entirely absent.

## Canonical detections/GT JSONL schema (`rotcert/io.py`)

Detection record: `image_id` (crop/tile id), `scene_id` (source image id, may be
auto-derived), `class`, `obb` (`[cx, cy, w, h, theta]`, theta radians, ANY input
convention — canonicalized to le90 by every scoring function downstream, not at
ingest time), `score` (detector confidence). GT record: same minus `score`, plus
optional `gt_id`. Matched record (`rotcert match` output): `image_id`, `scene_id`,
`class`, `pred_obb`, `gt_obb`, `pred_score`, `iou`, `match_type` (`tp`/`fp`/`fn`).

## Detectors

**RTMDet-R-l** (arXiv:2212.07784 [VERIFY]), mmrotate zoo, Apache-2.0 — the ANCHOR
(only fully license-clean high-accuracy option; SOTA-REPRODUCTION-PLAN §1). Published
DOTA-v1.0 mAP 81.3 MS / 78.9 SS (test-set, multi-scale) — the reproduction target
(K3) is the **val, single-scale, no-TTA** zoo-consensus number instead (design §4.2's
val-only protocol), within `ROTCERT_REPRO_TOL` (default 0.5 mAP points, env-
overridable, `orchestration/phase0.py`).

**Oriented R-CNN** R-50-FPN (arXiv:2108.05699 [VERIFY]), mmrotate zoo, Apache-2.0 —
the second detector (two-stage, tests score-agnosticism). **LSKNet-S+ORCNN**
(arXiv:2303.09030 [VERIFY]) is an OPTIONAL Phase-3 exploratory arm, CC-BY-NC
(research-only; derived checkpoints not redistributable, but derived RESULT JSONs
are, per design §4.1).

mmrotate is stale (last release 2023-02, design §7 risk register): **a commit MUST be
pinned** (`--mmrotate-commit`, required, no default baked into
`orchestration/score_rtmdet.py`), configs vendored, mmcv/mmdet versions recorded
alongside it. `rotcert`'s core package (`rotcert/*.py`, everything under `tests/`)
**never imports mmrotate/mmcv/mmdet/torch** — only `orchestration/score_rtmdet.py`
does, lazily, so the core stays pip-installable and testable with numpy+scipy+shapely
alone (SOTA-REPRODUCTION-PLAN's binding rule: "our core never imports mmrotate").

## Container spec (per `SOTA-REPRODUCTION-PLAN-2026-07-10.md` §5, revised §4.7 of the
design for +DIOR-R)

1x RTX 4090D 24 GB, 64 GB CPU RAM recommended (32 GB floor), **~170 GB data disk
provisioned at container creation** (~30 GB DOTA raw + ~2x split-crops + ~20-25 GB
DIOR-R + **mmrotate work_dirs ~15 GB** for the in-house DIOR-R training + staging
headroom + results). Compute ceiling **~90-110 GPU-h** (design §4.7 + the A2 addendum):
the ≤46 GPU-h inference/repro table PLUS ~45-60 GPU-h of new in-house DIOR-R training
(Oriented R-CNN R-50 1x ~8-10 GPU-h; RTMDet-R-l 3x ~25-35 GPU-h at batch ~8; +~15 GPU-h
contingency). The DOTA arm remains inference-only + CPU-once-cached; only the DIOR-R arm
carries training, forced by the checkpoint gap above. Nothing exceeds 24 GB at batch-8
training or inference.

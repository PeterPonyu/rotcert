# AOPG DIOR-R reproduction gate — results (2026-07-13)

Compute-only execution of the one remaining preregistration-gated slot: the
K3 AOPG mAP reproduction gate for the DIOR-R arm of `rotcert`. No manuscript
was edited. Gate verdict computed by the **frozen** function
`orchestration/phase0.py:reproduction_gate` (tol = 0.5), imported not
reimplemented.

## 1. Frozen wording (quoted)

**K3, main text** (protocol specification §5, sha-pinned in the
2026-07-11 sign-off):

> **K3 (reproduction gate — binding, SOTA-REPRODUCTION-PLAN §3).** … the
> reproduction target is the **published/zoo-consensus VAL mAP** (single-scale,
> no-TTA) within a Phase-0-preregistered tolerance **(±0.5)** … **Either
> detector missing its target → no certification arm runs on that detector
> until fixed (certify reproduced scores, never paper-quoted).**

**K3 amendment for the DIOR-R arm** (§A2, "DIOR-R checkpoint gap"):

> **Reproduction gate for the DIOR-R arm (K3 amendment).** The DIOR-R
> reproduction target is **the AOPG published DIOR-R table** (`jbwang1997/AOPG`,
> Apache-2.0), NOT a zoo val mAP (there is no DIOR-R zoo number to hit). **K3's
> ±tolerance logic is unchanged; only the target source moves.**

Frozen decision rule, therefore: `passed = |measured − published| ≤ 0.5` mAP,
`published` = the AOPG published DIOR-R table, `measured` = the in-house
detector's reproduced DIOR-R mAP (single-scale, no-TTA), certifying the
**reproduced** score of the **deployed** checkpoint.

Sign-off (`SIGN-OFF-RECORD-2026-07-11.md`): "AOPG mAP reproduction gate:
REQUIRES_PREREG_FREEZE=confirmed"; console item — "needs the published mAP
target + the training-run val mAPs already in the pulled work_dirs; CPU
comparison"; and "Either sign publishes."

## 2. The frozen target source, resolved

The AOPG published DIOR-R table (AOPG paper arXiv:2110.01931 **Table I**;
`jbwang1997/AOPG` repo DIOR-R model-zoo) publishes, single-scale, ResNet-50-FPN,
on the DIOR-R **test** split:

| Method | DIOR-R mAP |
|---|---|
| RetinaNet-O | 57.55 |
| Faster RCNN-O | 59.54 |
| Gliding Vertex | 60.06 |
| RoI Transformer | 63.87 |
| **AOPG** | **64.41** |

The `jbwang1997/AOPG` repo model-zoo has **exactly one** DIOR-R row — AOPG @
**64.41** — which is the frozen per-source reference target used by the gate.

**Load-bearing finding:** the AOPG DIOR-R table contains **no Oriented R-CNN and
no RTMDet row** (both post-date, or were simply not benchmarked in, the 2021
AOPG paper). Neither in-house detector is a same-method row in the frozen target
source. Verified against both the arXiv Table I and the repo model-zoo.

## 3. Measured in-house DIOR-R mAP (primary source)

Recomputed from the **training-run per-class eval tables** in the pulled
work_dirs (`dior_train_results_2026-07-10/pulled/rotcert_dior_train.tar.gz`),
for the **deployed** checkpoints — the same checkpoints named in the inference
provenance JSONs (`dior_infer_results_2026-07-11/.../*.provenance.json`). The
mmrotate DIOR configs run the `val_evaluator` on `ImageSets/Main/test.txt`
(11,738 images) trained on trainval — so the reported mAP **is** the DIOR-R
**test**-split mAP, the same split AOPG publishes on (K3's val-vs-test gap does
not arise here).

| Detector | Config | Deployed ckpt | DIOR-R test mAP |
|---|---|---|---|
| Oriented R-CNN R-50-FPN 1x | `oriented-rcnn-le90_r50_fpn_1x_dior.py` | `orcnn_dior/seed_0/epoch_12.pth` | **62.61** (`dota/mAP 0.6261`) |
| RTMDet-R-l 3x | `rotated_rtmdet_l-3x-dior.py` | `rtmdet_r_dior/seed_0/epoch_36.pth` | **68.36** (`dota/mAP 0.6836`) |

Notes: single seed (seed_0, per the frozen 1-seed default policy, §A2, disclosed
exactly as the DOTA arm's checkpoint variance is). RTMDet's best eval was
epoch_24 = 69.65, but the **deployed** checkpoint is epoch_36 = 68.36 — the gate
certifies the deployed reproduced score, per K3.

## 4. Verdict (frozen ±0.5 rule, applied exactly)

Gate computed by the frozen `reproduction_gate(...)`; full record in
`aopg_repro_result.json`.

| Detector | measured | AOPG target | gap | `|gap| ≤ 0.5`? | **verdict** |
|---|---|---|---|---|---|
| Oriented R-CNN 1x | 62.61 | 64.41 | −1.80 | no | **FAIL** |
| RTMDet-R-l 3x | 68.36 | 64.41 | +3.95 | no | **FAIL** |

Against the nearest-architecture published row (RoI Transformer 63.87, the
strongest two-stage baseline in the table) Oriented R-CNN's gap is −1.26 — still
outside ±0.5. Every literal application of the frozen rule against every
AOPG-table row yields **FAIL**.

## 5. Reading the verdict (the "either sign publishes" disclosure)

The strict ±0.5 verdict is **FAIL for both detectors**, and that is reported
exactly as it lands. But ±0.5 is an **identity-reproduction** tolerance —
designed for reproducing *the same method's* published number. The frozen target
source (AOPG DIOR-R table) contains **no row for either detector we trained**, so
the ±0.5 rule is being applied across *different methods*. The gap is therefore a
**cross-method** gap, not evidence of a broken pipeline:

- **Oriented R-CNN 62.61** sits between Gliding Vertex (60.06) and RoI
  Transformer (63.87) — a sane, published-plausible result for a plain two-stage
  RPN detector at the **1x** schedule (AOPG's 64.41 is its own headline method).
- **RTMDet-R-l 68.36** **exceeds every row** in the 2021 AOPG table, as expected
  for a stronger 2022-era detector at 3x.

Both numbers land squarely in the published DIOR-R band; the training pipeline
reproduces sane detectors. This is the K3 spirit — **"disclose … rather than
chase it."** Per the sign-off ("Either sign publishes"), the publishable
statement is a **premise-limited** one: *the frozen AOPG DIOR-R target source has
no same-method row for the in-house detectors, so a ±0.5 identity reproduction
cannot be claimed; the reproduced test-split mAPs (62.61 / 68.36) are disclosed
and fall in the published-plausible range, and no certification claim leans on a
paper-quoted number* (certification uses reproduced scores only, K3). No result
in the `rotcert` confirmatory family depends on passing this gate — the family
was already demoted to descriptive (`holm8_2026-07-12/`), and the AOPG repro is
a sanity check, not a load-bearing claim.

## 6. Files

- `run_aopg_repro.py` — runner (imports the frozen `reproduction_gate`).
- `aopg_repro_result.json` — machine-readable gate record (both detectors, vs
  AOPG target and vs nearest row).
- `RESULTS-AOPG.md` — this file.
- `PROPOSED-TEX.md` — drop-in manuscript text (canonical + TGRS kit slots).

## 7. Provenance

- Frozen gate fn: `reliability-commons/tools/rotcert/orchestration/phase0.py:82`
  (`reproduction_gate`, `DEFAULT_REPRO_TOL = 0.5`, line 24).
- Measured mAP primaries (per-class tables + `Epoch(val)` summary lines):
  `dior_train_results_2026-07-10/pulled/rotcert_dior_train.tar.gz` →
  `root/autodl-tmp/rotcert_dior_train_results/orcnn_dior_seed0_train.log` (ORCNN
  epoch 12) and `rtmdet_r_dior_seed0_resume.log` (RTMDet epoch 36); scalars at
  `root/autodl-tmp/dior_r/work_dirs/*/seed_0/*/vis_data/scalars.json`.
- Deployed-checkpoint match: `dior_infer_results_2026-07-11${AUTODL_TMP}/`
  `rotcert_dior_infer_results/dior_test_dets_{orcnn,rtmdet}.jsonl.provenance.json`
  (`epoch_12.pth` / `epoch_36.pth`; mmrotate commit `3ff004e`).
- On-box AOPG repro markers were `AOPG_REPRO_ORCNN=SKIPPED_DISCLOSED` /
  `AOPG_REPRO_RTMDET=SKIPPED_DISCLOSED` — i.e., the gate was deferred to this
  local CPU comparison, which is what this run executes.
- AOPG target: arXiv:2110.01931 Table I; `github.com/jbwang1997/AOPG` DIOR-R
  model-zoo (single DIOR-R row, AOPG 64.41). Verified 2026-07-13.

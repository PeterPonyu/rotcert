# Fourth grid cell: published Oriented R-CNN on DOTA-v1.0 val (2026-07-11)

**Provenance.** Detections: published mmrotate checkpoint
`oriented_rcnn_r50_fpn_fp16_1x_dota_le90-57c88621.pth` scored over the SAME
5,297 dota_split val tiles as the pilot (box round 2 — round 1 was discarded:
full-image substrate + METAINFO class-order mislabel, see UPDATE 40). Dets
92,414 records / 4,608 distinct tiles (689 tiles have no det above score 0.05
— legitimate two-stage sparsity; pilot RTMDet covered 5,124). GT: the pilot's
frozen `dota_val_gt.jsonl`. Match: tp=50,943 fp=41,471 fn=4,861 (IoU 0.5,
rotated). Pipeline identical to the other three cells (r20_generic.py, α=0.10).

## G1 out-of-sample (R=20 scene splits, BCa 95%) — PRIMARY
ORCNN-DOTA: **0.8902 [0.8780, 0.8995]** (min 0.8283, max 0.9280, std 0.0249)

Completed 2×2 grid (all OOS R=20):

| G1 coverage | DOTA-v1.0 | DIOR-R |
|---|---|---|
| Oriented R-CNN | 0.8902 [0.878, 0.900] | 0.9004 [0.899, 0.902] |
| RTMDet-R-l | 0.8879 [0.877, 0.898] | 0.9012 [0.900, 0.903] |

BOTH detector families are ~1–1.2 pt below nominal on DOTA and at nominal on
DIOR-R — the full cross confirms the DOTA shortfall is a DATASET artifact
(overlapping 1024-px crops breaking scene-level exchangeability), not a
detector or score property. This is the strongest form of the paper's
same-detector-two-datasets argument (now: both detectors, both datasets).

## In-sample angle-strata diagnostic (secondary; whole-set)
- gwd: marginal 0.9000 (identity, as expected) | interior 0.8964, boundary
  0.9350, square **0.8612**
- naive-coord: marginal 0.9328 | interior 0.9494, boundary 0.9056, square
  **0.8591**

HONEST NUANCE (affects one paper sentence): on THIS cell GWD's square-regime
coverage (0.8612) is nearly as low as naive's (0.8591) — unlike the pilot
RTMDet-DOTA cell, where GWD held ~0.88 at square vs naive 0.8405. "GWD is
seam/square-stable in every cell" must be scoped: GWD is square-stable on
DIOR-R (both detectors) and degrades least on RTMDet-DOTA, but on ORCNN-DOTA
both scores show a comparable square-regime deficit. The square regime on
DOTA is hard for everyone; GWD's advantage there is detector-dependent.
The rigorous score comparison remains the prereg-gated Holm-8.

Files: matched.jsonl, r20_coverage.json, r20.log, audit_gwd.json,
audit_naive.json, match.log (this dir). Box round: orcnn_dota_results_2026-07-11/
(dets + markers + provenance).

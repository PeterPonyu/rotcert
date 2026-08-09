# HRSC training-seed variance — verified digest (2026-07-14)

Config-B's HRSC 3-seed arm (box run, seeds 0,1,2 × {Oriented R-CNN 1x, RTMDet-R 3x},
in-house training; seed 0 = the Config-A cells, seeds 1–2 certified locally via the
byte-repro-verified `orchestration/cert_cell.sh` chain on the pulled detections).
Sources: `configA_cert_2026-07-14/hrsc_{orcnn,rtmdet}/` (seed 0) and
`hrsc_seedcells_2026-07-14/{orcnn,rtmdet}_seed{1,2}/` — every number below is read
directly from `audit_gwd.json` / `r20_coverage.json` / `recall.json`.
All at α=0.10 (nominal coverage 0.90), GWD score, frozen pipeline.

## Per-seed certification quantities

| detector | seed | audited GWD coverage [CI] | n TP | R=20 mean (min–max) | G2 (LTT-HB β=.2 δ=.05) |
|---|---|---|---|---|---|
| Oriented R-CNN | 0 | 0.9011 [0.8798, 0.9201] | 799 | 0.8989 (0.8528–0.9500) | refused (power floor) |
| Oriented R-CNN | 1 | 0.9023 [0.8803, 0.9228] | 727 | 0.8982 (0.8414–0.9360) | refused (power floor) |
| Oriented R-CNN | 2 | 0.9024 [0.8780, 0.9246] | 758 | 0.9005 (0.8249–0.9545) | refused (power floor) |
| RTMDet-R | 0 | 0.9015 [0.8760, 0.9227] | 670 | 0.9085 (0.8659–0.9568) | refused (power floor) |
| RTMDet-R | 1 | 0.9016 [0.8799, 0.9235] | 630 | 0.9113 (0.8839–0.9402) | refused (power floor) |
| RTMDet-R | 2 | 0.9023 [0.8796, 0.9232] | 737 | 0.8903 (0.8444–0.9311) | refused (power floor) |

## The claim this supports (scope it exactly like this)

- **Audited coverage is training-seed-stable**: within-detector spread across seeds is
  ≤0.0013 (ORCNN 0.9011–0.9024; RTMDet-R 0.9015–0.9023), an order of magnitude smaller
  than the CI half-widths (~±0.021) — the certificate's coverage behavior is a property
  of the frozen calibration procedure, not of a lucky training run.
- **R=20 out-of-sample means all straddle nominal** (0.8903–0.9113 across the six cells);
  the wider RTMDet-R spread (0.021) is within the per-cell reshuffle range (each cell's
  own min–max spans ≈0.07–0.13).
- **Refusal is seed-stable too**: the G2 LTT-HB power-floor refusal (n_img=440 < floor)
  fires identically in all six cells — refusal behavior does not flicker with training
  randomness.
- Matched-TP counts vary with seed (630–799) because detection sets differ; this is the
  expected detector-side variation the certification wraps, not a pipeline instability.

## Honesty notes
- Post-hoc/exploratory (Config-B arm; not in any preregistered confirmatory family).
- Seed 0's numbers are the already-published Config-A cells — the seed arm ADDS seeds
  1–2, it does not recompute seed 0.
- DIOR-R detector-depth cells (RoI-Transformer, S2A-Net) are still training on the box
  and are NOT part of this digest.

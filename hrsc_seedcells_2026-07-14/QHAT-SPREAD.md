# HRSC 3-seed arm — per-seed q̂ (region-scale) spread (2026-07-14)

Extension of `SEED-VARIANCE-DIGEST.md`. Marginal coverage lands at nominal (0.90) for
every seed *because split-conformal calibration guarantees it under exchangeability* —
that is not a nontrivial stability discovery, it is what the procedure pins by
construction, and the reading here is only that **no seed-driven exchangeability
breakdown occurred**. The genuinely seed-sensitive quantity is the region *scale*: the
conformal quantile q̂ (the GWD nonconformity radius, in Gaussian--Wasserstein score
units). This table reports it.

Source: the six frozen `r20_coverage.json` files — `q_hat_across_repeats.mean` plus the
per-repeat `q_hat` min/max over the 20 scene reseeds. Extracted by
`extract_qhat_spread.py` (alongside this file); every number below is reproduced by
running it.

## Per-seed q̂ across the R=20 reseeds

| detector | seed | q̂ mean | q̂ [min, max] | q̂ std |
|---|---|---|---|---|
| Oriented R-CNN | 0 | 40.17 | [36.95, 45.34] | 1.95 |
| Oriented R-CNN | 1 | 39.57 | [35.98, 42.15] | 1.72 |
| Oriented R-CNN | 2 | 39.84 | [34.10, 44.61] | 2.31 |
| RTMDet-R | 0 | 49.14 | [46.06, 52.55] | 1.60 |
| RTMDet-R | 1 | 54.26 | [51.05, 58.67] | 2.76 |
| RTMDet-R | 2 | 48.23 | [45.33, 52.52] | 1.97 |

## Across-seed spread of the mean q̂ (within detector family)

| detector | mean-q̂ range across seeds | spread | as % of family mean |
|---|---|---|---|
| Oriented R-CNN | [39.57, 40.17] | 0.61 | 1.5% |
| RTMDet-R | [48.23, 54.26] | 6.03 | 11.9% |

## Reading (scope it exactly)

- Coverage is nominal for every seed by construction (exchangeability held; no breakdown).
- The region scale q̂ *is* seed-sensitive, and unevenly so across detectors: Oriented
  R-CNN's mean q̂ moves only 1.5% across the three training seeds, while RTMDet-R's moves
  ~12% (48.2–54.3). So the certificate's *coverage* is seed-robust, but the *size* of the
  certified region carries real training-seed variation, larger for RTMDet-R.
- Post-hoc/exploratory Config-B arm; seed 0 = the already-certified Config-A cells (not
  recomputed). Not part of any preregistered confirmatory family.

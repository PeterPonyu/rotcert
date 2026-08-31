# Coverage-matched efficiency ablation (Part A) — RESULTS

**Date:** 2026-07-13 · **Runner:** `coverage_matched_runner.py` · **Output:** `results.json`
**Purpose:** repair the efficiency-claim support gap (internal review items M3 / M1). The Holm-8 head-to-head and
its R=20 companion contrast GWD vs. baselines at **identical nominal** α=0.10, where the union-bound
baselines **over-cover** (realized eval coverage 0.915–0.948) while GWD sits near nominal. Part of the
GWD size advantage is therefore the price of baseline over-coverage, not a sharper region. This
ablation removes that confound by re-tuning each baseline to a **common realized coverage** before
comparing region sizes.

## Protocol
Same 4 `(detector,dataset)` cells, 2 baselines (`naive-coord`, `hull`), and 40/20/40 scene split
(split-seed = repeat index) as `holm8_2026-07-12/holm8_r20_exploratory.py`. Per split: calibrate on the
**matching** scenes (pooled split-conformal, `mondrian_field=None`), evaluate on the disjoint **eval**
scenes. GWD is held at nominal α=0.10 → realized eval coverage `c_gwd`, region area
`A_gwd = π q̂²`. Each baseline is re-calibrated at an adjusted level **α′** (bisection on the
calibration-set quantiles; realized coverage is monotone in α′) so that its realized eval **joint**
coverage matches `c_gwd`. Region area `A_base(α′) = 4 q_cx q_cy`. Reported metric:
`ratio = A_gwd / A_base(α′)` (the same `(cx,cy)`-slice area used throughout; `<1` favors GWD).

Both the GWD ball and the baseline Bonferroni box are **joint** ("the true box lies in the region")
statements, so matching realized coverage is apples-to-apples. `naive-coord` and GWD both certify the
full 5-DOF oriented box (naive uses Euclidean θ); `hull` certifies only the coarser 4-DOF
axis-aligned envelope (no angle). **This ablation is post-hoc and coverage-fair, NOT preregistered or
confirmatory** — α′ is selected against realized eval coverage, so it is an efficiency-at-a-fixed-
operating-point comparison, not a coverage guarantee for the baseline. The region *shape* is still
calibrated out-of-sample; only the scalar level is retuned to a common target.

**Validation gate (passed):** at α′=0.10 the runner reproduces the `holm8_r20_exploratory` baseline
coverages and nominal size ratios; the GWD quantile from the fast per-pair path is asserted equal
(<1e-9) to `certify.g1_calibrate`'s SplitConformal threshold on split 0 of every cell. Frozen
split-0 RTMDet·DOTA nominal ratios 0.67/0.86 (vs naive/hull) reproduce the paper's frozen Holm-8
numbers exactly.

## Headline result — region-size ratio (GWD ÷ baseline) at MATCHED realized coverage

| Cell | GWD cov (R20) | vs **naive-coord** nominal → **matched** [min,max] (splits GWD-smaller) | vs **hull** nominal → **matched** [min,max] (splits GWD-smaller) |
|---|---|---|---|
| RTMDet·DOTA  | 0.883 | 0.600 → **0.929** [0.471,1.256] (12/20) | 0.776 → **1.484** [1.359,1.631] (0/20) |
| ORCNN·DOTA   | 0.883 | 0.577 → **0.931** [0.417,1.229] (12/20) | 0.749 → **1.459** [1.336,1.609] (0/20) |
| ORCNN·DIOR-R | 0.899 | 0.343 → **0.791** [0.673,0.886] (20/20) | 0.494 → **1.759** [1.655,1.812] (0/20) |
| RTMDet·DIOR-R| 0.900 | 0.382 → **0.825** [0.741,0.917] (20/20) | 0.549 → **1.698** [1.629,1.818] (0/20) |

Matched-coverage quality: baseline realized coverage lands on `c_gwd` to a mean absolute gap
≤ 0.0002 in every cell (α′ ≈ 0.145–0.204). R=20 means shown; per-split arrays in `results.json`.

## What survives coverage-matching (the honest bottom line)

1. **The large nominal advantage was mostly baseline over-coverage.** Every ratio moves toward (or
   past) 1 once realized coverage is equalized: e.g. ORCNN·DIOR-R vs naive 0.34 → 0.79; vs hull
   0.49 → 1.76.

2. **vs the like-for-like `naive-coord` baseline (both certify the full oriented box):** GWD's
   advantage **survives on DIOR-R** — ≈0.79× (ORCNN) and ≈0.83× (RTMDet), GWD smaller on **all 20/20
   splits, both detectors**. On **DOTA it does not survive** — a wash (mean ≈0.93, only 12/20 splits
   favor GWD, range spans 1). The DOTA null is consistent with GWD's DOTA under-coverage (matching
   pushes the baseline to a sub-nominal ≈0.883 target, where the naive box is already tight).

3. **vs the coarser `hull` baseline:** GWD is **larger at matched coverage in all 4 cells**
   (1.46–1.76×, 0/20 splits favor GWD). The entire nominal hull advantage was over-coverage **plus**
   hull certifying a lower-dimensional object — it drops the angle, so its `(cx,cy)` footprint is
   tighter at matched marginal coverage. This is a representational caveat, not a GWD defect, but it
   means the nominal "GWD smaller than hull" claim does not hold at matched coverage.

**Consequence for the manuscript.** The abstract's "near-nominal coverage at 0.33–0.86× the baseline
region size" is a nominal-level (over-coverage-confounded) statement and is replaced. The single
surviving, coverage-fair efficiency advantage is **GWD ≈0.8× the like-for-like naive Bonferroni
oriented-box region on DIOR-R (both detectors, all 20 splits)**; elsewhere the nominal advantage is
the price of baseline over-coverage (and, for hull, of certifying the coarser axis-aligned envelope).
This coverage-matched ablation is now the primary quantitative characterization of efficiency
(post-hoc, coverage-fair, not confirmatory); the frozen Holm-8 stays preregistered-descriptive and the
R=20 nominal-ratio analysis is re-cast as split-sensitivity of the nominal contrast (no p-value).

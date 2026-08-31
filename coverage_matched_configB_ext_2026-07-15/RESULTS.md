# Coverage-matched efficiency ablation — Config-B extension — RESULTS

**Date:** 2026-07-15 · **Runner:** `coverage_matched_configB_ext_runner.py` · **Output:** `results.json`
**Extends:** `coverage_matched_2026-07-13/` (frozen parent, untouched). **New cells:** RoI Transformer ·
DIOR-R, S2A-Net · DIOR-R (`configB_cells_2026-07-15/{roi_trans,s2anet}/matched.jsonl`).

## Protocol (identical to the parent, zero deviations)
Every per-split helper (`_threshold`, `precompute_residuals`, `baseline_coverage_and_area`, `match_alpha`,
and the one-time `r==0` cross-check against `certify.g1_calibrate`'s SplitConformal threshold) is imported
from `coverage_matched_2026-07-13/coverage_matched_runner.py` and reused verbatim — this script only supplies
the two new `(detector, dataset, matched.jsonl path)` cells and drives the same loop. Same 40/20/40 scene
split (split-seed = repeat index), same 2 baselines (naive-coord, hull), GWD held at nominal α=0.10, each
baseline re-tuned to α′ so its realized eval joint coverage matches GWD's realized eval coverage on the same
split. **Validation gate (passed):** the `g1_calibrate` cross-check assertion (`abs(q_hat_fast - q_hat_g1_calibrate) < 1e-9`)
fired silently (no `AssertionError`) for both new cells — the fast per-pair path reproduces the frozen
certification's SplitConformal threshold exactly, as it does for the four original cells.

POST-HOC, coverage-FAIR, **NOT** preregistered/confirmatory (identical caveat to the parent; these two cells
were never part of the signed Holm-8 preregistration).

## Headline result — region-size ratio (GWD ÷ baseline) at MATCHED realized coverage

| Cell | GWD cov (R20) | vs **naive-coord** nominal → **matched** [min,max] (splits GWD-smaller) | vs **hull** nominal → **matched** [min,max] (splits GWD-smaller) |
|---|---|---|---|
| RoI Transformer · DIOR-R | 0.9010 | 0.33 → **0.76** [0.62,0.86] (20/20) | 0.47 → 1.75 [1.67,1.81] (0/20) |
| S2A-Net · DIOR-R         | 0.8984 | 0.38 → **0.85** [0.74,0.97] (20/20) | 0.54 → 1.66 [1.56,1.72] (0/20) |

Compare to the two original DIOR-R cells (`coverage_matched_2026-07-13/results.json`): ORCNN 0.79 [0.67,0.89],
RTMDet 0.82 [0.74,0.92] vs naive. The four-cell DIOR-R range vs naive-coord is now **0.76–0.85×**, GWD smaller
on all 20/20 splits in every one of the four cells. Vs hull the four-cell range was 1.46–1.76×; the two new
cells (1.66×, 1.75×) fall within that same band. Coverage-matched to ≤0.0001 abs gap in every cell.

## Scope notes carried with these cells

- The "nominal" column for the two new cells comes from this ablation's own R=20 nominal-α mean, not the
  preregistered Holm-8 family, which stays untouched — these cells were never part of that signed freeze.
- The coverage-fair efficiency statement widens from the two original detectors (ORCNN/RTMDet) to all four,
  over the 0.76–0.85× range against naive-coord.
- Angle-regime coverage for the two new cells is sourced directly from
  `configB_cells_2026-07-15/{roi_trans,s2anet}/{audit_naive.json,audit_gwd.json}` with no new computation.
  S2A-Net's naive-coordinate boundary coverage (0.867) is the only DIOR-R regime cell in the study that dips
  clearly below nominal — deeper than DIOR-ORCNN's marginal dip (0.898) — while its square over-coverage
  (0.988) is the highest of any cell. Direction (boundary worst, square best) is unchanged; magnitude is not.
- The aspect-ratio-conditioned regime ablation is **NOT extended** to these cells: that would require
  re-running `coverage_matched_regime_runner.py`'s conditioning logic, which was out of scope. Its
  "four original cells" framing therefore remains accurate as written.

Every number quoted for the new cells was re-extracted from `results.json` and
`configB_cells_2026-07-15/*/audit_*.json`. No existing table row or previously reported number was altered.

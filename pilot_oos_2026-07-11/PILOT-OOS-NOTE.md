# In-sample → out-of-sample G1 coverage correction (2026-07-11)

**Author:** write-geo. **Trigger:** red-team finding (verified). **Scope:** corrects the G1 marginal-coverage
number the rotcert paper reports; does not touch frozen inputs.

## The defect (confirmed)
The DOTA pilot's headline **G1 marginal coverage 0.9000** is an **in-sample** number, not a validity result.
`orchestration/next_boot_rotcert.sh` (stage-4 pilot) passes the **same** `matched.jsonl` to both
`rotcert calibrate` (which fits the split-conformal quantile $\hat q$ on that set) and `rotcert audit`
(which measures coverage on that same set). By the split-conformal construction the calibration-set coverage
is $\lceil(1-\alpha)(n+1)\rceil/n$ **by definition** — for the pilot, $48{,}719/54{,}131 = 0.900020\ldots$,
matching the reported point to ~16 significant figures. This is **tautological for any nonconformity score**
(GWD, naive-coord, hull all give ~0.90 in-sample) and therefore certifies nothing about held-out coverage.

## The fix: out-of-sample R=20 scene-split coverage (the correct estimand)
For each cell we apply the **already-designed** R=20 repeated scene-split protocol (40/20/40 cal/match/eval,
split-seed = repeat index): fit $\hat q$ on **calibration** scenes, measure coverage on **held-out eval**
scenes, 20 repeats, BCa CI across repeats. This is the number that should be reported as G1 from now on;
the in-sample `audit.json` values become secondary/diagnostic.

| Cell | old (IN-SAMPLE, tautological) | **new OUT-OF-SAMPLE R=20** (BCa 95% CI) | verdict |
|---|---|---|---|
| **DOTA pilot (RTMDet-R-l)** | 0.9000 | **0.8879** [0.8774, 0.8979] (min 0.843, max 0.923) | slightly **below** nominal |
| DIOR-R ORCNN (orig GT) | 0.9000 | 0.9001 [0.8985, 0.9019] | at nominal |
| DIOR-R ORCNN (fixed GT) | 0.9000 | (see DIOR-R-CERT-REPORT) | — |
| DIOR-R RTMDet-R | (pending) | (see DIOR-R-CERT-REPORT) | — |

(`pilot_oos_2026-07-11/dota_r20_coverage.json`; DIOR-R values in `dior_cert_results_2026-07-11/*/r20_coverage.json`.)

## What this means for the paper (honest)
- **The method survives, the pilot number does not.** Out-of-sample, the GWD certificate lands **at** nominal
  on DIOR-R (single-tile images, clean scene=image exchangeability, 0.900) and **just below** nominal on the
  DOTA pilot (0.888, CI upper bound 0.898). The ~1.2-point DOTA shortfall is the honest cost of DOTA's
  overlapping 1024-px crops: scene-level splitting removes cross-split object duplication but within-scene
  correlation and the crop heterogeneity leave a small residual under-coverage that the in-sample number hid.
- **The paper must replace "0.9000 marginal coverage" with the out-of-sample number** (DOTA 0.888
  [0.877, 0.898]) and state that G1 coverage is measured out-of-sample over R=20 scene splits. The
  dataset-dependence (DIOR-R at nominal, DOTA slightly under) is itself a finding worth reporting.
- **No fabrication or gaming** — the in-sample audit was a protocol bug in the pilot orchestration, not a
  fudged number; it is now corrected with the paper's own pre-designed R=20 machinery.

## Provenance
DOTA R=20 from the frozen pilot `matched.jsonl`
(`pilot_results_2026-07-10/pulled/.../pilot/matched.jsonl`) via `dior_cert_results_2026-07-11/r20_generic.py`;
CPU-only; suite unaffected (no source touched). Prereg-gated Holm-8 NOT run.

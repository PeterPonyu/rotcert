# Regime-conditional coverage-matched efficiency — RESULTS (2026-07-13)

**Runner:** `coverage_matched_regime_runner.py` (CPU, ~91 s, local). **Extends:**
`coverage_matched_2026-07-13/`. **Regression guard:** the `all` regime reproduces the frozen
parent ablation **bit-for-bit** (8/8 cells, `max_abs_diff = 0.0`) — the parent's per-split
helpers (`_threshold`, `match_alpha`) are imported and reused, so the conditioned numbers are
the same computation with an added aspect-ratio mask.

## Verdict (honest, leads)

**The elongation hypothesis HOLDS — against the naive-coord baseline, in all 4 cells, under
both a fixed AR=3 cut and a per-cell median cut.** At *matched realized coverage*, GWD's
(cx,cy)-region relative to the per-coordinate Bonferroni box shrinks monotonically as the
object gets more elongated. It does **not** hold against the hull, and it should not be claimed
to: the hull is a genuinely tighter oriented region regardless of shape (GWD loses to it, 1.4–1.8×,
on 0/20 splits in every regime). So this is a *repair of the efficiency story specifically vs the
naive per-coordinate CP baseline* — which is exactly the construction the paper's premise indicts
(naive CP degrades at the seam / for oriented boxes). It does not resurrect a "GWD beats every
baseline" claim.

The single most useful consequence: **the DOTA "wash" is explained, not explained away.** The
pooled DOTA ratio vs naive (~0.93, a wash) is the average of GWD *winning* on the elongated
minority and *slightly losing* on the compact majority — not evidence that GWD's region is never
sharper.

## Aspect ratio = max(w,h)/min(w,h) of the matched GT box (angle-convention–free)

| cell | n TP | AR mean | AR median | AR p90 | frac AR≥3 |
|---|---|---|---|---|---|
| RTMDet-R · DOTA | 54,131 | 2.69 | 2.47 | 4.20 | 0.291 |
| ORCNN · DOTA | 50,943 | 2.71 | 2.51 | 4.23 | 0.302 |
| ORCNN · DIOR-R | 89,125 | 2.38 | 2.25 | 3.55 | 0.223 |
| RTMDet-R · DIOR-R | 98,690 | 2.34 | 2.20 | 3.48 | 0.209 |

Note this already refutes the *between-dataset* version of the elongation story: **DOTA objects
are on average MORE elongated than DIOR-R** (mean AR 2.70 vs 2.35), yet GWD wins pooled on DIOR-R
and only ties on DOTA. So the between-dataset difference (DIOR win, DOTA wash) is **not** an
elongation effect — it is consistent with the DOTA crop/exchangeability artifact already
documented in the cert report. The elongation effect is a **within-cell, conditional** effect,
and that is where it is clean.

## Coverage-matched ratio A_gwd / A_naive at matched coverage (<1 ⇒ GWD smaller)

Coverage was matched to ≤0.0004 abs gap in every regime (apples-to-apples). `all` = frozen parent.

**Fixed cut, compact = AR<3, elongated = AR≥3:**

| cell | all (pooled) | compact (AR<3) | elongated (AR≥3) | gradient |
|---|---|---|---|---|
| RTMDet-R · DOTA | 0.929 | **1.027** | **0.852** | ✓ elongated < compact |
| ORCNN · DOTA | 0.931 | **1.076** | **0.743** | ✓ |
| ORCNN · DIOR-R | 0.791 | **0.985** | **0.867** | ✓ |
| RTMDet-R · DIOR-R | 0.825 | **0.974** | **0.863** | ✓ |

**Per-cell median cut (threshold-robustness; cut ≈ 2.2–2.5):**

| cell | compact (below median) | elongated (above median) |
|---|---|---|
| RTMDet-R · DOTA | 1.021 | 0.959 |
| ORCNN · DOTA | 1.105 | 0.823 |
| ORCNN · DIOR-R | 1.022 | **0.594** |
| RTMDet-R · DIOR-R | 1.018 | **0.651** |

Both cuts give the same direction in 4/4 cells. The median cut (balanced halves) sharpens it:
on DIOR-R the more-elongated half reaches **0.59–0.65×** while the less-elongated half sits at
~1.02× — GWD is ~40% smaller at matched coverage on the elongated half and a wash on the
compact half.

### On the apparent "pooled below both strata" (DIOR-R, fixed cut)
With the fixed AR=3 cut the pooled DIOR-R ratio (0.79) is below *both* AR=3 sub-strata
(0.99 compact, 0.87 elongated). This is a pooling/heterogeneity effect, not a contradiction: the
single axis-aligned naive box, when calibrated on the *mixed* AR population, must be inflated to
serve the elongated tail, so it pays a heterogeneity penalty on top of its per-object one — GWD
therefore looks even better pooled. The balanced median split removes this (pooled 0.79 lies
between the median halves 1.02 and 0.59), which is why the median cut is the cleaner statement of
the monotone gradient.

## vs the hull (reported for completeness — gradient does NOT hold, honest)

| cell | all | compact | elongated | GWD ever smaller? |
|---|---|---|---|---|
| RTMDet-R · DOTA | 1.484 | 1.453 | 1.537 | no (0/20 all regimes) |
| ORCNN · DOTA | 1.459 | 1.486 | 1.426 | no |
| ORCNN · DIOR-R | 1.759 | 1.723 | 1.816 | no |
| RTMDet-R · DIOR-R | 1.698 | 1.659 | 1.768 | no |

GWD is 1.4–1.8× the hull's (cx,cy)-slice area at matched coverage everywhere, and the elongation
gradient is inconsistent (elongated ≥ compact in 3/4 cells). The hull is the tighter *oriented*
region by construction; the efficiency contribution is not "smaller than the hull" and this file
does not claim it.

## The HRSC prediction this sets up (falsifiable, pre-registered here)

HRSC2016-MS ships (prefetched + gate-passed locally, 2026-07-13) are **far more elongated** than
either existing dataset: AR **mean 5.22, median 5.11, 82% AR≥3, 52% AR≥5** (n=3,611 GT over the
source train+test splits). Ships are almost entirely inside the "elongated" regime where GWD's
coverage-matched edge vs naive-coord is largest here (0.74–0.87×, and 0.59–0.65× on the
median-elongated DIOR-R half). **Prediction:** the HRSC cells produced by the Config-A box run
should show GWD's *pooled* coverage-matched ratio vs naive-coord land in the low-0.7s or below —
the most decisive win of any cell — because HRSC has essentially no compact majority to dilute it.
If HRSC instead comes back a wash, the elongation mechanism is weaker than the within-cell
gradient suggests, and the efficiency claim stays scoped to "elongation-graded vs naive-coord,
strongest on DIOR-R." Either way the within-cell gradient above stands on its own.

## Honesty caveats (unchanged from parent)

- POST-HOC, coverage-FAIR, **NOT** preregistered/confirmatory: α′ is tuned against realized eval
  coverage per regime; this is an efficiency-at-a-fixed-operating-point comparison, not a coverage
  guarantee for the baseline.
- The regime is a property of the **GT** object (elongation of the matched ground-truth box), so
  the strata are defined without reference to the certified region — no circularity.
- The correctness contributions (seam-continuity, square-safety, coupled G1/G2) do not depend on
  any of this; this analysis only concerns *efficiency*.

## Files
- `coverage_matched_regime_runner.py` — runner (imports parent helpers; regression-guarded).
- `results.json` — full per-(cell,baseline,regime) table incl. per-split ratios, coverage-match
  gaps, α′, and the `regression_guard` block (`passed: true`, `max_abs_diff: 0.0`).

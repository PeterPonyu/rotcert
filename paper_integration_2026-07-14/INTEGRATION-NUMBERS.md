# Config-A → paper integration digest (2026-07-14)

Verified by direct extraction from the artifact JSONs (paths below). Every number in the
manuscript edits MUST come from this file or from the frozen parent results already in the paper.
Source artifacts: `configA_cert_2026-07-14/<cell>/{audit_*.json,r20_coverage.json,recall.json,cert_gwd.json}`,
`configA_regime_extend_2026-07-14/results.json`, exclusion memos in `configA_pulled_2026-07-14/dota_zoo/`
and `configA_cert_2026-07-14/zoo_gliding_vertex/`.

## 1. Four NEW certified cells (Config-A box run, pulled + certified locally 2026-07-13/14)

All at α=0.10 (nominal coverage 0.90), GWD score, same frozen pipeline as the paper's four cells.

### G1 audited marginal coverage (v1, eval split; point [CI], n TP / n scenes)

| cell | GWD coverage | naive-coord coverage | n TP | n scenes |
|---|---|---|---|---|
| Oriented R-CNN · HRSC2016-MS | 0.9011 [0.8798, 0.9201] | 0.9212 [0.9000, 0.9418] | 799 | 314 |
| RTMDet-R · HRSC2016-MS | 0.9015 [0.8760, 0.9227] | 0.9179 [0.8937, 0.9405] | 670 | 310 |
| RoI Transformer · DOTA-v1.0 | 0.9004 [0.8232, 0.9447] | 0.9279 [0.8846, 0.9554] | 4246 | 70 |
| Oriented RepPoints · DOTA-v1.0 | 0.9003 [0.8545, 0.9299] | 0.9329 [0.9082, 0.9559] | 5617 | 205 |

### R=20 out-of-sample recalibration (marginal coverage across 20 reseeds)

| cell | mean | min | max |
|---|---|---|---|
| Oriented R-CNN · HRSC | 0.8989 | 0.8528 | 0.9500 |
| RTMDet-R · HRSC | 0.9085 | 0.8659 | 0.9568 |
| RoI Transformer · DOTA | 0.8882 | 0.7431 | 0.9736 |
| Oriented RepPoints · DOTA | 0.8988 | 0.7818 | 0.9720 |

(hrsc_orcnn BCa95 from cert log: [0.8884, 0.9096].)

### G2 recall (LTT-HB, β=0.2, δ=0.05): REFUSED in all four cells — by design

Power-floor refusal (Bentkus floor infinite at these n): HRSC n_img=440 (r_hat 0.5464 orcnn /
0.6076 rtmdet); DOTA zoo n_img=1093 (r_hat 0.9380 roi_trans / 0.8496 reppoints). The pipeline
refuses rather than reporting an uncertifiable number (design §2.4/§5 K5). Report as refusals,
NOT failures — this is the paper's refusal-is-a-feature thesis operating.

### Per-stratum G1 refusals (certifiability floor α_min = 1/(n_cal+1) > 0.1)

Oriented RepPoints · DOTA only (6 Mondrian strata): large-vehicle (n_cal=3, α_min=0.25),
soccer-ball-field (n_cal=4, α_min=0.20). All other new cells: zero refusals (HRSC has 1 stratum: ship).

## 2. Exclusions (disclose)

- **S2A-Net** — checkpoint-load failure (mmrotate v0.1.0-era ckpt incompatible with dev-1.x);
  memo `configA_pulled_2026-07-14/dota_zoo/S2ANET-EXCLUDED-2026-07-14.md`.
- **Gliding Vertex** — same class of failure; memo
  `configA_cert_2026-07-14/zoo_gliding_vertex/GLIDINGVERTEX-EXCLUDED-2026-07-14.md`.
- Net zoo breadth: planned 6 DOTA detectors → realized 4 (RTMDet-R, ORCNN + RoI-Trans, RepPoints).

## 3. Coverage-matched efficiency, new cells (POST-HOC, coverage-fair, NOT confirmatory)

Ratio = A_GWD / A_baseline at matched realized coverage (<1 ⇒ GWD smaller). 20 splits unless noted.

### vs naive-coord (pooled `all` regime)

| cell | ratio (mean) | splits GWD smaller | eval n/split |
|---|---|---|---|
| Oriented R-CNN · HRSC | **1.2413** | 7/20 | 322 |
| RTMDet-R · HRSC | **1.0198** | 7/20 | 269 |
| RoI Transformer · DOTA | **1.6618** | 2/20 | 1740 |
| Oriented RepPoints · DOTA | **0.7056** | 16/20 | 2222 |

Frozen parent cells for context: DOTA (RTMDet/ORCNN) 0.929/0.931 (wash), DIOR-R 0.791/0.825 (win).

### vs hull: GWD larger everywhere — 1.30–2.44× (HRSC 1.82/1.30; zoo 2.44/1.43). Unchanged conclusion.

### THE REGISTERED HRSC PREDICTION IS REVERSED (verdict block in results.json)

Registered (coverage_matched_regime_2026-07-13/RESULTS.md §"The HRSC prediction"): HRSC pooled
ratio vs naive should land in the low-0.7s or below (most decisive GWD win). Observed: **1.2413
and 1.0198 (mean 1.1305)** — GWD's matched-coverage region is LARGER than naive-coord's on the
most-elongated dataset. Both detectors ≥1.0 individually. The elongation mechanism does NOT
transfer between datasets; the zoo adds detector-dependence in the same direction (1.66 vs 0.71
on the same dataset+eval set). What survives: the *within-cell* elongation gradient on the four
frozen cells (4/4 direction under both cuts) — as a descriptive, post-hoc observation only.

## 4. Population caveats (must accompany any cross-cell comparison)

- HRSC eval: AR mean 5.63/5.95, frac AR≥3 = 0.90/0.93 — essentially no compact stratum
  (compact regime rows SKIPPED: no split met MIN_CAL/MIN_EVAL).
- DOTA zoo cells run on a 1093-image DOTA subset; matched-TP populations are far smaller and
  markedly LESS elongated (AR mean 1.27 roi_trans / 1.47 reppoints; frac AR≥3 ≈ 0) than the
  frozen DOTA cells (AR mean ≈ 2.7, frac AR≥3 ≈ 0.29–0.30). Zoo ratios are therefore NOT
  directly comparable to the frozen DOTA cells' pooled ratios; they add detector-breadth
  evidence, not a third elongation point.

## 5. What this means for the manuscript (framing, for user review)

1. **Correctness/coverage story STRENGTHENED:** 4 → 8 healthy G1 cells, now spanning 3 datasets
   and 4 distinct detector architectures (2-stage ORCNN, dynamic-label RTMDet-R, RoI-Trans,
   point-based RepPoints; six were planned, two excluded), with principled
   refusals where n is too small (G2 power floor; 2 tiny strata) — the wrapper claim is now
   evidenced.
2. **Efficiency story: disclose the reversal.** The registered falsifiable prediction failed in
   reverse. The efficiency claim must be scoped to: descriptive, within-cell,
   dataset-and-detector-dependent; GWD never beats the hull. No mechanistic elongation claim.
3. HRSC disclosed as HRSC2016-MS (multi-scale variant); trained in-house (Oriented R-CNN 1x,
   RTMDet-R 3x) mirroring DIOR configs.

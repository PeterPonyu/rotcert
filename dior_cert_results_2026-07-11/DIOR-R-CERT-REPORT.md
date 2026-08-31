# DIOR-R + DOTA certification report (consolidated, final) — 2026-07-11

CPU certification mirroring the DOTA pilot protocol (α=0.10, β=0.20, δ=0.05, rotated-IoU τ=0.5, per-class
Mondrian). **Primary G1 metric is out-of-sample R=20 scene-split coverage** (in-sample `audit.json` is
secondary/diagnostic — see the correction below). Prereg-gated GWD-vs-naive **Holm-8 NOT run**; no AOPG.
Frozen inputs read-only. Suite: 218 passed, 1 skipped — no regression (only result files added).

## G1 correction: in-sample was tautological; out-of-sample R=20 is the real number
The pilot orchestration fed the **same** `matched.jsonl` to both `calibrate` and `audit`, so the reported
"marginal coverage 0.9000" is the split-conformal **calibration-set** coverage
($\lceil(1-\alpha)(n+1)\rceil/n$ = 0.90002 for **any** score) — tautological, not held-out validity. All G1
numbers below are the correct **out-of-sample R=20** (fit $\hat q$ on 40% cal scenes, measure on held-out 40%
eval scenes, 20 repeats, BCa CI). Full trail: `pilot_oos_2026-07-11/PILOT-OOS-NOTE.md`.

## Four-cell G1 (out-of-sample R=20, BCa 95% CI; nominal 0.90)
| Cell | detector · dataset | **OOS R=20 coverage** | in-sample (diagnostic) | verdict |
|---|---|---|---|---|
| DOTA pilot | RTMDet-R-l · DOTA-v1.0 | **0.8879** [0.8774, 0.8979] (min 0.843) | 0.9000 | ~1.2 pt **below** nominal |
| DIOR-ORCNN (orig GT) | Oriented R-CNN · DIOR-R | **0.9001** [0.8985, 0.9019] | 0.9000 | at nominal |
| DIOR-ORCNN (fixed GT) | Oriented R-CNN · DIOR-R | **0.9004** [0.8987, 0.9020] | 0.9000 | at nominal |
| DIOR-RTMDet | RTMDet-R-l · DIOR-R | **0.9012** [0.8995, 0.9030] | 0.9000 | at nominal |

**Reading:** the GWD certificate holds **at** nominal out-of-sample on all three DIOR-R cells (single-tile
images → clean scene=image exchangeability) and **~1.2 pt below** on the DOTA pilot only. The **same detector
(RTMDet-R-l)** covers at nominal on DIOR (0.901) but below on DOTA (0.888) — so the DOTA shortfall is a
**dataset artifact** (overlapping 1024-px crops leave residual within-scene correlation), not a detector
artifact. The in-sample 0.9000 hid this dataset-dependence entirely.

## Mondrian per-class (α=0.10)
| Cell | certified / total | refused | notes |
|---|---|---|---|
| DOTA pilot | 15/15 | 0 | — |
| DIOR-ORCNN (orig GT) | **18/20** | 0 | 2 expressway classes **excluded** (0 TP) — substrate casing bug |
| DIOR-ORCNN (fixed GT) | **20/20** | 0 | both expressway classes **recovered** after GT casing fix |
| DIOR-RTMDet (fixed GT) | **20/20** | 0 | all 20 classes present |

**Casing fix verified (zero drift).** Original DIOR GT capitalized `Expressway-Service-area` /
`Expressway-toll-station` while dets lowercased them → class-exact matcher gave 0 TP → excluded. The
regenerated GT lowercases them. Re-running ORCNN vs fixed GT recovers both (`expressway-service-area` q̂=59.76
n=962; `expressway-toll-station` q̂=8.94 n=508) and leaves the **other 18 classes bit-identical** — every q̂
and n_cal matches the first run to **0.00%** (verified class-by-class). Surgical fix; OOS coverage held
(0.9001→0.9004).

## G2 certified recall (β=0.20, δ=0.05)
| Cell | certified | classes |
|---|---|---|
| DOTA pilot | 2/15 | small-vehicle, tennis-court |
| DIOR-ORCNN (orig GT) | 4/20 | basketballcourt, tenniscourt, stadium, groundtrackfield |
| DIOR-ORCNN (fixed GT) | **5/20** | + expressway-service-area (recovered) |
| DIOR-RTMDet (fixed GT) | **6/20** | baseballfield, basketballcourt, tenniscourt, storagetank, groundtrackfield, expressway-service-area |
Refusals are honest: some classes have an **infinite** LTT-HB power floor (realized miss rate ≥ β), others
fall below the Bentkus image floor. DIOR-R certifies more than DOTA (more images/class); RTMDet (denser dets,
more TPs) certifies more than ORCNN.

## naive-coord conditional coverage (review item C2 premise — DESCRIPTIVE only, no Holm)
The paper's premise: naive coordinate-wise (Bonferroni-box) CP breaks at the angle seam / square regime.
Angle-regime audit (`audit_naive.json`, nominal 0.90):

| Cell | naive marginal | naive interior | naive **boundary** | naive **square** | GWD (int/bnd/sq) |
|---|---|---|---|---|---|
| **DOTA pilot** | 0.9331 (over) | 0.9505 | 0.9194 | **0.8405** [0.801, 0.864] | 0.893 / 0.937 / 0.880 |
| **DIOR-ORCNN fixed** | 0.9355 (over) | 0.9440 | **0.8976** [0.892, 0.902] | 0.9615 | 0.886 / 0.910 / 0.933 |
| **DIOR-RTMDet fixed** | 0.9352 (over) | 0.9446 | 0.9121 [0.906, 0.917] | 0.9398 | 0.885 / 0.917 / 0.927 |

**Verdict (honest, cell-dependent).** naive **over-covers marginally everywhere** (0.933–0.936, Bonferroni
conservatism). Its coverage is **regime-dependent** — the seam (boundary) or square is always its worst or
most-depressed regime relative to interior — but whether it actually drops **below nominal** varies by cell:
- **DOTA square: 0.8405** — clearly below nominal (CI excludes 0.90), an 11-pt deficit vs its 0.9505 interior.
  Strong support for the square-degeneracy premise.
- **DIOR-ORCNN boundary: 0.8976** — at/just below nominal (straddles 0.90). Moderate support (seam).
- **DIOR-RTMDet boundary: 0.9121** — its worst regime but still **above** nominal; naive over-covers at every
  regime here. **No** under-coverage.

So the C2 premise ("naive breaks at seam/square") is **supported as a regime-conditional claim on DOTA and
(weakly) DIOR-ORCNN, but NOT on DIOR-RTMDet** — a hostile reviewer would note naive over-covers everywhere on
RTMDet-DIOR. The rigorous test is the prereg-gated Holm-8 (not run). GWD is seam/square-stable in every cell
(no regime below ~0.88).

## Cross-comparisons
**Same detector, two datasets (RTMDet-R-l):** DOTA OOS 0.8879 [0.877, 0.898], Mondrian 15/15, G2 2/15  →
DIOR OOS **0.9012** [0.900, 0.903], Mondrian **20/20**, G2 **6/20**. The certificate is *better* on DIOR
(at nominal, more G2 certs) — the DOTA shortfall is crop-driven, not detector-driven.
**Two detectors, one dataset (DIOR-R, fixed GT):** ORCNN OOS 0.9004 / RTMDet 0.9012 (both at nominal);
Mondrian 20/20 both; G2 5 vs 6. The certificate is **score/detector-agnostic** — nearly identical G1 across a
two-stage (ORCNN) and a single-stage (RTMDet) detector.

## τ sensitivity (DIOR-ORCNN orig GT; matched-TP conditioning not fragile)
τ=0.5/0.6/0.7 → in-sample coverage 0.9000 at all (τ-robust); mean disk area 6,514 → 4,409 → 2,729 px²
(tightens as matching strictens: higher-IoU TPs = smaller localization error). τ sweep for RTMDet/fixed-GT is
descriptive and not re-run (the DIOR-ORCNN sweep already establishes the τ-robustness conclusion).

## Files (`dior_cert_results_2026-07-11/`)
`orcnn/` (orig-GT), `orcnn_fixedgt/`, `rtmdet/` — each: matched, cert_{gwd,naive-coord,hull}, recall,
audit_gwd/audit_naive, r20_coverage; `r20_generic.py`. `pilot_oos_2026-07-11/`: dota_r20_coverage.json,
dota_audit_naive.json, PILOT-OOS-NOTE.md. Never touched `pulled/` or the DOTA pilot result dirs.
Note: the RTMDet match was re-run once after an early concurrent-writer corruption (multiple relaunches
across session drops wrote the same file); the final single-writer match is byte-consistent (tp=98,690,
deterministic vs the box's own run) and all RTMDet numbers above are from that clean file.

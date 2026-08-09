# DIOR-R per-class Mondrian conditional-coverage analysis (GWD G1) — 2026-07-15

**POST-HOC / DESCRIPTIVE.** Reports the per-class conditional coverage of the GWD
localization certificate on the two core DIOR-R cells (Oriented R-CNN fixed-GT, RTMDet-R-l
fixed-GT), mirroring the paper's Mondrian machinery (`mondrian_field="class"`) and the
out-of-sample R=20 scene-split protocol of `dior_cert_results_2026-07-11/r20_generic.py`.
Not a preregistered confirmatory family; it characterizes conditional (per-class) behavior
to preempt a "marginal-only" reviewer question.

## Protocol
- Inputs: frozen `matched.jsonl` (matched true positives), `cert_gwd.json` (full-data per-class
  q̂ / n_cal), `recall.json` (frozen G2 recall-certificate statuses) — all read-only, from
  `dior_cert_results_2026-07-11/{orcnn_fixedgt,rtmdet}/`.
- Score: GWD. α = 0.10 (nominal coverage 0.90). Mondrian stratum = object class (20 DIOR classes).
- Splits: 20 repeated scene-level splits (40% calibration / 20% match-holdout / 40% eval),
  split-seed = repeat index; scene = image for DIOR (single-tile).
- Per repeat: `g1_calibrate(cal, "gwd", α=0.10, mondrian_field="class")` then
  `g1_coverage(cert, eval)` → per-class coverage on held-out eval scenes. Aggregated per class
  across the 20 repeats (mean / std / min; mean held-out n).
- Code: `perclass_conditional_coverage.py`; numbers: `results.json`.

## Headline
Every one of the 20 DIOR classes clears the G1 certifiability floor (α_min = 1/(n_cal+1) < α;
smallest class dam, n_cal = 266/284 → α_min ≈ 0.0037 ≪ 0.10) — **0 classes refused at G1** in
both cells. Per-class out-of-sample conditional coverage is tightly clustered around nominal:
**no class deviates more than ≈1.3 points from 0.90 in either cell.**

| Cell | overall OOS cov. | per-class range | classes ≥ nominal | classes refused (G1 floor) | recall-cert refused |
|---|---|---|---|---|---|
| DIOR-ORCNN (fixed GT) | 0.9003 | 0.8866 (golffield) – 0.9127 (dam) | 11/20 | 0/20 | 15/20 |
| DIOR-RTMDet (fixed GT) | 0.8998 | 0.8914 (stadium) – 0.9074 (tenniscourt) | 5/20 | 0/20 | 14/20 |

(Overall OOS here pools the **per-class Mondrian** certificate and is distinct from the paper's
**marginal** R=20 OOS numbers — 0.9004 ORCNN / 0.9012 RTMDet — which fit a single global q̂.)

## Reading
1. **Conditional coverage holds, not just marginal.** The GWD Mondrian certificate delivers
   per-class coverage within ≈1.3 pt of 0.90 for all 20 classes on both a two-stage (ORCNN) and
   a single-stage (RTMDet) detector. The classes that dip slightly below nominal are within one
   split-to-split standard deviation of 0.90 (e.g. ORCNN golffield 0.8866 ± 0.041, RTMDet stadium
   0.8914 ± 0.027) — descriptive scatter, not a structural conditional-coverage failure.
2. **The certifiability floor bites on recall (G2), not localization (G1).** All 20 classes are
   localization-certifiable at G1. The small classes that ARE refused are refused by the G2
   recall certificate's LTT-HB power floor (airplane, chimney, dam, golffield, airport,
   trainstation, … — 15/20 ORCNN, 14/20 RTMDet), whose floor is images-per-class-driven. This is
   the honest "which small classes are refused" answer: the geometry certificate covers every
   class conditionally; the recall certificate honestly abstains on the rare ones.

## Per-class table (OOS conditional coverage, sorted ascending by ORCNN coverage)
See `results.json` for full numbers (per-class mean/std/min coverage, mean held-out n, full-data
n_cal, q̂, G1 floor status, recall-cert status). Representative rows:

**DIOR-ORCNN (fixed GT)** — worst/best: golffield 0.8866 (n_cal 459), dam 0.9127 (n_cal 266);
largest classes ship 0.8991 (n_cal 30 731), storagetank 0.9003 (n_cal 16 081), vehicle 0.9040
(n_cal 12 958).
**DIOR-RTMDet (fixed GT)** — worst/best: stadium 0.8914 (n_cal 576), tenniscourt 0.9074
(n_cal 6 577); largest classes ship 0.8991 (n_cal 32 112), storagetank 0.8995 (n_cal 18 963),
vehicle 0.8994 (n_cal 16 704).

## Provenance
- Frozen inputs untouched (read-only). Suite unaffected (only new result files added).
- All numbers above trace to `results.json` (written by `perclass_conditional_coverage.py`).
- Env: `/home/zeyufu/miniconda3/envs/dl/bin/python`,
  `PYTHONPATH=reliability-commons:reliability-commons/tools/rotcert`.

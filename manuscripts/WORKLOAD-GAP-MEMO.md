# Workload-gap memo — rotcert (oriented-detection certification) vs IEEE TGRS

**Author:** write-geo (agent). **Date:** 2026-07-11. **Scope:** analysis only — no experiments run,
no paper edits. Assesses the gap between the current pilot content and the IEEE TGRS empirical bar for an
oriented-detection reliability-method paper, with a costed gap list, gating notes, and the reviewer-2 attack
surface.

## 1. Current inventory
- **DOTA-v1.0 pilot (complete):** GWD v1 certificate, **1 detector** (RTMDet-R-l), val, **R=1** scene split.
  G1 marginal coverage 0.900 [0.879, 0.916] (n=54,131 dets, 455 scenes, α=0.10) + angle-regime coverage
  (interior/boundary/square) + **15-class Mondrian, 0 refused**. G2: **2/15** classes certified at
  (β=0.20, δ=0.05), the rest honestly refused (LTT-HB power floor). Efficiency vs naive-coord/hull is
  **descriptive only** (GWD region smaller on 11/15 classes; no significance test yet).
- **DIOR-R substrate — landed 2026-07-11 (today):** ORCNN detection substrate
  (`dior_test_dets_orcnn.jsonl`, ~160,610 dets over 11,723 test images) + GT (`dior_test_gt.jsonl`,
  ~124,445 boxes) + a preliminary `orcnn_coverage.txt`; the RTMDet-R DIOR inference marker is present
  (second detector finishing). **This is detection substrate, not certification** — running
  `rotcert calibrate/recall/audit` on it is the next CPU step and remains a `\todo` in the paper.
- **Skeleton `\todo` slots (24):** DIOR-R G1/G2 for both detectors + the confirmatory head-to-head Holm-8.

## 2. TGRS empirical norm for oriented-detection / reliability-method papers
Method papers standardly evaluate on the **DOTA + DIOR-R + HRSC2016** trio (2 minimum, 3 typical),
report **multiple detectors**, and include **ablation studies**. GWD is TGRS-familiar territory
(e.g. OrientedFormer, TGRS 2024, uses Gaussian-Wasserstein self-attention), which helps position the
conformalization as the contribution. TGRS's bar is **materially heavier than JSTARS**: a 1-detector,
1-dataset, single-split pilot reads as workshop-scale here.

## 3. Gap list (cost · MUST / STRENGTHENER / OPTIONAL · gating)
| Gap | Cost | Class | Notes |
|---|---|---|---|
| **DIOR-R certification** (run calibrate/recall/audit on the landed substrate) | CPU-only, ~hours (substrate already on disk) | **MUST-HAVE** | Turns the single-dataset pilot into the 2-dataset, multi-detector claim TGRS expects. **Unblocked now** for ORCNN; RTMDet-R substrate finishing. Top priority. |
| **R=20 scene-split repeats** (currently R=1) | CPU-only, cheap | **MUST-HAVE** | The frozen/1-seed detectors have no training variance; the paper's own preregistered design mandates R=20 split repeats as the honest variance source. R=1 gives no CIs on the head-to-head. |
| **Confirmatory head-to-head Holm-8** (GWD vs naive-coord/hull, coverage + set-size, significance) | CPU-only (on cached scores) once R>1 lands | **MUST-HAVE** | The paper's *premise* (naive CP breaks near the seam / inflates sets) is currently only descriptive. TGRS will require the confirmatory test. This is the load-bearing novelty claim (skeleton Phase-2). |
| **Oriented R-CNN on DOTA** (2nd detector on dataset 1) | 1 val-inference GPU pass + CPU cert | **MUST-HAVE (for the 2×2 grid)** | The Holm-8 family is 2 baselines × 2 detectors × 2 datasets = 8; DIOR-R already adds ORCNN, so this completes the detector×dataset grid on DOTA. |
| **HRSC2016 third dataset** | GPU (checkpoint/train + inference) + CPU cert | **STRENGTHENER (top)** | Completes the canonical TGRS trio and stress-tests the certificate on a single-class, very different object distribution (ships, extreme aspect ratios — a good test of the square-safety / seam claims). |
| **IoU-match / score-threshold sensitivity** (τ∈{0.5,0.6,0.7}) | CPU-only | STRENGTHENER | Preregistered exploratory; shows the matched-TP conditioning is not fragile. |
| **More detectors (R3Det / S2A-Net)** | GPU per detector | OPTIONAL | Score-agnosticism breadth; diminishing returns beyond RTMDet-R + Oriented R-CNN. |

**Gating (do NOT presume):** the **AOPG-table reproduction arm is preregistration-gated** (K3 mAP sanity
check — *not* required for any certification claim); the **full-grid Holm-8 is Phase-2** and needs the
R=20 × 2-detector × 2-dataset grid assembled first. Both are called out in the skeleton and stay gated.

## 4. Reviewer-2 attack surface (TGRS's heavier bar)
- **"Single dataset, single detector, R=1"** — **FATAL as-is** for TGRS. The DIOR-R certification + R=20
  repeats + Oriented R-CNN + the confirmatory Holm-8 are exactly the difference between "pilot" and "TGRS
  paper." Encouragingly, **most of this is CPU on already-landed / soon-landing substrate**, plus one or two
  val-inference GPU passes.
- **"No confirmatory head-to-head"** — the premise is only descriptive today; TGRS will demand the Holm-8
  significance test that naive CP measurably under-covers the seam/square strata and inflates sets. MUST.
- **"Only 2 datasets (no HRSC2016)"** — MEDIUM. DOTA + DIOR-R is acceptable; the trio is the norm. HRSC2016
  is the top strengthener and a genuinely informative stress test (ship aspect ratios).
- **"GWD is a re-used loss, not a contribution"** — mitigated by crisp positioning (the contribution is the
  *conformalization* + the two guarantees + the head-to-head falsification), which the skeleton already does;
  GWD's TGRS familiarity (OrientedFormer) actually helps here.
- **Net verdict:** TGRS is reachable but the MUST list is non-trivial — however it is **mostly CPU on the
  DIOR-R substrate that landed today** plus R=20 repeats and one ORCNN-on-DOTA inference pass. Sequence:
  (1) DIOR-R cert on the landed ORCNN + RTMDet-R substrate, (2) R=20 repeats, (3) ORCNN-on-DOTA pass,
  (4) confirmatory Holm-8, then (5) HRSC2016 as the top strengthener toward the full trio. AOPG repro and the
  full Phase-2 grid stay prereg-gated.

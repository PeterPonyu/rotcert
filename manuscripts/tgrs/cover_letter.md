# Cover letter — IEEE Transactions on Geoscience and Remote Sensing (TGRS)

Date: TODO-USER

Dear Editor-in-Chief and Associate Editors,

We submit our manuscript, **"Angle-Aware Conformal Certification for Oriented Object Detection:
Out-of-Sample Localization Coverage and Certified Recall on DOTA and DIOR-R,"** for consideration
in the IEEE Transactions on Geoscience and Remote Sensing.

Oriented (rotated) object detectors are the workhorse of aerial and remote-sensing image analysis,
yet a deployed detector emits a point prediction with no distribution-free statement about how far
the true box may lie from it, nor about how many objects it silently misses. This paper supplies
both, as a reliability layer on the exact detectors and benchmarks the TGRS readership uses. It is a
methods-and-audit contribution — not a new detector and not a mapping product — that certifies
recognized oriented detectors (RTMDet-R-l, Oriented R-CNN) on DOTA-v1.0 and DIOR-R.

The technical core is that porting conformal prediction to oriented boxes is not mechanical: the box
angle is periodic with a discontinuity at the ±90° seam and degenerate at square aspect ratios, so a
naive coordinate-wise interval is structurally mis-specified in those regimes. We conformalize a
Gaussian–Wasserstein (GWD) nonconformity score — the angle representation is drawn from the
oriented-detection loss literature but used here as a conformal score, not a loss — that is
continuous across the seam and safe at square boxes. On it we build two distribution-free
guarantees: **G1**, per-detection localization coverage given detection, and **G2**, a
Learn-then-Test (Hoeffding–Bentkus) certified rotated-IoU false-negative-rate bound that refuses
below its statistical power floor. The system ships as `rotcert`, a backbone-agnostic command-line
tool.

The empirical grid is the multi-dataset, multi-detector breadth a TGRS methods paper expects. Across
the full RTMDet-R-l × Oriented R-CNN by DOTA × DIOR-R cross, G1 coverage — measured **out-of-sample**
over R=20 scene-level splits — holds at nominal on both DIOR-R cells (0.900–0.901) and ≈1–1.2 points
below on both DOTA cells (0.888–0.890); both detector families reproduce the split, so the shortfall
is a dataset artifact of DOTA's overlapping 1024-px crops, not a detector or score defect. Per-class
Mondrian certificates cover every class with 0 refused (15/15 DOTA; 20/20 DIOR-R for both detectors).
G2 certifies 2/15 DOTA and 5–6/20 DIOR-R classes, refusing the rest honestly, with the certified
count tracking images-per-class. Against the canonical corner-wise/axis-aligned-hull conformal
baselines, the preregistered Holm-8 family (2 baseline contrasts × 2 detectors × 2 datasets),
executed exactly as frozen, finds the GWD region smaller in all 8 of 8 cells at nominal α; we report this
as a preregistered descriptive outcome rather than statistical confirmation, and — because that contrast
is at matched nominal α, where the union-bound baselines over-cover — characterize efficiency primarily
through a post-hoc, coverage-fair coverage-matched ablation (see disclosure (ii) below).

Consistent with the paper's reliability emphasis, the manuscript discloses its methodological limits
directly rather than leaving them for a reader to infer. Three disclosures are stated in the paper
itself and we flag them here: (i) an earlier in-sample coverage audit was an algebraic identity that
certifies nothing — we found, corrected, and report the out-of-sample number throughout, keeping the
in-sample value only as a labelled diagnostic; (ii) the frozen single-split Holm-8 family is
**demoted from confirmatory to a preregistered descriptive outcome** — its single-split test is
**degenerate** at the 1/2001 p-floor under pooled calibration (effective n=1), so its substantive
content is the effect size (the region-size ratios), not the p-values, and "8/8 at the p-floor" is
**not** eight independent strong rejections. We therefore do not present the frozen family as
confirmatory and make no preregistration amendment; our primary quantitative characterization of
efficiency is a post-hoc, coverage-fair coverage-matched ablation that re-tunes each baseline to GWD's
realized coverage — it shows the nominal size gap is largely baseline over-coverage, surviving only
against the like-for-like naive baseline on DIOR-R (≈0.8×) and not against the coarser axis-aligned hull;
the R=20 across-split analysis is reported as split-sensitivity of the nominal contrast (GWD smaller on
20/20 splits, worst single-split ratio 0.974) with no p-value, since those splits reshuffle one fixed
scene set and are not independent; (iii) the "naive conformal breaks at the seam/square"
hook is reported as a **conditional**, not universal, claim — the naive baseline under-covers at the
square regime on DOTA but merely over-covers on RTMDet–DIOR-R. We regard this transparency as central
to the paper's rigor.

**Scope match to TGRS.** Oriented object detection on DOTA and DIOR-R is core TGRS territory — DIOR-R
itself was introduced in TGRS — and referees will recognize every detector, dataset, and baseline
without preamble. A distribution-free reliability/uncertainty-quantification layer for these exact
detectors lands squarely in the readership; this is an applied remote-sensing reliability question,
not a new network architecture.

We confirm that this manuscript is original work, is not under consideration or review elsewhere, and
has not been submitted in whole or part to any other venue. The single author has approved the
submission. Data and code availability are as stated in the manuscript's availability statement: the
`rotcert` tool (numpy/scipy/shapely only) and all frozen result JSONs accompany the paper, the CLI is
golden-file-tested against them, and the datasets (DOTA-v1.0, DIOR-R) are public. Two items remain
open and are stated as such in the paper: a full-text differentiation against the concurrent EAV-DETR
system (a named pre-submission action), and the DIOR-R AOPG mAP reproduction, now executed and disclosed
(reproduced test mAP 62.61 / 68.36 vs AOPG 64.41; the AOPG table has no same-method row, so the ±0.5
identity tolerance is not met cross-method; a same-method external anchor of 64.30 for Oriented R-CNN R-50
on DIOR-R is disclosed, within 1.7 points of our 1× reproduction; disclosed, not claimed as passing).

Thank you for your consideration.

Sincerely,

Zeyu Fu
TODO-USER: affiliation line (department, institution, city, country)
e-mail: fuzeyu99@126.com
ORCID: 0009-0001-8329-0108

---
**Suggested reviewers** (TODO-USER: supply three; leave blank if you prefer the editors choose):
1. TODO-USER — expertise: oriented/rotated object detection on aerial imagery (DOTA, DIOR-R; GWD/KLD losses).
2. TODO-USER — expertise: conformal prediction / distribution-free uncertainty quantification and risk control.
3. TODO-USER — expertise: reliability and calibration for remote-sensing deep learning.
Pick reviewers without a recent co-authorship or shared-institution conflict with the author.

**Funding statement:** TODO-USER (state grant/support, or "The author received no specific funding
for this work.").

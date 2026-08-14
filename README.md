# rotcert

Angle-aware conformal certification for oriented object detection in aerial imagery.

Frozen records: Zenodo DOI [10.5281/zenodo.21392293](https://doi.org/10.5281/zenodo.21392293).

## Thesis

Oriented detectors emit a point oriented bounding box (OBB) and say nothing
distribution-free about how far the true box may lie, or how many objects they
silently miss. An OBB angle lives on ℝ/πℤ: a ±90° seam and square-box
unidentifiability make coordinate-wise conformalization geometrically
misspecified, not merely a bit loose.

RotCert conformalizes a **seam-continuous, square-safe Gaussian–Wasserstein
(GWD) nonconformity**. GWD is consumed as a score, not claimed as a new
detector loss.

- **G1** is a Mondrian per-class GWD-ball localization certificate with a
  finite-sample coverage guarantee, validated out-of-sample over R=20 scene
  splits.
- **G2** is a Learn-then-Test Hoeffding–Bentkus certified rotated-IoU
  false-negative-rate bound that **refuses** below its power floor instead of
  emitting a number it cannot back.

The operator-facing object is a per-class certificate card: center-offset
radius (px), orientation half-extent (deg) or unconstrained, rotated-IoU
floor — or an explicit refuse with a planning readout.

## Frozen evidence

Ten G1 cells · three aerial datasets (DIOR-R, DOTA-v1.0, HRSC2016-MS) · five
detector families.

| Object | Frozen record |
| --- | --- |
| G1 on DIOR-R | 20/20 classes certified (four architectures; none refused) |
| G1 on DOTA-v1.0 | overlapping crops are an exchangeability stress case (~1–1.2 points below nominal 0.90) |
| G2 at (β, δ) = (0.20, 0.05) | DOTA pilot 2/15 · DIOR-R S2A-Net 7/20 · HRSC 0/1 |

Coverage-matched compactness is DIOR-R-specific (near parity on DOTA; reverses
on HRSC). Holm-8 size-ratio tests are descriptive.

## Refuse

G1 refuses when the certifiability floor α_min = 1/(n_cal+1) exceeds the
target α. G2 refuses below the LTT–HB power floor (finite floor, or
unreachable at the chosen β). Vacuous and refused are first-class states, not
empty table cells. Splits are scene-level; crop-level splitting is refused.

## Archive

Software v0.2.0, MIT. Cite `CITATION.cff` or the Zenodo record. Package code
is in `rotcert/`. The test suite is synthetic and does not retrain detectors.
Multi-GB detection trees stay local; frozen certification statistics are in
this tree and on Zenodo.

This repository is the certification protocol and its frozen records — not a
detector, not a training stack, and not a data portal.

---
title: Pipeline
kicker: Reproduce
lede: Frozen records live on Zenodo. Splits are scene-level. The public archive is the interface.
---

The public archive issues localization and recall certificates from sealed detector outputs. It does not retrain detectors.

**G1** fits a Mondrian Gaussian–Wasserstein ball (default α = 0.10). **G2** fits a Learn-then-Test recall bound at (β, δ) = (0.20, 0.05). Matching uses rotated IoU ≥ 0.5. Splits are **scene-level**; crop-level splitting is refused.

Frozen result records for the ten G1 cells are archived at [Zenodo 10.5281/zenodo.21392293](https://doi.org/10.5281/zenodo.21392293). The repository test suite is the unit check; it does not retrain detectors.

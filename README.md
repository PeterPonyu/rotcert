# RotCert

RotCert provides angle-aware conformal certification and reliability diagnostics
for oriented object detection. It supports the article *Angle-aware conformal
certification for oriented object detection in optical aerial imagery*.

The published version 0.3.0 contains the numerical package, analysis programs,
reported aggregate results and table inputs, together with historical results
retained from version 0.2.0. This repository's documentation may be updated after
a release; a version DOI and its tagged archive identify the unchanged software
snapshot, not every later manuscript or illustration update.

- Source: [GitHub repository](https://github.com/PeterPonyu/rotcert).
- Version 0.3.0: [10.5281/zenodo.23184005](https://doi.org/10.5281/zenodo.23184005).
- Historical version 0.2.0: [10.5281/zenodo.22211671](https://doi.org/10.5281/zenodo.22211671).
- All versions: [10.5281/zenodo.21392292](https://doi.org/10.5281/zenodo.21392292).

## Scientific scope

Localization coverage targets a matched object drawn uniformly within a new
true-positive-bearing source scene, under the stated source-population and
fixed-design assumptions. The article's proofs assume independent and identically
distributed source scenes; objects within a source may be dependent. Marginal
coverage averages over calibration and a future source. It is not conditional
coverage of each prediction, recall for missed objects, or high-probability risk
control after one realized calibration. Source-uniform calibration, object-weighted
risk control, scene-simultaneous coverage and recall-risk validation have different
targets.

Gaussian Wasserstein distance and the general conformal and risk-control tools are
established methods. The geometric analysis connects the calibrated score event
to center and orientation projections with candidate dimensions free. A smaller
center region may accompany a wider orientation range; no universal superiority
over other scores is claimed. Falling within both projections is not sufficient
for membership in the original score set.

Historical DIOR-R design exposure is not erased by retraining or the validation
check. CODrone validation and test share locations and recordings, and different
training seeds are not independent geographic replications. Native EAV
margin-score coverage and common-HCP analyses use different matched populations;
the score event must not be called rendered-polygon containment. Missed objects,
unsupported targets, infinite thresholds and adverse results remain explicit.
False-positive shares are separate detection readouts, not controlled by the
localization or recall certificates. These results are not a safety-deployment
guarantee.

## Install and test

In a Python environment with the declared dependencies:

```bash
python -m pip install -e '.[test]'
python -m pytest tests -q
```

The numerical package uses NumPy, SciPy and Shapely without importing a GPU
framework. Required helper modules are included with their original MIT license.
See [testing scope](TESTING.md) for analysis-suite commands, external input
requirements and the limits of the software checks.

## Interfaces and evidence

The package provides Gaussian Wasserstein scores, source-uniform hierarchical
calibration, geometric projections, fixed-grid recall validation, and distinct
modular and direct end-to-end routes. Their interfaces are `rotcert.gwd`,
`rotcert.scene`, `rotcert.geometry`, `rotcert.g2`, `rotcert.e2e` and
`rotcert.scores_ext`.

For example, one score vector per source gives a source-uniform threshold:

```python
from rotcert.scene import hcp_threshold
q = hcp_threshold([[0.2, 0.4], [0.3], [0.1, 0.5]], alpha=0.1)
# Too few calibration sources: the result is infinity, not a finite certificate.
```

The [analysis guide](revision_2026-10/README.md) describes the scientific questions
and input requirements. The [aggregate evidence](revision_2026-10/results/README.md)
contains reported values and their file hashes. The command-line interface's
historical object-pooled calibration is not a substitute for the revised
source-uniform analyses. Recall outputs changed in version 0.3.0; use version
0.2.0 when inspecting its historical outputs.

## Reproducibility and availability

The release manifest identifies files in each archived snapshot. Aggregates
permit inspection of reported values; statistical reanalysis additionally needs
the identified source-level inputs. Recompiling an article, redrawing a figure,
reanalysing exported detections and retraining a detector are different levels of
reproducibility. Passing software tests does not establish full experimental
reproduction.

Version 0.3.0 does not contain the later supplementary illustration generators or
their scene and source-role records. Those additions must not be represented as
already available under its DOI. Its new evidence is aggregate data, not dataset
images, annotations, per-object exports, source-level reanalysis inputs or detector
weights. Historical directories retained from version 0.2.0 do contain derived
detection and ground-truth records, including compressed files; they are not new
independent samples.

Dataset and upstream-code terms continue to apply. The MIT license covers the
original software and does not relicense provider data. Underlying research
records may be requested from the corresponding author, subject to those terms
and actual availability. The four DIOR-R validation-check checkpoints are not
included and their availability has not been confirmed; saved outputs and
checkpoint hashes do not substitute for the weight files. No claim is made that
all weights are lost or that replacement training recreates them.

See `CITATION.cff`, `LICENSE` and [archive information](ZENODO.md) for citation and
licensing. No new study, journal submission or publication decision is implied by
this software release or its documentation updates.

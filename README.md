# RotCert

RotCert provides angle-aware conformal certification and reliability diagnostics
for oriented object detection. Version 0.3.0 accompanies the revised article
*Angle-aware conformal certification for oriented object detection in aerial imagery*.
It contains the numerical package, revised analysis programs, accepted aggregate
results and table inputs, and the historical result records already in 0.2.0.

- Source: [github.com/PeterPonyu/rotcert](https://github.com/PeterPonyu/rotcert).
- Version 0.3.0 archive: [10.5281/zenodo.23184005](https://doi.org/10.5281/zenodo.23184005).
- First-submission version 0.2.0: [10.5281/zenodo.22211671](https://doi.org/10.5281/zenodo.22211671).
- All versions: [10.5281/zenodo.21392292](https://doi.org/10.5281/zenodo.21392292).

## Scientific scope

The revised protocol keeps its sampling targets explicit. Localization coverage
concerns a matched-object draw from a new source, under the stated frozen-design
and exchangeability assumptions. It does not imply conditional coverage of every
individual detection, recall for missed objects, or a PAC guarantee for a fixed
realized calibration. Source-level maxima, object-weighted CRC, source-uniform
HCP and recall-risk certificates answer different questions.

GWD is an established Gaussian Wasserstein representation, used here as a
seam-continuous and square-safe nonconformity score. The geometric readouts connect
its calibrated event to center and orientation projections with candidate box
dimensions free. Smaller center regions can accompany broader orientation ranges;
the experiments do not establish universal dominance over other scores.

The empirical restrictions are part of the result: earlier DIOR-R design exposure
is not erased by retraining or the validation-split check; CODrone validation and
test share locations; seed arms are not independent geographic replications.
EAV's native containment and common-HCP analyses use different matched populations.
Missing support, infinite thresholds and unavailable endpoints are retained.

## Install and test

From this checkout, in a Python environment with the declared dependencies:

```bash
python -m pip install -e '.[test]'
python -m pytest tests -q
```

The package needs NumPy, SciPy and Shapely, and does not import a GPU framework.
The four MIT-licensed `relmetrics` modules required by the legacy adapters are
included, unmodified, under `rotcert._vendor.relmetrics`; no private sibling
repository or unpublished PyPI package is needed. The original license is retained.
See `TESTING.md` for exact verification scope and separate analysis-suite commands.

## Revised interfaces

- `rotcert.gwd`: canonical oriented boxes and the GWD score.
- `rotcert.scene`: source-uniform HCP, object pooling, object-weighted CRC,
  count-adaptive CRC and scene-maximum calibration.
- `rotcert.geometry`: free-dimension center and orientation readouts.
- `rotcert.g2`: recall-risk certification on a threshold grid fixed before testing.
- `rotcert.e2e`: the distinct modular and direct end-to-end routes.
- `rotcert.scores_ext`: score alternatives used in the revision.
- [`revision_2026-10/`](revision_2026-10/README.md): analysis programs and their input contracts.
- [`revision_2026-10/results/`](revision_2026-10/results/README.md): aggregate evidence and hash manifest.

For example, a source-uniform threshold uses one score vector per source:

```python
from rotcert.scene import hcp_threshold
q = hcp_threshold([[0.2, 0.4], [0.3], [0.1, 0.5]], alpha=0.1)
# This small calibration sample yields infinity. It is not a finite certificate.
```

The `rotcert` command-line interface and the first-submission records are retained
for historical comparison. G2 outputs changed in 0.3.0: use version 0.2.0 to replay
those historical outputs. The CLI's legacy object-pooled calibration must not be
substituted for the revised source-uniform analysis drivers.

## Reproducibility and availability

`RELEASE-MANIFEST.json` binds the released files. The aggregate-evidence manifest
also records hashes of the original accepted files before public path redaction.
No reported experimental numbers were changed when creating this release.

The newly added `revision_2026-10` evidence contains aggregates, not images,
annotations, per-object exports, per-source replay inputs or detector weights.
Historical first-submission directories retained from 0.2.0 do contain derived
detection, ground-truth and matched-object records, including compressed copies;
they are not new revision inputs or additional independent samples. The MIT
license covers the original software, not a relicensing of provider data.
Dataset files come from their original providers; underlying research records
may be requested from the corresponding author subject to those terms. The EAV
upstream geometry reference is not redistributed. The four DIOR-R validation-split weights had not been locally
recovered at release preparation; their saved outputs and checkpoint identities
are retained, but the weights are not promised as immediately available.

The released aggregates allow inspection of the reported results. Full statistical
replay requires the separately identified inputs and frozen provenance manifests;
passing unit tests is not evidence that those full experiments were rerun from
the public archive. No full manuscript or private review correspondence is published
in this software release. See `CITATION.cff`, `LICENSE`, `ZENODO.md` and
`revision_2026-10/README.md` for citation, license and scope.

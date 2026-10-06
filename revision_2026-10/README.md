# Analysis code of the revised article (version 0.3.0)

This folder holds the programs behind the revised version of *Angle-aware conformal certification for oriented
object detection in aerial imagery*. The `rotcert` package at the repository root implements the certification
methods; the programs here apply them in the analyses of the article. They are the files that produced the reported
results. Release adaptations change dependency/input locations and documentation, without changing the reported
scientific quantities, seeds or numerical definitions. The release manifest and test record identify these adaptations.

Accepted aggregate results and table inputs are public in [`results/`](results/README.md), with source and public-file
hashes. Per-source detection, matching, calibration and evaluation records, trained detectors, and dataset images
and annotations are not included. Underlying records can be requested from the corresponding author, subject to
the terms of DIOR-R, DOTA-v1.0, HRSID, DroneVehicle, CODrone and HRSC2016-MS. Dataset files must be obtained
from their providers. EAV's final weights are preserved locally; the four DIOR-R validation-split weights had not
been recovered locally at release preparation, so their availability is not promised. Saved detections, checkpoint
hashes and aggregate results remain preserved.

## Layout

| Folder | Content | Article |
|---|---|---|
| `execution/s1` | Source-uniform hierarchical calibration, object pooling, source-level bootstrap and the fixed-role sensitivity | Results 3.2 |
| `execution/s2` | Simulation of how object pooling departs from source-uniform coverage | Results 3.2 |
| `execution/s3/v3` | Sixteen nonconformity scores at design-matched operating points, with center and orientation projections | Results 3.3 |
| `execution/s5`, `execution/s5-bootstrap` | Region readouts and their source-resampling intervals across cells | Results 3.4 |
| `execution/g2-planning` | Recall (G2) certificate planning | Results 3.5 |
| `execution/s4/v3` | Statistics kernel of the photometric shift experiment | Results 3.6 |
| `execution/synthesis` | Syntheses of the S1 and S5 outputs | Results 3.2, 3.4 |
| `closeout`, `wpR` | Matching of detections to ground truth and per-cell caches | Methods |
| `reviewer_analyses` | Coverage within strata and scoring cost, false-positive share and recall, DOTA re-tiling, G2 simulation, sampling-unit audit, near-duplicate exclusion | Results 3.2–3.5 |
| `shifts/photometric` | Multicore driver and synthesis of the photometric shift experiment | Results 3.6 |
| `shifts/attenuation` | Synthetic atmospheric attenuation and official-split readouts, including DroneVehicle | Results 3.4, 3.6 |
| `eav_detr_codrone` | Complete EAV-DETR system on CODrone: tiling, export, analysis and detection accuracy | Results 3.3 |
| `validation_split_check` | DIOR-R validation-split check: train-only configurations, cell preparation and synthesis | Results 3.2 |
| `detectors` | Dataset staging, configuration generation, training and inference driver, export of detections | Methods |

## Running

Programs read their inputs below a project root given by `ROTCERT_ROOT` (default: this repository) in the layout
used for the article; the input folders named in the code (for example `revision_2026-10/inputs` and
`revision_2026-10/closeout/analysis`) are part of the records described above. Further variables:

- `EAV_DETR_ROOT`: a checkout of the EAV-DETR code at commit `fb99da7b0f49edffcd1fa9e45a4424e9ae86e1eb`. The
  EAV-DETR geometry module is not redistributed here, because its repository carries no license. The S3 reference
  parity tests can read that module from `ROTCERT_EAV_GEOMETRY`, or from
  `execution/s3/v3/snapshot/eav_geometry_official.py` if supplied separately. The vectorized S3 score itself does
  not require copying the upstream module. Use the pinned commit when checking numerical parity.
- `ROTCERT_SSH_CONFIG`: SSH configuration used only by `validation_split_check/transfer.py`.

Training configurations keep the container paths of the compute host (`/root/autodl-tmp`).

## Tests

`python -m pytest tests` at the repository root runs the package tests. The tests in `execution/s1`, `execution/s2`,
`execution/s5`, `execution/s5-bootstrap` and `eav_detr_codrone/tests` run without research records. S3 needs the
external EAV reference module for two parity tests, and `ROTCERT_S3_CELLS` plus its caches for one identity test.
S4 includes its earlier first-party kernel oracle; two tests additionally require historical records, located via
`ROTCERT_S4_RECORDS_ROOT`. See [`../TESTING.md`](../TESTING.md) for commands, environment versions, exact
pass/expected-failure counts and the explicit failures when those external inputs are absent.

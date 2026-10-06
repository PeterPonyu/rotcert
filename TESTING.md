# Verification of version 0.3.0

These are CPU software tests. They do not assert that all paper experiments can be
replayed using only this archive, validate scientific independence assumptions, or
constitute detector/GPU certification. No training was rerun for the release.

## Verified suites

- Root `tests/`: 350 passed, 2 expected failures. Import of a separately installed
  `relmetrics` was deliberately blocked; the tests used the bundled modules.
  The two expected failures document that angle-residual baselines are not
  square-safe. They are not counted as passes.
- `execution/s1/test_core.py`: 7 passed.
- `execution/s2/test_s2.py`: 14 passed.
- `eav_detr_codrone/tests`: 60 passed.
- `execution/s3/v3/{test_s3.py,test_v2.py}`: 16 passed with the separately held,
  pinned upstream EAV geometry module and the original cell/cache records.
- S4/v3, S5-bootstrap and S5 together: 88 passed, plus 1,185 subtest assertions.
  This comprises 70 S4, 16 bootstrap and 2 S5 test items, using the original S4
  historical inputs. Subtest assertions and overlapping suite runs are not
  additional independent experiments.

Root/S1/S2/EAV used Python 3.13.5, NumPy 2.2.6, SciPy 1.16.3, Shapely 2.1.2,
pytest 8.4.2 and OpenCV 4.13.0. S3/S4/S5 used the existing Python 3.13.5 environment
with NumPy 2.5.1, SciPy 1.18.0 and pytest 9.1.1. OpenMP, BLAS and MKL used one
thread. An earlier root run in the environment without OpenCV produced 349 passes,
one skipped polygon-conversion test and two expected failures; the complete root
run above used the already installed OpenCV and had no skipped items.

## Running the public tests

Run suites in separate processes to respect the frozen analysis modules' import
layout. From the repository root:

```bash
python -m pytest tests -q
python -m pytest revision_2026-10/execution/s1/test_core.py -q
python -m pytest revision_2026-10/execution/s2/test_s2.py -q
python -m pytest revision_2026-10/eav_detr_codrone/tests -q
python -m pytest revision_2026-10/execution/s5-bootstrap revision_2026-10/execution/s5 -q
```

The DIOR polygon-input test requires OpenCV in addition to the core package
dependencies. No GPU framework is required for these suites. The S4 tests use the
pytest subtests interface (pytest 9.1 or an equivalent subtests plugin).

## Tests that require external artifacts

S3 reference parity requires `ROTCERT_EAV_GEOMETRY` pointing to the documented
EAV-DETR `tools/geometry_utils.py`. One S3 identity/RNG test additionally requires
`ROTCERT_S3_CELLS` pointing to the original `cells.json`, whose cache paths must be
available. Without these, three S3 tests fail explicitly; they are not auto-skipped.

S4's old v2 first-party oracle code is included under `execution/s4/v2`. S4's
synthetic fixtures are generated in private temporary directories. Two tests still
require historical prepared arrays, saved thresholds and a recorded failed
checkpoint. Set `ROTCERT_S4_RECORDS_ROOT` to the supplied records root, containing
`cpu-runtime-binding-v1/code/execution/s4/v2/runs/real-input-pilot-v2-budgeted`
and `LOCAL-STATUS-20260929T120155Z/FIRST-FAILURE-SCALAR-REPRODUCTION.json`, with the
original checkpoint and prepared records identified by that receipt. With
no records supplied, the full S4 suite gives 68 passes and two explicit failures,
not an all-pass result. These artifact requirements were not bypassed.

Full-run verifiers also require the original input caches, role assignments,
protocols, source-freeze manifests and run outputs. Path-portable release copies
have different source hashes from the original execution snapshots; do not replace
an execution-freeze hash with the release hash to make a historical verifier pass.
The code archive is accompanied by aggregate evidence, not every replay input.

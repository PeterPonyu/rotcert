# Analyses accompanying the article

The programs support the reported calibration, geometry, recall and detection
analyses. Version 0.3.0 includes analysis code and aggregate evidence; it does not
include all source-level records needed to repeat each study. Later supplementary
illustration scripts and their input records are not part of that archive.

The analyses address source-uniform versus object-pooled calibration, simulations
of unequal object counts, geometric trade-offs across nonconformity scores,
training-run variation, recall-risk planning, false-positive shares, source
grouping and image tiling, synthetic image perturbations, the complete EAV-DETR
comparison on CODrone, and the four-family DIOR-R validation check. These are
different statistical questions, not interchangeable demonstrations of one
universal guarantee.

## Inputs and execution

Programs need detection and annotation records, the specified source grouping,
role assignment, operating thresholds and matching rules. Dataset files should
come from the original providers. Underlying records may be requested from the
corresponding author, subject to provider terms and actual availability.

`ROTCERT_ROOT` selects the data root; program headers state the expected input
layout. Existing package paths and environment-variable interfaces are retained
for compatibility. Compute-host paths in archived training configurations are
deployment settings, not credentials or universally portable defaults; adapt
them to the authorized local environment rather than assuming a particular host.

Numerical parity checks requiring the EAV-DETR geometry reference must use the
identified upstream commit `fb99da7b0f49edffcd1fa9e45a4424e9ae86e1eb`. That reference
is not redistributed because the required permission has not been established.
The vectorized score can be used without copying it. See [testing scope](../TESTING.md)
for the commands and explicit external-input requirements.

## Interpretation and availability

Source grouping keeps related tiles together but does not establish independent
physical acquisition. Historical DIOR-R exposure and shared CODrone locations
and recordings remain limitations. Multiple seeds vary the complete procedure,
including source-role assignment and operating-point selection, rather than
training randomness alone.

Native EAV margin-score coverage is not rendered-polygon containment. Its
matching population differs from that of common-HCP comparisons. Recall and
false-positive shares are separate from TP-conditional localization coverage.
Infinite regions, missing support and non-issuance of a finite report must retain
their distinct meanings.

The four DIOR-R validation-check weight files are not in this release and their
availability has not been confirmed. Saved detections and checkpoint identities
remain available for the documented analysis; they are not the weight files.
No substitute retraining is presented as recovery of the originals. EAV weight
preservation does not imply that every detector checkpoint is included here.

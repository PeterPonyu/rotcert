# Aggregate evidence for the revised article

These files preserve accepted aggregate results, support counts, uncertainty
intervals, and provenance of the revised article. `MANIFEST.json` lists the hash
of each original record and of its public copy. The only transformations are JSON
serialization and replacement of workstation/compute roots by labelled
placeholders; numeric values, missing values, infinity encodings, and acceptance
flags are unchanged. This is an evidence release, not a new analysis.

- `eav-codrone/`: the final full-split EAV-DETR analysis (56,000 validation tiles,
  84,056 test tiles), full-test detection accuracy, and its acceptance checks.
- `dior-validation/`: the four-family frozen validation-split results and primary
  and fixed-role verification summaries.
- `article-tables/`: saved numerical inputs and formatted values used in the
  article's main and supplementary tables. These are transcriptions of accepted
  outputs; no experimental quantities were recalculated for this release.

The EAV detection class-macro recall is the arithmetic mean of each class's
aggregated TP/GT, not a frame-uniform average. Native EAV containment and the
common-HCP score events have different matched populations. CODrone validation
and test data share all 46 locations and 576 recording IDs. Counts refer to
overlapping tile instances, without physical-object deduplication. AP is reconstructed
from exports retained at score >=0.05, which do not fully recover lower-score
candidates in the authors' top-300 output; the comparison does not isolate training
differences. The DIOR-R validation-split check avoids selection
on its outcomes but does not repair prior exposure during method development.
The per-class, aggregate, marginal and PAC targets must not be interchanged.

These summaries are sufficient to inspect reported values, but are not the
per-source inputs needed to rerun the full statistical pipelines. Images,
annotations, per-object detections, per-source matching/calibration records,
detector weights and third-party EAV code are not redistributed here. Obtain
datasets and EAV code from their providers. Requests for underlying research
records should identify the method, cell, split and the manifest hash; availability
and redistribution are subject to the original providers' terms. In particular,
the four DIOR-R validation-split weight files had not been locally recovered when
this release was prepared; the saved outputs and checkpoint hashes are retained.

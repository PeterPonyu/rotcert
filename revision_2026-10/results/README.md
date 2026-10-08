# Aggregate evidence accompanying the article

The records contain reported aggregate results, support counts, uncertainty
intervals and provenance. The file manifest identifies the original records and
their public copies. Public preparation did not recalculate experimental values.
These summaries allow inspection of results, not full statistical reanalysis
without the separate source-level inputs.

The evidence covers complete validation and test exports of EAV-DETR on CODrone
(56,000 and 84,056 tiles), full-test detection accuracy, the four-family DIOR-R
validation check, and numerical inputs to the article's main and supplementary
tables. Historical results retain their original interpretation and are not
additional independent samples.

The EAV class-macro recall is the mean of class-level TP/GT ratios, not a
frame-uniform average. Native EAV margin-score coverage and common-HCP score
events use different matched populations. Counts are overlapping tile instances,
not deduplicated physical objects. CODrone validation and test share 46 locations
and 576 recording identities. AP uses score-truncated exports and does not
recover every lower-score candidate of the original system. The DIOR-R
validation check avoids selection on its outputs but does not remove historical
method-development exposure. Marginal, source-level, object-weighted and PAC
targets must not be interchanged.

Dataset images, annotations, per-object exports, source-level calibration and
matching inputs, trained weights and third-party reference code are not supplied
as new revision evidence. Historical derived records have a different scope and
remain subject to provider terms. Requests for research records should identify
the dataset, detector, split, method and result-file hash. The four DIOR-R
validation-check parameters are now publicly downloadable, together with the
final evaluated EAV-DETR parameters, in
the separate [model collection](https://huggingface.co/PeterPonyu/rotcert-models). Its [manifest](https://huggingface.co/PeterPonyu/rotcert-models/blob/a6303b7ab4effbe6cdde6757bd350bb9e5a802d7/MANIFEST.json)
provides the file hashes and parameter-equivalence records. These later model
files do not change the earlier archive or supply every missing analysis input.

Later supplementary illustration generators and their scene/source-role records
are not present in version 0.3.0. Do not treat the software archive as a complete
redrawing package for subsequent manuscript versions. Software checks, file
hashes and reported coverage values do not prove the independence assumptions,
safe deployment or publication acceptance.

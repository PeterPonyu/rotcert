# Zenodo deposition

- Current version DOI: [10.5281/zenodo.23184005](https://doi.org/10.5281/zenodo.23184005)
  (Zenodo record 23184005, version 0.3.0, published 2026-10-06). It archives this repository at tag `v0.3.0`:
  the `rotcert` package, the analysis code and accepted aggregate evidence of the revised article in
  `revision_2026-10/`, and the frozen first-submission result records already present in 0.2.0.
- Concept DOI: [10.5281/zenodo.21392292](https://doi.org/10.5281/zenodo.21392292)
  (parent record 21392292; resolves to whichever version is newest).
- Earlier versions: 0.2.0 (record 22211671, published 2026-08-31), the archive of the first submission; and
  0.1.0 (record 21392293, published 2026-07-16), superseded by 0.2.0 because its archive carried internal
  planning and review documents.
- `CITATION.cff` tracks the published record, so the repository never advertises a release that no one can
  retrieve.
- The new `revision_2026-10` evidence contains aggregate records with a hash manifest, not its per-source detection,
  matching, calibration and evaluation inputs or trained detectors. Historical directories retained from 0.2.0
  do contain derived detections, ground-truth and matched-object records; provider terms still apply to those
  records, and the software MIT license does not relicense them. Requests for underlying revision records are subject
  to provider terms; the four DIOR-R validation-split weight files had not been locally recovered at preparation.
  See `revision_2026-10/README.md` for the precise boundary.
- The 0.2.0 tarball is deposited in ten 5 MB parts. Reassemble with
  `cat rotcert_zenodo_v0.2.0.tar.gz.part* > rotcert_zenodo_v0.2.0.tar.gz`; the
  result is sha256 `7f361f415422bcb892bee5543a752b8aa6abfde335f556aa60d6d8568b629d7c`.
- The source repository is [github.com/PeterPonyu/rotcert](https://github.com/PeterPonyu/rotcert).
- Full article sources and private review correspondence are not tracked on this branch; historical standalone
  figure sources and rendered figures are retained.

# Zenodo deposition

- Public version DOI: [10.5281/zenodo.22211671](https://doi.org/10.5281/zenodo.22211671)
  (Zenodo record 22211671, version 0.2.0, published 2026-08-31).
  The manuscript cites this version DOI on purpose, so a reader always lands on
  the exact archive that produced the reported numbers.
- Concept DOI: [10.5281/zenodo.21392292](https://doi.org/10.5281/zenodo.21392292)
  (parent record 21392292; resolves to whichever version is newest).
- Superseded: version 0.1.0 (record 21392293, published 2026-07-16). Its archive
  carried internal planning and review documents and a port of the manuscript to
  another venue's template; 0.2.0 removes them. Cite 0.2.0 or the concept DOI.
- `CITATION.cff` tracks the published record, so the repository never advertises
  a release that no one can retrieve.
- The 0.2.0 tarball is deposited in ten 5 MB parts. Reassemble with
  `cat rotcert_zenodo_v0.2.0.tar.gz.part* > rotcert_zenodo_v0.2.0.tar.gz`; the
  result is sha256 `7f361f415422bcb892bee5543a752b8aa6abfde335f556aa60d6d8568b629d7c`.
- The public record archives the frozen certification result records referenced
  by the manuscript's `% source:` comments and the `rotcert` tool snapshot.
- The source repository is [github.com/PeterPonyu/rotcert](https://github.com/PeterPonyu/rotcert).
- Manuscript sources are not tracked on this branch. This repository carries the
  software, the frozen result records, and the figure-generation scripts, which is
  what the paper's data-availability statement points to.

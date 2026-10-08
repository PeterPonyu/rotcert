# Archived software and evidence

Version 0.3.0 was published on October 6, 2026 at
[10.5281/zenodo.23184005](https://doi.org/10.5281/zenodo.23184005). It identifies the
software snapshot tagged `v0.3.0`, its analysis programs, aggregate evidence and
historical result records. Later documentation or manuscript changes do not alter
the contents of this versioned archive.

The concept DOI [10.5281/zenodo.21392292](https://doi.org/10.5281/zenodo.21392292)
groups the software versions. Version 0.2.0, published on August 31, 2026, is
available at [10.5281/zenodo.22211671](https://doi.org/10.5281/zenodo.22211671).
Earlier versions are historical snapshots, not the current revised analysis.
Use a version DOI when citing a specific set of files.

The release includes aggregates and table inputs for the revised analysis, but
not its dataset images, source-level reanalysis inputs or trained weights.
Historical derived detection and ground-truth records remain subject to provider
terms. The software license does not grant new rights to redistribute dataset
content. See the [evidence guide](revision_2026-10/results/README.md) for the
availability boundary. Later supplementary illustration generators and their
scene/source-role records are not included in version 0.3.0.

To reassemble the version 0.3.0 archive, download all its numbered parts and
`SHA256SUMS.txt`, then run:

```bash
cat rotcert-0.3.0.tar.gz.part* > rotcert-0.3.0.tar.gz
sha256sum -c SHA256SUMS.txt
tar -xzf rotcert-0.3.0.tar.gz
```

Use these filenames only for version 0.3.0. Older archives have different
filenames and checksums. A successful checksum verifies file identity, not the
scientific assumptions, reproduction of a complete experiment or publication
acceptance.

The [source repository](https://github.com/PeterPonyu/rotcert) may receive
documentation corrections independently of an immutable release. `CITATION.cff`
and the version DOI identify the released software, while the current repository
guide explains its scope and known limitations.

## Separately released evaluated models

The [public model collection](https://huggingface.co/PeterPonyu/rotcert-models) supplies five saved evaluated models
and their portable configurations and parameter-verification records. Its
[fixed repository revision](https://huggingface.co/PeterPonyu/rotcert-models/tree/a6303b7ab4effbe6cdde6757bd350bb9e5a802d7) identifies the added files.
They are separate from the unchanged version 0.3.0 software archive. Do not
cite the earlier DOI as if it already contained these model parameters.

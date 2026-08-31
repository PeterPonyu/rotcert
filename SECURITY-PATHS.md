# Path hygiene (portal tip)

This repository's **current tip** is maintained without developer workstation absolute paths
(`/home/<user>/…`, `Desktop/…` layouts) or local tooling state directories.

Experiment scripts resolve external roots via environment variables when needed:

- `COMMONS_ROOT` / `RELIABILITY_COMMONS` — optional sibling reliability-commons checkout
- `DATA_ROOT` — local scientific data (defaults to `~/data` when unset)
- `AUTODL_TMP` / `CONDA_ROOT` — optional compute-environment overrides

**Author name, email, and ORCID in `CITATION.cff` remain intentional.**

**Git history** may still contain older absolute paths from pre-scrub commits. Tip-only
hygiene is the peer-facing guarantee; full history rewrite is out of scope unless requested.

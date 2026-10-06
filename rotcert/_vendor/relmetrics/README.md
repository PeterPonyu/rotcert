# Preserved relmetrics dependency

The four modules in this directory are copied byte-for-byte from the author's
MIT-licensed reliability-commons worktree used to test RotCert: `bootstrap.py`,
`conformal.py`, `multiplicity.py` and `provenance.py`. The upstream repository HEAD
was `d1e76a001cd6e0808ab6d5c1971655da874bbf88`; the release file manifest records
the actual copied file hashes and is authoritative for their identity.

The upstream repository and distribution were not publicly retrievable when
version 0.3.0 was prepared. Preserving only the required modules makes the public
RotCert package usable without access to that repository. Numerical code is
unchanged. Imports use the private `rotcert._vendor.relmetrics` namespace. Other
reliability-commons tools, data, figures and projects are not included.

Copyright (c) 2026 PeterPonyu. The original MIT license is included as `LICENSE`.

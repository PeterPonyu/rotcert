---
title: Pipeline
kicker: Reproduce
lede: Public interface is the rotcert command family. Splits are scene-level. Frozen records live on Zenodo.
---

Install the public `rotcert` package (NumPy / SciPy / Shapely; no detector training stack). The commands are:

<pre class="cli">rotcert ingest
rotcert match
rotcert calibrate
rotcert recall
rotcert certify
rotcert audit
rotcert report</pre>

`calibrate` fits G1 (Mondrian GWD, default α=0.10). `recall` fits G2 at (β, δ) = (0.20, 0.05). `match` uses rotated IoU ≥ 0.5. Splits are **scene-level**; crop-level splitting is refused.

Frozen result records for the ten G1 cells are archived at [Zenodo 10.5281/zenodo.21392293](https://doi.org/10.5281/zenodo.21392293). The repository test suite is the unit check; it does not retrain detectors.

Do not point the package at private orchestration scripts. Those are not the public interface.

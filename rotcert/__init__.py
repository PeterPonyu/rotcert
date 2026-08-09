"""rotcert: angle-aware GWD-based conformal certification for oriented object detection.

See ``apps-design/05-APP-rotdet-cert.md`` for the full design; ``README.md`` in this
directory for the quickstart and the deviations-from-design-doc register.

Modules
-------
gwd       Gaussian-Wasserstein-distance nonconformity: OBB->(mu,Sigma), the closed-form
          2x2 Bures term, le90 canonicalization. THE paper's centerpiece.
sets      G1 coverage sets: the GWD-ball certificate + its conservative per-parameter
          envelope (reporting-only).
matching  Rotated-IoU / hull-IoU (shapely) + the preregistered greedy matching rule.
splits    Scene-level (never crop-level) 3-way repeated calibration/matching/eval splits.
scores    The six nonconformity constructions (gwd, naive-coord, hull, wrapped-coord,
          doubled, iou) through one calibrate/cover/set-size interface.
ltt       Learn-then-Test (Hoeffding-Bentkus / empirical-Bernstein) for the G2 certified
          image-level FNR, plus the a-priori LTT-HB power floor.
certify   G1 (per-Mondrian-cell split conformal) + G2 (certified FNR) + the honest-
          uncertainty refusal rules.
audit     Scene-clustered bootstrap coverage CIs, the confirmatory Holm-8, K1
          premise-death.
io        Canonical detections/GT/matched JSONL schemas.
cli       ``rotcert {ingest,match,calibrate,recall,certify,audit,report}``.
"""

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]

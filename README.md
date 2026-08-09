# rotcert

**Oriented-detection certificate reliability research theme** (public theme repo).
Contains package code, frozen experiment statistics needed to rebuild the
manuscript, figure SSOT, and venue kits (Pattern Recognition + TGRS fallback).

Multi-GB train/pull trees stay local-only (see `.gitignore`);
public archive: Zenodo DOI [10.5281/zenodo.21392293](https://doi.org/10.5281/zenodo.21392293).

Portfolio layout: sibling of `reliability-commons/`; commons path
`tools/rotcert` is a symlink here. Audit kits via `papers/rotcert-*`.

Angle-aware GWD-based conformal certification for oriented object detection (OBB) on
DOTA + DIOR-R. Given a set of oriented detections and ground truth, `rotcert` fits
**G1** — per-detection localization coverage regions with a distribution-free marginal
coverage guarantee — and **G2** — a confidence threshold with a certified rotated-IoU
recall/false-negative-rate bound — on a nonconformity score that is continuous across
the box-angle's +-90 degree seam and safe at square aspect ratios, then runs the
head-to-head audit against naive coordinate-wise CP baselines that the paper's whole
premise rests on. A thin wrapper over
[`relmetrics`](../../relmetrics) — see
`apps-design/05-APP-rotdet-cert.md` for the full design spec this package implements.

## Quickstart

```bash
# From reliability-commons/tools/rotcert:
python3 -m venv .venv && source .venv/bin/activate
pip install -e ../../                 # relmetrics (editable, from reliability-commons root)
pip install -e .                      # rotcert itself (numpy/scipy/shapely only — no torch/mmrotate)
pip install -e '.[test]'              # + pytest

python -m pytest                      # 205 tests, ~40s, no network/GPU/mmrotate
```

```bash
# Canonical detections/GT JSONL in, at every step (image_id, scene_id, class,
# obb=[cx,cy,w,h,theta], [score]).
rotcert ingest    --detector jsonl --data raw_dets.jsonl -o dets.jsonl
rotcert match     --dets dets.jsonl --gt gt.jsonl --iou-thr 0.5 -o matched.jsonl
rotcert calibrate --matched matched.jsonl --score gwd --alpha 0.10 --mondrian -o cert.json
rotcert recall    --matched matched.jsonl --beta 0.20 --delta 0.05 --mondrian -o recall.json
rotcert certify   --cert cert.json --dets new_dets.jsonl -o regions.jsonl
rotcert audit     --matched matched.jsonl --score gwd --alpha 0.10 -o audit.json
rotcert report    --cert cert.json --audit audit.json -o report.md
```

Full box-side pilot: `orchestration/next_boot_rotcert.sh` (see below).

## Honest-uncertainty rules (design §3.3)

| Situation | Tool behavior |
|---|---|
| stratum's certifiability floor `alpha_min = 1/(n_cal+1) > alpha` | G1 **refuses**: prints the floor, emits no certificate for that stratum (`certify.g1_calibrate`'s `refused` list) |
| G2 class-stratum below the LTT-HB power floor | **refuses** the per-class FNR certificate; falls back to pooled-marginal FNR (`certify.g2_certify_fnr_mondrian`) |
| class absent from calibration / out-of-support | G1 coverage reporting excludes-and-counts (`n_out_of_support`), never silently pools |
| no scene ids resolvable in a detections/GT table | `splits.assert_scene_level_splits` **refuses** — never silently falls back to crop-level splitting |
| `certify` applied to a non-`gwd` cert | **refuses** — only GWD has a well-defined ball/envelope; the Bonferroni-box constructions have no per-box "region" reporting path in this build |

## Module notes

- `gwd.py` — the paper's centerpiece: OBB->(mu,Sigma) under le90 canonicalization,
  the closed-form 2x2 Bures term, seam continuity + square safety. Read this before
  touching anything else.
- `sets.py` — the GWD-ball certificate + its conservative per-parameter envelope
  (reporting-only — the ball carries the guarantee, never the envelope's widths).
- `matching.py` — rotated/hull IoU (shapely) + the preregistered greedy match rule.
- `splits.py` — scene-level (never crop-level) 3-way repeated splits.
- `scores.py` — the six nonconformity constructions (gwd, naive-coord, hull,
  wrapped-coord, doubled, iou) through one calibrate/cover/set-size interface.
- `ltt.py` — Learn-then-Test (HB/EB) for the G2 certified image-level FNR, plus the
  a-priori LTT-HB power floor (design §2.4's exact arithmetic).
- `certify.py` — G1 + G2 + the refusal rules.
- `audit.py` — scene-clustered bootstrap coverage CIs, the confirmatory Holm-8, K1
  premise-death.
- `io.py` — canonical detections/GT/matched JSONL schemas.
- `cli.py` — `rotcert {ingest,match,calibrate,recall,certify,audit,report}`.

## Deviations from the design doc's literal wording (and why)

1. **The LTT-HB math lives in `rotcert/ltt.py`, not `relmetrics`.** The design says
   "REUSE for LTT/HB/Holm," but `relmetrics` does not yet have an LTT module (only
   `conformal`/`bootstrap`/`multiplicity`/`nulls`/`aurc`/`provenance`) — the same
   situation `asr-gate` hit, whose `asr_gate/ltt.py` module docstring says outright
   "NOT in relmetrics yet." `rotcert/ltt.py` mirrors that module's validated HB/EB
   p-value math (same construction, same validity proofs) but is adapted for a
   PER-IMAGE risk matrix rather than per-row accept/reject (`asr-gate`'s G1 gates
   whole utterances by their own score; `rotcert`'s G2 risk is a continuous function
   of the confidence threshold across MULTIPLE detections per image at once — see
   `ltt.py`'s module docstring for the full reasoning). Promoting a shared risk-matrix
   LTT primitive into `relmetrics` (mirroring `ope-audit`'s CRC-upstreaming precedent)
   is a natural next step once a second tool needs the same shape.
2. **`hull_iou >= rotated_iou` is documented as an empirical/aggregate claim, not a
   per-pair inequality.** The design's phrasing ("hull-IoU >= rotated-IoU => optimistic
   recall") reads as a universal bound; a trivial counterexample (an axis-aligned box
   against a 45-degree-rotated box of the same size) falsifies it per-pair. `matching.
   hull_iou`'s docstring documents this precisely and points to `certify.py`'s G2
   anti-conservatism diagnostic as the place this gets checked EMPIRICALLY (aggregate
   recall, hull match vs rotated match) rather than assumed.
3. **The envelope (`sets.shape_envelope`) is a JOINT grid search over `(w,h,theta)`,
   not three independent 1-D scans holding the other two parameters fixed at the
   prediction.** The design's literal phrase is "conservative per-parameter envelope
   via 1-D scans." A true single-axis scan (varying only `w`, say, with `h`/`theta`
   pinned to the prediction) UNDER-covers the ball's true projection whenever the
   optimal off-axis combination extends further — i.e. it would violate the
   ball-subseteq-envelope property the design itself requires ("Reader warning" in
   §2.3). The implemented joint grid search is genuinely conservative (verified by
   `tests/test_sets.py::TestBallSubsetEnvelope`, rejection-sampling ball members and
   checking every one falls inside the reported envelope, including the seam-wrap and
   near-square full-arc cases) at the cost of being a numerical approximation rather
   than a closed form.
4. **B1 (naive-coord), B2 (hull), A1 (wrapped-coord), A2 (doubled) all use LITERAL
   per-coordinate Bonferroni** (K separate split-conformal intervals, each at level
   `alpha/K`) rather than a single max-normalized scalar with one split-conformal
   threshold (which gives EXACT joint coverage with no correction needed, and is
   arguably the more standard multivariate-conformal technique). The max-normalized
   construction would eliminate exactly the failure mode C2 needs to demonstrate (a
   naive angle interval that either misses wrapped GT or absorbs wraparound outliers
   into an inflated Bonferroni-corrected quantile) — see `scores.py`'s module
   docstring for the full argument.
5. **The confirmatory-8 permutation test is a class-blocked SIGN-FLIP permutation on
   the log set-size ratio, not `relmetrics.nulls.matched_abstention_null`.** The
   design's literal phrase ("within-class matched-abstention permutation p-value")
   borrows terminology from the selective-risk-deferral context that function was
   built for; a set-size RATIO contrast has no "abstention" concept to match. `audit.
   set_size_contrast`'s docstring documents this as a deliberate, stated adaptation —
   the class-blocked bootstrap CI (design's other stated construction) IS
   `relmetrics.bootstrap.blocked_bootstrap` verbatim, block = class.
6. **Set-size (the C2 efficiency metric) is measured as the AREA of each
   construction's `(cx,cy)` coverage slice at the predicted shape held fixed** — a
   disk (`pi*q_hat^2`) for `gwd`, a box (`4*q_cx*q_cy`) for the four Bonferroni-box
   constructions, undefined (`None`, exploratory-only) for `iou` (no closed form).
   This keeps the confirmatory Holm-8 (`gwd` vs `naive-coord`/`hull` only) on a
   closed-form, apples-to-apples quantity; a full 5-D "set volume" comparison across
   fundamentally different set SHAPES (a ball vs a box vs an implicit IoU-level-set)
   has no single natural definition, and the design's own §2.3 explicitly limits the
   envelope's role to reporting, never certification.
7. **`orchestration/{score_rtmdet.py,fetch_dior_r.py,phase0.py}` are box-side
   skeletons with lazy `mmrotate`/`mmcv`/`mmdet` imports**, not runnable in this local
   build (no GPU, no mmrotate installed, per the portfolio's standing "core never
   imports mmrotate" rule — see `SOTA-REPRODUCTION-PLAN-2026-07-10.md` §2). Each
   script's `--help` runs in ANY environment (verified); the actual RTMDet-R-l/
   Oriented R-CNN forward passes are a Phase-0 box task. `rotcert audit`'s
   `--holm-cells` flag accepts a PRE-ASSEMBLED roster of contrast cells (one per
   `(baseline, detector, dataset)`) rather than running the full 8-cell family
   end-to-end from raw detections — assembling that roster from real detector runs is
   `next_boot_rotcert.sh`'s Phase-2 job (not yet implemented there; the pilot stage
   through `rotcert audit --score gwd` on one detector x one dataset is).
8. **DOTA class-name list in `next_boot_rotcert.sh` (`DOTA_CLASS_NAMES`) is typed
   in from the published 15-class table**, not fetched from a config at runtime
   (matches `pc3-cert`'s "transcribed from memory/documentation, not a fetched copy"
   precedent for label maps) — VERIFY against the pinned mmrotate commit's actual
   class-index ordering at Phase 0; getting the class-index alignment wrong would
   silently mislabel every detection.

## Box-side pilot (design §7 Phase 1)

Runs on the AutoDL box (mmrotate + a GPU for RTMDet-R-l inference; everything
downstream is CPU-only, per design §4.7's "one GPU pass, then CPU forever" shape).

```bash
export DOTA_SRC=/root/autodl-pub/DOTA
export MMROTATE_COMMIT=<pinned-sha>          # REQUIRED, no default
export RTMDET_R_CONFIG=<path-to-vendored-config>
export RTMDET_R_CHECKPOINT=<path-to-zoo-checkpoint>
export PUBLISHED_VAL_MAP=<zoo-consensus-val-map>   # VERIFY at Phase 0
export MEASURED_VAL_MAP=<from-your-external-mAP-eval-step>

pip install mmcv mmdet mmrotate      # box-side only; NOT rotcert dependencies
pip install -e .                     # this package (numpy/scipy/shapely only)

bash orchestration/next_boot_rotcert.sh
```

Marker: `ROTCERT_PILOT_ALL_DONE` (or `ROTCERT_PILOT_PARTIAL` with the list of failed
stage markers). The FULL grid (2 detectors x 2 datasets x R=20 repeats x 6 scores,
design §4.3) is gated behind `REQUIRES_PREREG_FREEZE=confirmed` and is a structure-only
skeleton pending the design's Phase-2 prereg freeze (§8).

## Box-side DIOR-R in-house training (design §4.1 A2 addendum, 2026-07-10)

The 2026-07-10 direction survey found **no license-clean DIOR-R-trained checkpoint
anywhere** (the mmrotate zoo is DOTA/HRSC-only; AOPG publishes tables but no weights;
LSKNet is CC-BY-NC). So the DIOR-R arm is **inference + in-house training**: both
detectors are trained on DIOR-R trainval via Apache-2.0 mmrotate dev-1.x (~45-60 GPU-h;
revised total budget ~90-110 GPU-h). `orchestration/next_boot_rotcert_dior_train.sh`
stands that pipeline up on the box.

```bash
# HARD Phase-0 gate: a human reads the DIOR-R terms (incl. the derived-weights rehost
# clause) and touches the license-review marker before ANY training runs.
touch /root/autodl-tmp/dior_r/DIOR_R_LICENSE_REVIEWED       # only after a real review

export MMROTATE_COMMIT=<pinned-sha>                         # REQUIRED (training carries the pins now)
export ORCNN_DIOR_CONFIG=<vendored-orcnn-r50-dior-config>   # REQUIRED for the orcnn stage
export RTMDET_R_DIOR_CONFIG=<vendored-rtmdet-r-l-dior-config>  # REQUIRED for the rtmdet stage
# Optional named-default overrides: ORCNN_EPOCHS=12 ORCNN_BATCH=2 ORCNN_SEEDS=0
#                                   RTMDET_R_EPOCHS=36 RTMDET_R_BATCH=8 RTMDET_R_SEEDS=0
# Seed policy (A2 OPEN prereg decision): ORCNN_SEEDS/RTMDET_R_SEEDS default "0" (1-seed);
# set "0,1,2" for the 3-seed policy once frozen -- no code change needed.

bash orchestration/next_boot_rotcert_dior_train.sh
```

Gate order (each content-asserting): (1) DIOR-R **staged-data content gate** — counts
vs the staged layout (`>=23k` OBB xmls + `>=11k`/`>=11k` trainval/test jpgs + ImageSets),
floors from env, no literals; (2) **license-review** hard-refuse unless the marker file
exists; (3) **mm-stack import probe** (imports, never pip exit codes); (4) Oriented R-CNN
R-50 1x and (5) RTMDet-R-l 3x training, each per-seed with a **checkpoint-integrity**
(torch-zip CRC) gate; (6) **AOPG-table reproduction gate** (K3 target = the AOPG DIOR-R
table, `jbwang1997/AOPG`, Apache-2.0), guarded on `REQUIRES_PREREG_FREEZE=confirmed`
with `DIOR_R_REPRO_TOL` (default 0.5 mAP). Marker: `ROTCERT_DIOR_TRAIN_ALL_DONE` (or
`ROTCERT_DIOR_TRAIN_PARTIAL`). Per-minute nvidia-smi logging is provided by
`chain_prologue`'s boxkit gpu_util logger.

## Testing

```bash
python -m pytest -v
```

205 tests, synthetic data throughout (no network/GPU/mmrotate):
`test_gwd.py` (38 tests — the exhaustive property suite: seam continuity, le90
canonicalization, square isotropy, w/h-exchange invariance, metric-axiom spot checks,
hand-computed 2x2 Bures cases), `test_sets.py` (ball-subseteq-envelope across
non-square/seam-adjacent/near-square cases), `test_matching.py`, `test_splits.py`
(including the DOTA crop-filename scene-id convention), `test_scores.py` (including
the seam-pathology comparison naive-coord vs wrapped-coord that motivates the whole
paper), `test_ltt.py` (including the power-floor arithmetic matching the design's own
worked examples), `test_certify.py`, `test_audit.py` (including K1 premise-death),
`test_io.py` (including the scene-level-discipline refusal), and `test_cli_e2e.py`
(full `ingest -> match -> calibrate -> recall -> certify -> audit -> report` pipeline
via subprocess).

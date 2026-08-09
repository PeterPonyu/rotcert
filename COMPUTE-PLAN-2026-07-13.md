# rotcert COMPUTE PLAN — raising the certified-oriented-detection study to a TGRS-scale grid

**Author:** lever-rotcert (research agent). **Date:** 2026-07-13. **Scope:** strategy +
feasibility + turnkey box recipe. **Constraint honored:** READ-ONLY research; no box was booted,
no `boxkit_api.py` run, no ssh, no spend. Everything below is *stageable*; the USER starts the box.

Source of record for the current paper state: `manuscripts/paper.tex` (four cells, OOS R=20,
coverage-matched efficiency), `manuscripts/{VENUE-FIT,WORKLOAD-GAP,RED-TEAM}*.md`, `DATA_MANIFEST.md`,
`coverage_matched_2026-07-13/RESULTS.md`, `dior_cert_results_2026-07-11/DIOR-R-CERT-REPORT.md`, and
`docs/records/HRSC2016-SOURCING-2026-07-13.md`.

---

## 0. Where the paper is now (the baseline this plan raises)

- **2 datasets × 2 detectors = 4 cells**, all certified end-to-end: DOTA-v1.0 (RTMDet-R-l frozen zoo
  ckpt; Oriented R-CNN frozen zoo ckpt) and DIOR-R (RTMDet-R-l + Oriented R-CNN **trained in house**,
  single seed). Each cell has: G1 out-of-sample R=20 scene-split coverage, per-class Mondrian
  certificates, G2 LTT-HB certified FNR, naive/hull regime audits, and the post-hoc **coverage-matched**
  efficiency ablation.
- **The honest weak points a TGRS reviewer will name** (from `RED-TEAM-REPORT.md` + `WORKLOAD-GAP-MEMO.md`):
  1. **Only 2 datasets** — the canonical oriented-detection trio is DOTA + DIOR-R + **HRSC2016** (MEDIUM).
  2. **Only 2 detectors** — "is the certification detector-agnostic, or an RTMDet/ORCNN artifact?"
  3. **Efficiency was honestly walked back** — the nominal 0.33–0.86× region-size headline was mostly
     baseline over-coverage; at *matched* realized coverage GWD's advantage survives only vs the
     like-for-like naive baseline **on DIOR-R** (~0.79–0.83×), is a **wash on DOTA**, and **loses to the
     hull** everywhere (`coverage_matched_2026-07-13/RESULTS.md`).
  4. **Single checkpoint per detector** — no detector-training-seed variance (only split variance).

The paper survives on its *correctness* contributions (seam-continuity, square-safety, coupled G1/G2)
even with the efficiency walk-back. This plan's job is to move it from *"2 datasets, 2 detectors,
efficiency walked back"* to *"the canonical 3-dataset trio, detector-agnostic across a modern zoo, with a
mechanistic coverage-matched efficiency result"* — and to do it where the compute is cheap.

---

## 1. The one structural fact that drives the whole cost model

**On DOTA-v1.0, adding a detector is inference-only. On DIOR-R and HRSC2016, every detector must be
trained.** All the named detectors (Oriented R-CNN, RTMDet-R, RoI Transformer, ReDet, S2A-Net, R3Det,
Oriented RepPoints, Gliding Vertex, …) ship **released Apache-2.0 DOTA-v1.0 checkpoints in the mmrotate
dev-1.x model zoo** — so on DOTA each new detector is a single val-inference pass (~0.5 GPU-h) plus
CPU certification. But **no license-clean DIOR-R or HRSC checkpoint exists for any detector** (established
for DIOR-R in `DATA_MANIFEST.md`; equally true for HRSC beyond the zoo's own HRSC weights), so every
detector on those two datasets carries a full training run.

Consequence: **detector breadth is cheap on DOTA and expensive on DIOR-R/HRSC.** The cost-optimal grid
therefore concentrates detector breadth on DOTA (inference), buys dataset breadth via HRSC (tiny → cheap
to train), and keeps DIOR-R training to the two detectors already done — expanding DIOR-R only if a
reviewer forces it.

`score_rtmdet.py` is **already fully detector-agnostic** (`--config <cfg> --checkpoint <ckpt>
--mmrotate-commit … --images-dir … --class-names … -o …`), so no new inference code is needed for any
mmrotate model — only the vendored config + the checkpoint URL.

---

## 2. Recommended dataset × detector matrix

### Config A — **RECOMMENDED** ("canonical trio + cheap detector breadth"). ~5–8 new GPU-h, ~1 box-day, ~$3–5.

| Dataset | Detectors | How | New GPU-h |
|---|---|---|---|
| **DOTA-v1.0** | RTMDet-R-l, Oriented R-CNN *(have)* **+ RoI Transformer, S2A-Net, Oriented RepPoints, Gliding Vertex** | released Apache-2.0 zoo ckpts → **inference only** | ~2–3 |
| **DIOR-R** | RTMDet-R-l, Oriented R-CNN *(have — no new work)* | in-house ckpts already trained | 0 |
| **HRSC2016(-MS)** | RTMDet-R-l, Oriented R-CNN | **train + infer** (scaffolding already written, §4) | ~2 |
| **all cells** | coverage-matched + regime-conditional efficiency, Holm family | **CPU, local** (add cell paths) | 0 |

Grid: **DOTA ×6 detectors, DIOR-R ×2, HRSC ×2 = 10 detector-cells across the canonical 3 datasets.**

**Why this changes the contribution class.** It converts every one of the three named weaknesses into a
strength at minimal cost: (1) **HRSC completes the canonical trio** — the exact "2 datasets" objection is
gone, and HRSC is the *ideal* stress test for the paper's core claim because its objects are ships with
**extreme aspect ratios** (the seam/square regime where a seam-continuous, square-safe score should matter
most). (2) **Six detectors on DOTA spanning two-stage (ORCNN, RoI Transformer), single-stage-refine
(S2A-Net), point-based (Oriented RepPoints), quad (Gliding Vertex), and dynamic-label (RTMDet-R)**
demonstrates the certification is a *wrapper around any oriented detector*, not an RTMDet/ORCNN artifact —
the score-agnosticism claim becomes evidenced rather than asserted. (3) The coverage-matched study run
across the enlarged grid (below) is where the efficiency story is *repaired mechanistically*, not by
volume.

### Config B — **STRONGER** (add DIOR-R detector depth + cheap multi-seed). ~25–30 new GPU-h, ~2–3 box-days, ~$12–15.
Config A **plus**: train **RoI Transformer + S2A-Net on DIOR-R** (~8 GPU-h each → cross-family breadth on
the *hard* dataset, not just the free one) and run **3-seed HRSC training** for both detectors (~3 GPU-h
total, tiny dataset → the cheapest possible detector-training-variance demonstration). Use only if a
reviewer pushes on "detector-agnostic *and* on the harder benchmarks" or explicitly demands training-seed
variance.

### Config C — completionist (full zoo × full trio × multi-seed everywhere). ~80–120 GPU-h, ~$40–60, 5–7 box-days.
**Not recommended.** Every additional DIOR-R/HRSC detector is a fresh training run; returns diminish
sharply past Config A/B and the marginal reviewer is not won by cell #11–#30.

**Bottom line:** do **Config A** now. It is the qualitative jump. The single highest-leverage item inside
it — the coverage-matched + regime-conditional efficiency analysis (§5) — is **pure CPU and free**, and is
what turns "efficiency walked back" into a defensible, mechanistic result.

---

## 3. Datasets — status and the HRSC recipe (verified)

- **DOTA-v1.0** — box-side public mirror `/root/autodl-pub/DOTA` (~30 GB, zero download); val tiles
  already the substrate for the two existing DOTA cells. New DOTA detectors reuse the identical val
  crops → the certification is directly comparable across detectors (same scenes, same matcher).
- **DIOR-R** — staged on the box data disk (`/root/autodl-tmp/dior_r`, 23,463 imgs); in-house
  checkpoints + detection substrate already pulled (`dior_infer_results_2026-07-11/`,
  `dior_train_results_2026-07-10/`). No new dataset work.
- **HRSC2016 — the one real data gap, and it is nearly closed.** The sourcing doc's **gdown Route 1 is
  verified** (`docs/records/HRSC2016-SOURCING-2026-07-13.md`): public Google-Drive file id
  `1UslulCCx8GoTflm1gpfIGZeXIsCAdMG5` → `HRSC2016-MS.zip` (2.3 GB), no login, credential-free
  (`pip install gdown && gdown <id> -O HRSC2016-MS.zip`). **Recipe cross-checked against the code that
  consumes it** (`orchestration/prepare_hrsc.py`, `orchestration/hrsc_run.sh`): the converter parses the
  HRSC2016-MS `<robndbox>` VOC XML (angle in radians) → DOTA 8-point-poly annfiles, with a **hard
  angle-convention self-check** (median corner-HBB-vs-XML-HBB IoU ≥ 0.65 or it *refuses before any GPU*;
  correct convention scores ~0.81, a w/h transpose ~0.35). Uses the **canonical `source` split**
  (train 436 / test ~453), single-class `ship`. **Disclosure obligation:** it is the multi-scale
  **HRSC2016-MS** variant — the paper must state HRSC2016-MS, not the vanilla split.
- **4th dataset?** Not needed for TGRS and not recommended. FAIR1M / SODA-A / DOTA-v2.0 each add a full
  training burden per detector for marginal breadth; DOTA-v2.0 is additionally *barred as an independent
  arm* (superset of v1.0, `DATA_MANIFEST.md`). The canonical **trio is the target**; stop there.

---

## 4. Detectors — configs, checkpoints, train-vs-infer, harness reuse

### 4.1 What already exists (no code to write)
- **DIOR-R configs** with the two disclosed stability fixes baked in: `configs_dior/oriented-rcnn-le90_r50_fpn_1x_dior.py`
  and `configs_dior/rotated_rtmdet_l-3x-dior.py` — **AdamW** (lr 1e-4, wd 0.05; SGD NaNs on DIOR-R) +
  **`mmdet.FilterAnnotations(min_gt_bbox_wh=(1e-2,1e-2))`** after Resize (drops the 2 zero-area DIOR-R
  GT boxes that deterministically NaN the loss). These are the fixes the manuscript discloses.
- **HRSC configs — already written and carrying the same fixes**: `configs_hrsc/oriented-rcnn-le90_r50_fpn_1x_hrsc.py`
  and `configs_hrsc/rotated-rtmdet_l-3x-hrsc.py` (num_classes→1, DOTA-format dataset, AdamW +
  FilterAnnotations, RTMDet SyncBN→BN + EMA-only hooks for single-GPU).
- **HRSC converter + box runner — already written**: `orchestration/prepare_hrsc.py` (CPU, numpy+cv2+PIL,
  refuse-gate) and `orchestration/hrsc_run.sh` (convert → deploy configs → prefetch backbones → smoke
  (1-epoch finite-loss + ckpt-integrity gate) → full train → infer test → GT prep → tar; content gates
  throughout, disk-frugal `max_keep_ckpts=1`). **This is turnkey** — HRSC needs no new box code.
- **Generic inference**: `orchestration/score_rtmdet.py` (detector-agnostic, lazy mmrotate import,
  le90-canonicalizes every box, stamps `--mmrotate-commit` provenance).
- **Certifier CLI** (`rotcert/`, numpy/scipy/shapely only, never imports mmrotate): subcommands
  `ingest · match · calibrate · recall · certify · audit · report`. Note on the task's "route": there is
  **no `route` subcommand** — the *routing/refusal* behavior is built into `calibrate` (Mondrian
  per-class refusal below the `1/(n_cal+1)` floor), `recall` (LTT-HB power-floor refusal), and `certify`
  (per-box "stratum not calibrated" refusal). The per-cell certification flow that produced the DIOR
  cells is: `match` → `calibrate --mondrian` (`cert_gwd.json`) → `recall --mondrian` (`recall.json`) →
  `audit` for gwd + naive regime tables → `r20_generic.py` for the OOS R=20 coverage headline.

### 4.2 mmrotate dev-1.x zoo — every named detector is present (Apache-2.0)
Config dirs confirmed on `open-mmlab/mmrotate@dev-1.x`: `oriented_rcnn`, `rotated_rtmdet`, **`roi_trans`**,
**`redet`**, **`s2anet`**, **`r3det`**, **`oriented_reppoints`**, **`gliding_vertex`**, `rotated_retinanet`,
`cfa`, `kfiou`, `gwd`, `kld`, `h2rbox`, `h2rbox_v2`, `convnext`, `csl`, `psc`, plus rotated
faster-rcnn/fcos/atss/reppoints, sasm_reppoints. Each has released DOTA-v1.0 checkpoints in the zoo.

| Detector | Family | DOTA | DIOR-R | HRSC | Notes |
|---|---|---|---|---|---|
| **Oriented R-CNN** | two-stage | released ckpt (**have**) | trained (**have**) | train (cfg ready) | anchor two-stage |
| **RTMDet-R-l** | one-stage dynamic-label | released ckpt (**have**) | trained (**have**) | train (cfg ready) | anchor one-stage |
| **RoI Transformer** | two-stage | released ckpt → **infer only** | train (Config B) | train (Config B) | classic strong two-stage |
| **S2A-Net** | one-stage feature-align | released ckpt → **infer only** | train (Config B) | train (Config B) | different one-stage design |
| **Oriented RepPoints** | point-set | released ckpt → **infer only** | — | — | non-box representation → best score-agnosticism test |
| **Gliding Vertex** | quad-offset | released ckpt → **infer only** | — | — | quad-regression angle-free head |
| ReDet | rot-equivariant two-stage | zoo ckpt (needs ReResNet backbone wt from ReDet repo) | — | — | optional; extra backbone dependency |
| LSKNet-S + ORCNN | CC-BY-NC backbone | released ckpt, **research-use infer only** | — | — | **results-JSON-only** (weights non-redistributable); exploratory arm |

**Recommended detector set for Config A DOTA-inference arm:** RoI Transformer, S2A-Net, Oriented
RepPoints, Gliding Vertex (four distinct families → the strongest agnosticism claim per GPU-hour). ReDet
carries an extra pretrained-backbone dependency; LSKNet is CC-BY-NC (consumable for a *results-only*
exploratory row, never rehosted — matches the design's LSKNet rule).

### 4.3 Small code that needs writing (all CPU/local, all trivial)
1. **`orchestration/cert_cell.sh`** — a generic wrapper that takes `(dets.jsonl, gt.jsonl, out_dir)` and
   runs `match → calibrate --mondrian → recall --mondrian → audit(gwd) → audit(naive) → r20_generic.py`,
   emitting the same file set as a DIOR cell. Today those steps were ad-hoc per cell; wrapping them makes
   each of the ~6 new cells one command. ~40 lines. **Stageable now.**
2. **Cell-list edits, not new code** — `coverage_matched_2026-07-13/coverage_matched_runner.py` and the
   Holm runner hold a hardcoded `CELLS = [...]` of `matched.jsonl` paths. Adding HRSC / new-DOTA-detector
   cells is **appending paths**, then re-running (CPU). **Stageable now** (paths can be pre-filled to the
   expected substrate locations).
3. **DOTA zoo-detector inference manifest** — a `configs_dota_zoo/` dir mirroring `configs_hrsc/`: the
   four vendored configs (copied from the pinned mmrotate commit, num_classes=15, le90) + a small
   `dota_zoo_infer.sh` looping `score_rtmdet.py` over `{cfg, released-ckpt-URL}` pairs on the DOTA val
   crops. ~1 config-copy + ~30-line runner. **Stageable now** (offline: vendor configs + record the four
   zoo checkpoint URLs/licenses).
4. **Regime-conditional coverage-matched extension** (§5) — extend the coverage-matched runner to report
   the matched-coverage size ratio *within the square / boundary / interior θ-strata* (reusing
   `rotcert.audit.classify_theta_stratum`). ~50 lines, pure CPU. **Stageable now.**
5. **HRSC multi-seed (Config B only)** — `hrsc_run.sh` hardcodes `seed_0`; parameterize `ORCNN_SEEDS` /
   `RTMDET_SEEDS` exactly as `next_boot_rotcert_dior_train.sh` already does. ~10-line edit. Stageable now.

---

## 5. The highest-leverage move: the coverage-matched efficiency study "done right" (CPU, ~$0)

This is the item that most changes TGRS odds *and costs no GPU time*. The current runner
(`coverage_matched_runner.py`) already re-tunes each baseline to GWD's realized coverage per split and
reports the honest result: GWD wins at matched coverage only vs naive on DIOR-R. Two extensions make it a
*positive, mechanistic* finding instead of a walked-back one:

1. **Add every new cell** (HRSC×2, DOTA×4 new detectors, and — Config B — DIOR-R×2 new detectors). Pure
   CPU: the runner reads each cell's `matched.jsonl` and recomputes. **Prediction to test:** GWD's
   matched-coverage advantage tracks object elongation. On DOTA (many compact objects) it is a wash; on
   DIOR-R (mixed) it survives at ~0.8×; **on HRSC (all ships, extreme aspect ratios) GWD should win
   most decisively** — because that is exactly the seam/square regime the score is built for. If HRSC
   confirms this, the efficiency story is *reborn* as "GWD is more efficient at matched coverage where it
   should be (elongated objects), and neutral where it needn't be (compact) — a predictable, principled
   pattern," not a headline that had to be retracted.
2. **Regime-conditional matched comparison** (code item §4.3.4): report the matched-coverage size ratio
   *within* the square / boundary / interior θ-strata. The paper already shows naive CP's coverage
   collapses on square/boundary strata (Table `tab:naive`); the natural completion is to show that at
   *matched* coverage GWD's *region* is smaller precisely on those strata. This is the coverage-fair
   version of the paper's own motivating hook and it is free.

Honest caveat to keep in the plan: more cells will **not** un-retract the efficiency claim by themselves.
The claim is repaired only if the elongation hypothesis holds in the data. If HRSC comes back a wash too,
the honest outcome is "GWD is coverage-efficient only vs naive on DIOR-R," and the paper still stands on
correctness — but HRSC is the dataset most likely to *win*, and it is cheap to find out.

---

## 6. GPU-hours, wall-clock, cost (RTX 4090D 24 GB, ~$0.5/GPU-h ceiling)

**Measured actuals** (from the pulled DIOR-R training logs — grounds all estimates):
- Oriented R-CNN R-50 **1x** on DIOR-R (12 ep, batch 2, 11.7k imgs): **~7.6 GPU-h** (0.39 s/iter ×
  5863 iter/ep × 12). 
- RTMDet-R-l **3x** on DIOR-R (36 ep, batch 8): **~6–8 GPU-h** (~10 min/epoch; two useful sessions summed
  ≈ 5 h 54 m). *Both are well under the design's conservative 25–35 GPU-h envelope — real oriented-detector
  training on these datasets is cheaper than the container spec assumed.*

**Per-unit costs used below:** DOTA val inference ~0.5 GPU-h/detector; DIOR-R test inference ~0.5–1
GPU-h; HRSC (436 train imgs) train ~0.5 GPU-h/detector (1x *or* 3x — dataset is tiny), HRSC test infer
~0.1 GPU-h; a fresh DIOR-R 1x detector ~8 GPU-h. Certification + coverage-matched + Holm are **CPU, local,
free**.

| Line item | GPU-h | Notes |
|---|---|---|
| **Config A** | | |
| HRSC convert + smoke + backbone fetch overhead | ~0.5 | CPU convert; 1-epoch smoke gate |
| HRSC train ORCNN 1x + RTMDet 3x (seed 0) | ~1.0 | tiny dataset |
| HRSC infer test (both) + GT prep | ~0.3 | |
| DOTA infer ×4 new detectors (RoI-Trans, S2A-Net, Ori-RepPoints, Gliding-Vertex) | ~2–3 | released zoo ckpts |
| certify all new cells + coverage-matched (+regime) + Holm | 0 (CPU) | local |
| **Config A total** | **~5–8 GPU-h** | **~1 box-day incl. gates; ~$3–5** |
| **Config B add-ons** | | |
| DIOR-R train RoI-Transformer 1x + S2A-Net 1x + infer | ~17 | cost driver |
| HRSC 3-seed × 2 detectors (2 extra seeds each) | ~3 | detector-variance, cheap |
| **Config B total** | **~25–30 GPU-h** | **~2–3 box-days; ~$12–15** |
| **Config C** (reference only) | **~80–120** | full zoo × trio + multi-seed; ~$40–60; not recommended |

**Wall-clock caveats:** oriented-detector *training* is the only real clock cost and the DIOR-R RTMDet run
above shows it may need one resume (checkpoint at epoch 12) — the chains already handle resume + integrity
gates, so wall-clock ≈ GPU-h + ~1 h overhead per box session (build/stage/tar/pull). Data-disk: HRSC-MS
2.3 GB zip → ~4 GB unzipped + converted; DOTA/DIOR-R already staged; fits the provisioned ~170 GB.

---

## 7. Local prep stageable NOW (before any box, zero spend)

All of these are offline/CPU and de-risk the box run:
1. **Write `orchestration/cert_cell.sh`** (§4.3.1) — the generic per-cell certification wrapper; test it
   locally against an existing DIOR `matched.jsonl` so it reproduces `dior_cert_results_2026-07-11/` byte
   for byte before it ever touches new substrate.
2. **Write `configs_dota_zoo/` + `dota_zoo_infer.sh`** (§4.3.3) — vendor the four DOTA detector configs
   from the pinned mmrotate commit and record each released checkpoint's URL + license into a small
   manifest. Makes the DOTA-inference arm one command on the box.
3. **Extend the coverage-matched runner** to regime-conditional (§4.3.4) and pre-append the expected new
   cell paths to its `CELLS` list (they resolve once substrate lands) — validate on the 4 existing cells
   (must reproduce `results.json`).
4. **Parameterize `hrsc_run.sh` seeds** (§4.3.5) for the Config-B multi-seed option.
5. **(Optional) Pre-fetch HRSC-MS locally** via the verified gdown route to checksum the 2.3 GB zip and
   push it to the box, eliminating the only box-side download risk. Then run `prepare_hrsc.py` locally
   (CPU, no mmrotate) to confirm the **angle-convention gate passes (~0.81 median IoU)** before spending
   any GPU time — the single most valuable pre-flight check.
6. **Freeze the HRSC disclosure line** ("HRSC2016-MS, canonical source split, single-class ship, one
   Mondrian cell") and the new-detector provenance (zoo commit + checkpoint URLs + LSKNet results-only
   rule) into the manuscript's data-availability section ahead of the run.

---

## 8. Honest verdict — what moves TGRS odds, ranked

1. **Coverage-matched + regime-conditional efficiency across the enlarged grid (§5).** Free (CPU). Repairs
   the single most-cited weakness *mechanistically* if the elongation hypothesis holds; HRSC is the test.
   **Do this regardless of budget.**
2. **HRSC2016 as the 3rd dataset (Config A).** ~2 GPU-h, scaffolding already written, kills the "only 2
   datasets" objection and supplies the extreme-aspect-ratio data that item 1 needs. **Top GPU spend.**
3. **DOTA detector-zoo breadth, inference-only (Config A).** ~2–3 GPU-h for four distinct detector
   families → converts "score-agnostic" from assertion to evidence at near-zero cost.
4. **HRSC 3-seed detector variance (Config B).** ~3 GPU-h → closes the single-checkpoint caveat cheaply.
5. **DIOR-R detector depth (Config B).** ~17 GPU-h → real, but the priciest per unit of reviewer goodwill;
   do only if pushed.
6. **Full zoo × trio (Config C).** Not worth it.

**Recommendation: execute Config A (≈$3–5, one box-day), with the CPU efficiency analysis of §5 as the
intellectual centerpiece.** It is the qualitative jump from a two-dataset pilot to a comprehensive
certified-oriented-detection study across the canonical trio and a modern detector zoo, and the compute is
dominated not by GPU cost but by CPU analysis that is already 90% built.

# DIOR-R multi-seed arm -- local post-pull plan (2026-07-14)

Box arm: `orchestration/next_boot_dior_seeds.sh` (runs on box 19514 after the Config-B
chain finishes S2A-Net). It produces, per new seed, a DIOR-R TEST-split detections JSONL:

```
rotcert_dior_seeds_results/
  dior_test_dets_orcnn_dior_seed1.jsonl
  dior_test_dets_orcnn_dior_seed2.jsonl
  dior_test_dets_rtmdet_r_dior_seed1.jsonl      # absent if SKIP_RTMDET=1 was used
  dior_test_dets_rtmdet_r_dior_seed2.jsonl      # absent if SKIP_RTMDET=1 was used
  <detector configs>, per-seed *_train.log, *_coverage.txt
```

Everything below is **local, CPU, post-pull**. The new seed cells reuse the SAME frozen
GT and matcher/calibration settings as seed 0 -- that is what makes them join the existing
coverage-matched DIOR-R grid (identical scenes, IoU matcher, alpha, Mondrian strata).

## 0. Inputs (already local)

- Frozen DIOR-R TEST GT (seed-independent -- ONE file, shared by all seeds):
  `dior_cert_results_2026-07-11/orcnn_fixedgt/` was produced with this GT; the canonical
  frozen GT JSONL is the one the seed-0 cells were certified against. Reuse it verbatim.
  Set `GT=<frozen dior_test_gt.jsonl>` (do NOT regenerate -- byte-identical GT is what keeps
  the grid coverage-matched).
- Seed-0 cells (the existing reference, for the spread comparison):
  - Oriented R-CNN seed 0: `dior_cert_results_2026-07-11/orcnn/` (or `orcnn_fixedgt/`)
  - RTMDet-R seed 0:       `dior_cert_results_2026-07-11/rtmdet/`

## 1. Certify each new seed cell (cert_cell.sh -- one call per cell)

`cert_cell.sh` runs the exact CPU chain that produced each DIOR/DOTA cell (match ->
calibrate gwd/naive-coord/hull -> recall -> audit gwd/naive -> r20_generic), emitting the
same file set (`matched.jsonl`, `cert_*.json`, `recall.json`, `audit_*.json`,
`r20_coverage.json`). Interface (see its header): `--dets D.jsonl --gt G.jsonl --out-dir OUT`.

After pulling `rotcert_dior_seeds_results/` into e.g.
`dior_seedcells_2026-07-14/`, run one cell per new (detector, seed):

```bash
ORCH=reliability-commons/tools/rotcert/orchestration
CELLS=dior_seedcells_2026-07-14
GT=<frozen dior_test_gt.jsonl>          # SAME file seed 0 used

# Oriented R-CNN seeds 1,2
for s in 1 2; do
  bash "$ORCH/cert_cell.sh" \
    --dets "$CELLS/orcnn_dior_seed${s}/dior_test_dets_orcnn_dior_seed${s}.jsonl" \
    --gt   "$GT" \
    --out-dir "$CELLS/orcnn_seed${s}"
done

# RTMDet-R seeds 1,2  (skip if the box ran with SKIP_RTMDET=1)
for s in 1 2; do
  bash "$ORCH/cert_cell.sh" \
    --dets "$CELLS/rtmdet_r_dior_seed${s}/dior_test_dets_rtmdet_r_dior_seed${s}.jsonl" \
    --gt   "$GT" \
    --out-dir "$CELLS/rtmdet_seed${s}"
done
```

Each `cert_cell.sh` uses the DIOR defaults (`--iou-thr 0.5 --iou-metric rotated`, alpha
0.10, Mondrian per-class) -- identical to the seed-0 DIOR cells. Do NOT override them.

## 2. q_hat / coverage spread extension (mirror extract_qhat_spread.py)

`hrsc_seedcells_2026-07-14/extract_qhat_spread.py` reports, per cell, mean/min/max q_hat
across the 20 scene reseeds in `r20_coverage.json`, then the across-seed spread of the
per-cell mean q_hat within each detector family (the seed-sensitive region-size quantity;
marginal coverage is pinned at nominal by split-conformal construction and is NOT the story).

Make a DIOR analogue `dior_seedcells_2026-07-14/extract_qhat_spread_dior.py` -- a
verbatim clone of the HRSC script with only the `CELLS` table repointed to the DIOR cells:

```python
CELLS = [
    ("Oriented R-CNN", 0, ROOT / "dior_cert_results_2026-07-11/orcnn/r20_coverage.json"),
    ("Oriented R-CNN", 1, ROOT / "dior_seedcells_2026-07-14/orcnn_seed1/r20_coverage.json"),
    ("Oriented R-CNN", 2, ROOT / "dior_seedcells_2026-07-14/orcnn_seed2/r20_coverage.json"),
    ("RTMDet-R",       0, ROOT / "dior_cert_results_2026-07-11/rtmdet/r20_coverage.json"),
    ("RTMDet-R",       1, ROOT / "dior_seedcells_2026-07-14/rtmdet_seed1/r20_coverage.json"),
    ("RTMDet-R",       2, ROOT / "dior_seedcells_2026-07-14/rtmdet_seed2/r20_coverage.json"),
]
```

`cell_stats()`, `main()`, and the output format are unchanged -- it prints per-cell
`q_hat mean [min,max] (std,n)` and the family across-seed `mean-q_hat range / spread /
% of family mean`. For the ORCNN-only variant (SKIP_RTMDET=1) drop the three RTMDet-R rows.

Fold the DIOR spread numbers into the seed-variance section (alongside the HRSC
`SEED-VARIANCE-DIGEST.md` / `QHAT-SPREAD.md`) so the "region size is seed-stable" claim is
demonstrated on the DIOR-R dataset the coverage-matched efficiency result actually defends,
not only on HRSC.

## 3. Cost / benefit (honest, from the compute plan's measured rates)

| Variant                         | New cells | GPU-h (box) | Defends |
|---------------------------------|-----------|-------------|---------|
| ORCNN-only (`SKIP_RTMDET=1`)    | 2         | ~16         | seed-stability of ORCNN DIOR cell |
| Both detectors (default)        | 4         | ~28-32      | both core detectors on DIOR-R (critic's ask) |

RTMDet-R 3x is ~6-8 GPU-h/seed (~10 min/epoch one-stage), NOT ~24 -- so the both-detector
arm is the recommended default (matches the critic's "two core detectors" recommendation
at only ~2x the ORCNN-only cost). Use `SKIP_RTMDET=1` only if box time is tight.

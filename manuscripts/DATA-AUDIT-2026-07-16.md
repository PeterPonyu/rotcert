# rotcert manuscript data audit — 2026-07-16

Read-only audit of `paper.tex` (981 lines) and `tgrs/paper_ieeetran.tex` (961 lines) against the
frozen result JSONs named in the `% source:` comments, plus the two figures' source CSVs. Two audit
dimensions: (1) cross-paper / cross-artifact number consistency, (2) assumption-based claims vs frozen
evidence.

## Verdict

The paper is **exceptionally clean**. Every headline number, every one of the 72 table rows, both
figures, and the abstract were traced to source JSONs and re-derived; the two manuscript versions carry
**identical** numbers wherever both display a value. Only **2 MINOR rounding blemishes** and **2 low-severity
prose-scope softenings** were found. No CRITICAL findings. No fabricated, contradictory, or
unsupported-by-source numbers.

Severity counts: **CRITICAL 0 · MINOR 4** (2 rounding, 2 scope-phrasing).

## Coverage (what was verified exact)

- **Table g1oos** (4 rows): mean + BCa CI vs `pilot_oos`, `orcnn_dota_cert`, `dior_cert_results/{orcnn_fixedgt,rtmdet}/r20_coverage.json` — exact.
- **Table g1mondrian** (all 15 DOTA rows): n_cal, q̂, π q̂² vs `pilot_results_2026-07-10/.../cert_gwd.json` — exact (e.g. ship 18125/3.50/38.6, ground-track-field 205/22.09/1533.1).
- **Table diorperclass** (all 20 rows × ORCNN cov + RTMDet cov + n_cal, and the G2 O/R/– column) vs `dior_perclass_2026-07-15/results.json` — exact; the G2 markers reconcile cell-by-cell with `recall.json` certified rosters.
- **Table g2** (2/15 DOTA + 5/20 & 6/20 DIOR-R, example classes + λ*/realized risk) — exact; DOTA small-vehicle (0.051,0.024), tennis-court (0.094,0.001) confirmed; RTMDet "−stadium,+baseballfield,storagetank" delta reconciles.
- **Table naive** (5 rows × 6 regime columns) vs the six `audit_{naive,gwd}.json` (`v2_stratum_coverage`) — exact except finding #1 below.
- **Table configA** (4 rows) vs `configA_cert_2026-07-14/*/audit_*.json + r20_coverage.json` — exact; n_TP (799/670/4246/5617) confirmed = `audit_gwd v1_marginal_coverage.n` (matched.jsonl also holds FPs, so its line count is not n_TP — no defect).
- **Table configB** (2 rows) vs `configB_cells_2026-07-15/*` — exact; RoI-Trans 89,974/10,966 and S2A-Net 95,192/10,940 confirmed.
- **Table seedvar** (6 rows): coverage, CI, n_TP (799/727/758/670/630/737), R=20 mean[min,max], G2 refusal vs `configA_cert` (seed 0) + `hrsc_seedcells_2026-07-14/*` (seeds 1,2) — exact.
- **Table r20holm** (8 rows: mean, [min,max], 20/20) vs `holm8_r20_exploratory.json` — exact; worst split 0.974 (RTMDet-DOTA hull) confirmed.
- **Table covmatched** (12 rows) vs `coverage_matched_2026-07-13` + `coverage_matched_configB_ext_2026-07-15` — exact except finding #2; the nominal column for the 4 original cells correctly sources from Table r20holm (as its caption states), not this file's own slightly different `ratio_nominal_r20_mean`.
- **Table regime** (4 rows × AR mean/frac + all/compact/elongated) vs `coverage_matched_regime_2026-07-13` — exact; `regression_guard.passed=true, max_abs_diff=0.0` confirmed.
- **Figure fig1_diorperclass**: all 80 class×cell points in `fig1_perclass.csv` bit-match `dior_perclass/results.json`; band min/max (0.88664.../0.91268...) exact.
- **Figure fig2_covmatched**: all 12 rows (mean/min/max) bit-match the two coverage_matched JSONs.
- **Holm-8 single-split ratios** 0.67/0.86, 0.60/0.78, 0.33/0.48, 0.40/0.59 and p_value=1/2001, p_holm=0.004 vs `holm8_result.json` — exact; the "8/8 GWD smaller" degeneracy structure verified.
- **HRSC reversal** 1.2413/1.0198 (mean 1.1305), zoo 1.6618/0.7056, hull ranges 1.30–2.44, AR figures (HRSC eval 5.63/5.95, source 5.22 & 82%, zoo 1.27/1.47 & 0/0.039) vs `configA_regime_extend` + `hrsc_prefetch` — exact.
- **Inline numbers**: DIOR ship q̂=3.99/area 49.9→50, dam q̂=73.9/area 17134; DOTA in-sample identity 48719/54131=0.90002; q̂ seed spread 1.5% (39.57–40.17) / ~12% (48.23–54.26); r_hat values 0.5464/0.6076/0.9380/0.8496/0.2230/0.2118 — all exact.
- **Cell-count claims**: "ten G1 cells / three datasets / five detector architectures" re-derived (DOTA pilot, DOTA-ORCNN, DIOR-ORCNN, DIOR-RTMDet, HRSC-ORCNN, HRSC-RTMDet, RoI-Trans·DOTA, RepPoints·DOTA, RoI-Trans·DIOR, S2A-Net·DIOR = 10; DOTA/DIOR-R/HRSC2016-MS = 3; RTMDet-R, Oriented R-CNN, RoI Transformer, Oriented RepPoints, S2A-Net = 5) — correct.
- **Cross-version**: abstract numbers **identical**; all 72 table rows agree on every shared cell. IEEEtran differences are deliberate two-column reductions only — configA/configB tables drop the naive-coord CI and one of the n_TP/scenes columns; seedvar drops the G2 column (moved to caption). No value diverges. The refusal/degeneracy sub-claims' scopes match exactly between versions.

## Findings

### MINOR-1 (rounding) — DIOR-ORCNN naive marginal shown as 0.936, source rounds to 0.935
- **Where**: Table `tab:naive`, DIOR-ORCNN row, "naive marg." cell — `paper.tex:437`, `tgrs/paper_ieeetran.tex:459` (both identical).
- **Source**: `dior_cert_results_2026-07-11/orcnn_fixedgt/audit_naive.json` → `v1_marginal_coverage.point = 0.9354838709677419`, which rounds to **0.935**. The manuscript shows **0.936** (a classic double-rounding: 0.93548 → 0.9355 → 0.936).
- **Impact**: cosmetic, 0.001. The prose range "over-covers marginally everywhere ($0.933$–$0.936$)" (`paper.tex:414`) stays **correct** — the true 0.936 upper bound is supplied by DIOR-RoI-Trans (0.9364), not this cell.
- **Fix**: change the DIOR-ORCNN naive-marg cell `0.936` → `0.935` in both files.

### MINOR-2 (rounding) — RTMDet–DIOR-R coverage-matched naive ratio shown as 0.82, source rounds to 0.83
- **Where**: `tab:covmatched` (`paper.tex:781`), `tab:regime` "all" column (`paper.tex:885`), and prose "$\approx0.82\times$ RTMDet" (`paper.tex:739`) — all show **0.82**; identical in the IEEEtran version (`:776`, `:866`).
- **Source**: `coverage_matched_2026-07-13/results.json` RTMDet-DIOR naive `ratio_matched_r20_mean = 0.82534` (also `regression_guard` frozen_parent_ratio 0.82534). Standard rounding → **0.83**; the manuscript truncates to 0.82.
- **Impact**: cosmetic and internally consistent (0.82 everywhere), but the convention is uneven — the sibling RoI-Trans cell in the same table rounds (0.7579 → 0.76). The range claim "0.76–0.85×" and the pooled "$\approx0.8\times$" are unaffected either way.
- **Fix**: prefer 0.82 → 0.83 in all three spots (or leave, but note the truncation is deliberate).

### MINOR-3 (scope phrasing) — Results states the DOTA under-coverage cause more flatly than Methods/Limitations
- **Where**: `paper.tex:262–266` — "Decisively … the shortfall is a dataset artifact — DOTA's overlapping 1024-px crops leave residual within-scene correlation that scene-level splitting reduces but does not remove — not a detector or score defect."
- **Assessment**: the "not a detector or score defect" half is **well-supported** by the 2×2 cross (same detectors and score cover at nominal on single-tile DIOR-R). The specific mechanism (residual within-scene crop correlation surviving scene-level splitting) is a **hypothesis, not directly measured**; Methods/Limitations correctly hedge it ("we attribute this to…", `paper.tex:897–899`). The Results phrasing "is a dataset artifact" + "Decisively" is more confident than the hedged version elsewhere.
- **Fix (optional)**: soften "is a dataset artifact" → "appears to be a dataset artifact" to match the Limitations hedge.

### MINOR-4 (scope phrasing) — "covers every class conditionally" summary drops its own qualifier
- **Where**: `paper.tex:346–347` — "The geometry certificate thus covers every class conditionally, while the recall certificate honestly abstains on the rare ones."
- **Assessment**: literally, several classes sit at 0.887–0.899 (below the 0.90 nominal), so read in isolation this overstates. It is heavily pre-qualified in the immediately preceding sentences (no class deviates >≈1.3 points; sub-nominal classes within one split-to-split std — **verified**: every sub-nominal class in both cells has |0.90 − mean| ≤ its per-class std). Not misleading in context, but the one-line summary drops "approximately."
- **Fix (optional)**: "covers every class approximately at nominal (within ≈1.3 points / one split-to-split std)."

## Assumption-based-claims dimension — overall

The manuscript is unusually disciplined. Every efficiency, degeneracy, and refusal claim is scoped to
exactly what the frozen evidence supports, and the scopes match between the two versions:
- Holm-8 "demoted to descriptive / p-values carry no evidence strength / effective n=1" matches the
  pooled-calibration single-scalar structure (verified: one unique set-size per cell).
- The registered HRSC elongation prediction is reported **reversed** with exact numbers, and the
  cross-dataset efficiency claim is explicitly **withdrawn** ("DIOR-R-specific, not general").
- G2 refusals are scoped to the LTT-HB power floor with the printed floor values (verified).
- The EAV-DETR differentiation is explicitly caveated as resting on the public abstract + code, full text
  paywalled.
No causal or generalization claim was found to exceed the tested datasets/detectors beyond the two mild
softenings above.

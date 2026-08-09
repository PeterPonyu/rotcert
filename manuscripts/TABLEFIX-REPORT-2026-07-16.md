# rotcert table-fix + figure-typography report — 2026-07-16 (review directive #2)

Applies `docs/records/REVIEW-DIRECTIVE-2-2026-07-16.md` to the canonical `paper.tex`
(`article`) and the venue kit `tgrs/paper_ieeetran.tex` (IEEEtran two-column), in lockstep,
plus the shared figure sources under `figures-src/`.

Backups (pre-edit): `paper.tex.bak-pre-tablefix`, `tgrs/paper_ieeetran.tex.bak-pre-tablefix`,
`figures-src/fig0_overview.tex.bak-pre-tablefix`.

## §1 Small-table audit (< 5 DATA rows)

Every `table`/`table*` enumerated; data rows exclude header/rule/group-heading rows.

| Label (before) | Data rows | Flagged (<5) | Verdict |
|---|---|---|---|
| `tab:g1oos` | 4 | yes | **KEEP + justification** |
| `tab:g1mondrian` | 15 | no | keep |
| `tab:diorperclass` | 20 | no | keep |
| `tab:g2` | 3 | yes | **CONVERT to prose** |
| `tab:naive` | 5 | no | keep |
| `tab:configA` | 4 | yes | **MERGE → `tab:extcells`** |
| `tab:configB` | 2 | yes | **MERGE → `tab:extcells`** |
| `tab:seedvar` | 6 | no | keep |
| `tab:r20holm` | 8 | no | keep |
| `tab:covmatched` | 12 | no | keep |
| `tab:regime` | 4 | yes | **KEEP + justification** |

Five flagged; resolved as 1 merge (2→1 table), 1 prose conversion, 2 justified keeps.

### Remedies applied
- **`tab:configA` + `tab:configB` → `tab:extcells` (MERGE, remedy 1a).** Identical result
  family (extension G1 cells: audited GWD + naive coverage, out-of-sample R=20 recalibration,
  matched-TP + scene counts). Merged into one 6-data-row table with two plain (non-bold)
  `\multicolumn` group-heading rows: "Config-A --- HRSC2016-MS and DOTA-v1.0 detector zoo" and
  "Config-B --- cross-family DIOR-R". Canonical merged-table data cells are **byte-identical**
  to the two source tables (verified: `diff` of numeric multiset = empty). All 3 `\ref` sites
  updated (canonical: 2 body refs; TGRS: 2 body refs); the old `tab:configB` caption's
  self-reference to `tab:configA` was dropped (now references `tab:g1oos` for "same frozen
  pipeline"). Float declared once, at the first-reference site in §configA.
  - TGRS note: the pre-existing TGRS `configA` table showed scene count only, `configB` showed
    n_TP only (venue width trims). To carry BOTH families in one table with every number
    preserved, the merged TGRS `tab:extcells` shows both columns; this **adds 6 frozen tokens**
    to the TGRS file that canonical already displayed — n_TP {799, 670, 4,246, 5,617} for the
    Config-A rows and scenes {10,966, 10,940} for the Config-B rows (all from the same
    `configA_cert_2026-07-14` / `configB_cells_2026-07-15` records). No existing value altered or
    dropped (diff = additions only). Both venue tables now share one 6-column schema.
- **`tab:g2` → prose (CONVERT, remedy 1b).** The 3 count rows (2/15, 5/20, 6/20) were already
  stated verbatim in the preceding paragraph; only the example-certified-class column was
  table-only. Folded into one sentence in both files, preserving every number
  (small-vehicle 0.051/0.024, tennis-court 0.094/0.001) and every class name. The
  `% source: ... recall.json` provenance comment retained inline. Removed the table's
  `p{0.52\textwidth}` column spec (the sole benign numeric-token loss, `0.52`; `array` package
  left loaded, harmless). `\ref{tab:g2}` site de-referenced (prose now self-contained).
- **`tab:g1oos` KEEP + justification.** Load-bearing headline G1 marginal-coverage table for the
  four core cells; unique schema (out-of-sample R=20 point+BCa CI alongside the in-sample
  self-consistency diagnostic) not shared by any other table, so no clean merge partner. A forced
  merge into `tab:extcells` would leave the naive/n_TP/scenes columns blank for these rows and the
  diagnostic column blank for the extension rows.
- **`tab:regime` KEEP + justification.** Load-bearing descriptive finding (within-cell
  elongation gradient — the efficiency-mechanism narrative). AR-stratified schema
  (all / compact / elongated + AR mean) is incompatible with `tab:covmatched`'s
  cell×baseline schema (regime covers only the 4 original naive cells, not the hull rows nor the
  two Config-B cells); merging would blank the hull rows. No further frozen rows exist to expand
  (regime analysis was, by the paper's own statement, not re-run on the new cells).

All kept/merged tables remain referenced in first-occurrence order; 0 undefined refs.

## §2 Figure typography — effective size (native px × included width)

Native PDF widths (measured, pt): fig0 233.9, fig1 455.4, fig2 256.0.
Included: canonical = native (no width spec, scale 1.00); TGRS fig0/fig2 `\columnwidth` (~252pt),
fig1 `\textwidth` in `figure*` (~516pt).

| Figure | text tier | canonical eff. (before → after) | TGRS eff. (before → after) |
|---|---|---|---|
| fig0 | primary node text | 8.0 → 8.0 | 8.6 → 8.6 |
| fig0 | "refuse" edge label | **6.8 → 7.5** | 7.3 → 8.1 |
| fig1 | axis tick | 8.0 → 8.0 | 9.1 → 9.1 |
| fig1 | reference-line label | 7.5 → 7.5 | 8.5 → 8.5 |
| fig2 | axis tick | 8.0 → 8.0 | 7.9 → 7.9 |
| fig2 | reference-line label | 7.5 → 7.5 | 7.4 → 7.4 |

- Only outlier was **fig0's 6.8pt "refuse" edge labels**, which fell below the 7pt floor when
  fig0 is included at native size in the canonical. Raised to 7.5/8.5 — this also harmonizes the
  reference-annotation tier at 7.5pt across all three figures (fig1 "nominal", fig2 "parity").
- Primary text tier is uniform at 8pt native across all three figures; effective sizes land in the
  8–9pt band in both venues (TGRS range 7.9–9.1pt tick), none below 7pt after the fix.

## §3 Decorative bold/italic — including inside figures

**Inside figures (fig0 only; fig1/fig2 had none):**
- Removed `\textit{...}` decorative parenthetical ("(black box — any architecture: …)") → plain.
- Removed `\textbf{G1}`, `\textbf{G2}`, `\textbf{Refuse}` box titles ("prefer none"; the boxes are
  already color+border coded, so the bold was redundant).
- fig0 rebuilt via the Makefile (`latexmk` exit 0); native width unchanged (233.9pt), so effective
  sizes above are stable.

**Figure decorative-markup counts:** fig0 `\textbf` 3→0, `\textit` 2→0; fig1 0→0; fig2 0→0.

**Main-text re-sweep (crept back via recent additions):**
- Removed one decorative full-sentence bold that returned via the 2026-07-15 coverage-matched
  additions: `\textbf{At the pooled level the coverage-fair advantage is narrow: …on DIOR-R}`
  (both files). This was the exact offender directive #1 had removed once already.
- Table cell number highlights `\mathbf{...}` removed everywhere (11 in canonical, 11 in TGRS:
  `tab:naive` ×3, `tab:covmatched` ×4, `tab:regime` ×4). Directive #1 had deliberately left these;
  directive #2's stricter "no decorative bold" re-sweep removes them, making all tables uniformly
  bold-free. Digits unchanged (unwrap only) — verified. **(Reversible if the team prefers the
  best-value-bold convention; flagged as a judgment call.)**
- All remaining `\textbf` are run-in paragraph headers (period-terminated lead-ins) + the title +
  the never-rendered `\todo` macro — permitted by the house rule. Zero mid-sentence `\textbf`.
- `\emph` / `\textit` / `\itshape` / `\bfseries` in main text: **0 / 0** (both files). No bold in
  captions beyond the class-emitted "Figure N:" / "TABLE N" label; no italic emphasis in captions.

## §4 Rails

- **Zero result-number changes.** Distinct-decimal check (backup vs final): canonical loses only
  `0.52` (the removed g2 column-width spec, not data); TGRS loses none. Merged-table data cells
  verified against source tables (canonical identical; TGRS = additions of 6 already-frozen tokens
  only). All coverage / q̂ / CI / ratio / count / λ*/risk values preserved with their source
  comments.
- `latexmk` exit **0** on canonical + TGRS. **0** undefined / multiply-defined refs on both.
- Worst overfull hbox: canonical none ≥1pt; TGRS 1.10pt (pre-existing, negligible). The merged
  6-column TGRS `table*` fits (`\small`, no overflow).
- Backups in place; edit classes limited to table merge/convert, ref retargeting, `\mathbf`/
  decorative-markup removal, and the fig0 typography fixes.

## Page counts (delta)

| | before | after |
|---|---|---|
| canonical `paper.pdf` | 23 | **22** (−1) |
| TGRS `paper_ieeetran.pdf` | 13 | **13** (±0) |

**Prominent flag (TGRS trim decision):** the table merges did NOT drop a TGRS page. The three
removed floats (g2 + configA + configB) collapsed into one taller merged `table*`, and the g2→prose
fold added ~4 body lines — net page-neutral for the two-column layout. TGRS remains **13pp**, one
page into the OPC-charge zone (charges from p11). The standing trim/OPC decision is unchanged and
still the user's. The canonical single-column build did recover one page (23→22).

## Figure CONTENT audit (directive §3b, 2026-07-16 second pass)

Only 3 figures exist in this paper: `fig0_overview` (schematic), `fig1_diorperclass`, and
`fig2_covmatched`. No edits were needed — no backup taken (no source files touched).

| Figure | Source data available | Data actually plotted | Verdict |
|---|---|---|---|
| `fig0_overview` (`fig:overview`) | n/a — symbolic TikZ pipeline diagram (G1/G2 method schematic), no result data | n/a | **N/A** — not a results figure; §3b content-sparseness does not apply. No shading/gradient fills (flat TikZ colors only); no bold/italic beyond the already-audited box titles. |
| `fig1_diorperclass` (`fig:diorperclass`) | `dior_perclass_2026-07-15/results.json`: 4 detector cells (ORCNN, RTMDet, RoI-Trans, S2A-Net) × 20 DIOR-R classes = 80 per-class coverage points, plus `nominal_coverage` and global min/max band | All 80 class×cell points plotted (one series per detector, 20 classes each), plus the shaded reference band (global min/max) and the nominal-coverage line | **Already fully enriched** — this is the maximal breakdown the frozen record supports (every class, every certified DIOR-R detector); matches the team-lead's prior note. No change. |
| `fig2_covmatched` (`fig:covmatched`) | `coverage_matched_2026-07-13/results.json` + `coverage_matched_configB_ext_2026-07-15/results.json`: 6 cells (RTMDet/ORCNN/RoI-Trans/S2A-Net × DOTA/DIOR-R, as applicable) × 2 baselines (naive-coord, hull) = 12 cell×baseline points, each with `per_split_ratio_matched` (raw 20-value array) plus pre-reduced mean/min/max | All 12 cell×baseline points plotted as mean with min–max whiskers (the full realized `R=20` range, not just mean) | **Already fully enriched** — every cell in `Table~\ref{tab:covmatched}` (the same 12 rows) has a corresponding point in this figure; the whiskers already carry the complete `R=20` spread (min/max), so no information is being summarized away versus a raw 20-point strip plot. No change. |

**Decorative-element sweep (§3b directness check, all 3 figures):** no ornamental frames,
gradient fills, or drop shadows in any `.tex` source. `fig1`'s `\fill[black!8]` band and both
figures' dashed reference/parity lines are structurally meaningful (global coverage spread;
nominal target; ratio=1 parity), not decorative. No pgfplots `grid=...` option is set in either
axis environment — no gridlines of any kind, so none to remove. Legends: `fig1` has 4 entries (one
per certified detector, all load-bearing); `fig2` has 2 entries (one per baseline, all
load-bearing); neither duplicates caption content. Axis space in both figures is fully used by
data (no unused margin reserved for decoration). No changes required.

**Rebuild verification:** `latexmk -pdf` re-run on canonical (`paper.tex`) and TGRS
(`tgrs/paper_ieeetran.tex`) after this audit pass (no source changes, confirms the audit made no
regressions):

| | Canonical (`paper.tex`) | TGRS (`tgrs/paper_ieeetran.tex`) |
|---|---|---|
| `latexmk -pdf` exit code | 0 | 0 |
| Undefined references | 0 | 0 |
| Page count (at this rebuild) | 23 | 13 |

Note: canonical page count read 23 at this rebuild, not the 22 recorded in the table-audit
pass above — `paper.tex` carries edits with a later mtime than that pass (concurrent
table-fix work in this session), so the 23 reflects the current file state, not a
regression introduced by this figure-content audit (which made zero edits to any rotcert
source file; see "Files touched" below).

## Files touched (§3b pass)
- None. Audit-only; no figure source, data-extraction script, or caption required a change.

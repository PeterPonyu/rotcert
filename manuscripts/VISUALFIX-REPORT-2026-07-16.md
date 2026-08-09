# rotcert visual-fix report — 2026-07-16 (review directive #3, user visual audit)

Applies `docs/records/REVIEW-DIRECTIVE-3-2026-07-16.md` to the canonical `paper.tex` (`article`) and
the venue kit `tgrs/paper_ieeetran.tex` (IEEEtran two-column), in lockstep, plus the shared figure
source `figures-src/fig0_overview.tex`. Every fix verified VISUALLY: PDF pages rasterized (pdftoppm)
and read before AND after.

Backups (pre-edit): `paper.tex.bak-pre-d3`, `tgrs/paper_ieeetran.tex.bak-pre-d3`,
`figures-src/fig0_overview.tex.bak-pre-d3`, `refs.bib.bak-pre-d3`.

## Per-issue fixes

### 1. fig1 (`fig:overview`, Figure 1) — black-box + long texts → shapes/short labels [rotcert-specific]
Rebuilt `figures-src/fig0_overview.tex`:
- **Detector box now presentable, black-box concept kept.** Old node was a solid 100%-black rectangle
  crammed with four lines of white text (architecture roster). New node is a charcoal (`#1A1A1A`) fill
  with a light keyline border and a single short label "$\forall$ frozen oriented detector (black
  box)". Solid pure-black read as a print blob; charcoal+keyline+short-label reads as an intentional
  sealed box. The five-architecture roster moved to the caption (was already there — no new prose).
- **All boxes → short labels.** G1/G2 boxes reduced to a one-line name + one load-bearing formula each
  (`coverage $\ge 1-\alpha$`; `Pr(R(λ)>β) ≤ δ`); the set-builder line and the α_min formula dropped
  from the figure (both live in the caption / Methods). Refuse box reduced to one short line; its long
  α_min sentence dropped (in caption).
- **Annotation colour → black.** The two "refuse" edge labels were gray (`black!60`/`black!50`); now
  solid black per directive item 3. (The black-box label stays white — white-on-black is contrast, not
  a colour annotation.)
- **Compact.** Native height 232pt (was a tall tower); inter-node gaps tightened; added a small framed
  aerial-image glyph (horizon+sun, TikZ) left of the input node as a shape cue rather than more text.
- Rebuilt via Makefile, `latexmk` exit 0. Renders correctly at native size (canonical) and at
  `\columnwidth` spanning the TGRS column top (page 2). Caption unchanged (already carried the roster +
  α_min formula), so it remains the self-contained explanation the terse figure now needs.

### 2. Table 1 (`tab:g1oos`) text too small — fix sizing, NO resizebox shrink [rotcert-specific]
Only the **TGRS** build was affected (canonical single-column Table 1 was already full-size `\small`
and untouched — verified by identical canonical numeric multiset). The TGRS Table 1 used
`\resizebox{\columnwidth}{!}{…}`, which shrank a 4-column table with wide CI strings to ~7pt, uneven.
- Removed the `\resizebox` entirely.
- Dropped the redundant shorthand "Cell" **column header duplication**: the row-id column now carries
  the short cell names only (DOTA pilot, DOTA-ORCNN, …); the `detector · dataset` mapping moved into
  the caption ("Cells: DOTA pilot = RTMDet-R-l on DOTA-v1.0; …"), which grows the caption per directive
  item 2 and preserves every mapping.
- Set `\footnotesize` + `\tabcolsep 3pt`. Result: single-line point+CI at an even, readable 8pt, fits
  the column (residual overfull 2.6pt, negligible), **page-neutral (13pp)**.
- All eight coverage point estimates + CI bounds + the four in-sample 0.900 values verified present.
  Intermediate attempts rejected on evidence: `table*` full-width promotion cost a page (13→14);
  `\small` overflowed 80pt; stacked point-over-CI misaligned the estimate from its row.

### 3. Table 3 (`tab:diorperclass`) "-" placeholder cells [rotcert-specific + Common item 5]
The G2-recall column's refused cells were bare `--` (en-dash). These are structurally-absent (class
refused by the LTT-HB power floor), so per Common item 5 they become the venue-standard `\textemdash`
with a caption note. 13 cells/file → `\textemdash`; caption note `\texttt{--}=refused` →
`\textemdash{}=refused`. Both files. Renders as em-dash (verified page 8 canonical).

### 4. Table 10 (`tab:regime`) "-" cells + float embedding [rotcert-specific + Common items 5,7]
- **"-" cells:** the compact/elongated "no powered split" cells were `\texttt{--}`; → `\textemdash`
  (4 cells/file) + caption note `\texttt{--}=regime with no powered split` →
  `\textemdash{}=regime with no powered split`. Both files.
- **Embedding:** audited — Table 10 is NOT stranded after the bibliography in either venue. Canonical:
  embedded on p18 inside §Results, before §Limitations and well before references (p20). TGRS: renders
  at the top of p12, in the two-column flow, above the References head on the same page (before, not
  after, the bibliography). Both compliant with Common item 7; no float-placement change needed.

### 5. Common-item audits (all clean, no edits needed)
- **"?" artifacts (item 4):** `pdftotext | grep "?"` on both compiled PDFs → zero. No broken
  `\ref`/`\cite`/author fields.
- **Prereg/plan leakage in captions (item 6):** audited every caption for plan file names, dates-of-
  plan, roster/bookkeeping counts → none. rotcert captions use normal scientific provenance ("from the
  frozen … record", "post-hoc", "not preregistered") which is legitimate; no internal-plan dates like
  materials' "2026-07-13" leak. (Source-provenance `%` comments with dated dir names are LaTeX comments,
  never typeset.)
- **Annotation colour (item 3):** fig1/fig2 have zero colored text nodes (grep clean); fig0 fixed above.
- **Tables after references (item 7):** every float in both venues embedded in main content flow.
- **Figure geometry (item 8):** fig0 native size fits both venues; fig1/fig2 unchanged (already
  content-audited in the tablefix pass; no dead-white-space or colored-text issues).

### 6. Page/volume norms (Common item 1)
**TGRS live norm (verified 2026-07-16, GRSS Information-for-Authors + IEEE APC list):** mandatory
Overlength Page Charge **$230/printed page from page 11** for papers submitted after 2026-01-01
(GRSS members $200/page); optional sustaining charge $110/page for the first 11. No hard page limit.
**Our TGRS build is 13pp** → 3 overlength pages (≈ $600–690 OPC). This pass was **page-neutral**
(13pp before and after; the Table 1 resizebox fix added no pages). Reducing OPC from 13→10pp would
require substantive content removal: a redundancy scan (repeated "refuses/prints the floor",
"as designed" phrasings) found each occurrence is context-specific (intro vs methods vs per-cell
result), not true duplication — cutting them removes argument reviewers rely on, violating the
"no result loss" rail. **The OPC-vs-trim tradeoff is a content decision left to the user** (consistent
with the standing note in the tablefix report and the user's ledger); no prose was cut in this pass.

## Rails
- **Zero result-number changes.** Canonical numeric-token multiset: **identical** (backup vs final).
  TGRS: only additions — +1 "95" (CI-level label added to the Table 1 column header) and +2 "1.0"
  (from "DOTA-v1.0" in the new Table 1 caption mapping); no coverage / CI-bound / count value altered
  or dropped (all eight point+CI pairs and four 0.900 verified present).
- `latexmk` exit **0** on canonical + TGRS; **0** undefined / multiply-defined refs on both.
- Backups in place; edit classes limited to the fig0 rebuild, dash→`\textemdash` conversions, and the
  TGRS Table 1 resize/caption restructure.

## Page counts
| | before (pre-d3) | after |
|---|---|---|
| canonical `paper.pdf` | 23 | **23** (±0) |
| TGRS `paper_ieeetran.pdf` | 13 | **13** (±0) |

## Ready score (submission readiness, post-fix)
**rotcert: 90/100.** Certification results frozen and consistent across both venues; figures and tables
now clean and readable; no broken refs, no placeholder-hyphen or caption-leakage artifacts. Top
remaining gaps: (1) **TGRS 13pp → ~$600–690 OPC** — a user content-trim/pay decision, not a defect;
(2) DIOR-R AOPG mAP reproduction stays prereg-gated (disclosed in-paper, not a blocker); (3)
GitHub+Zenodo archival + DOI still `TODO-USER` in the data-availability statement (user publication
action). None are visual/layout defects.

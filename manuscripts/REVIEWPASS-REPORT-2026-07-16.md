# rotcert review-pass report — 2026-07-16

Applies the user review directive (`docs/records/REVIEW-DIRECTIVE-2026-07-16.md`) to both
the canonical `paper.tex` (journal-agnostic `article`) and the venue kit
`tgrs/paper_ieeetran.tex` (IEEEtran two-column), kept in lockstep.

Primary routing recommendation: **ISPRS J. Photogrammetry & Remote Sensing** (primary),
**IEEE TGRS** (fallback). Both venue limits verified live (below).

## Integrity rails (§8)
- Backups written before any edit: `paper.tex.bak-pre-reviewpass`,
  `tgrs/paper_ieeetran.tex.bak-pre-reviewpass`, `refs.bib.bak-pre-reviewpass`,
  `tgrs/refs.bib.bak-pre-reviewpass`.
- **Zero result-number changes — proven.** Multiset of all non-comment numeric tokens is
  identical between backup and final for BOTH files, except:
  - canonical: `+1` occurrence of `0.52` (the new `p{0.52\textwidth}` column width in Table `tab:g2`);
  - both files: the `2026`/`07`/`15` tokens of the `dior_perclass_2026-07-15/results.json`
    date-string dropped when that results-dir file reference was re-expressed in prose (leakage removal, §2).
  No coverage / q̂ / count / ratio / p-value / CI token changed. `\mathbf{...}` table-cell number
  highlights are untouched.
- `latexmk` exit **0** on both; **0** undefined/multiply-defined refs or citations on both.
- Page counts before→after: canonical **18 → 18**; TGRS **11 → 11** (no bloat from the `table*` conversions).

## §1 Style / tone — bold + italic
Before → after occurrence counts:

| macro | canonical | TGRS |
|---|---|---|
| `\emph{}`  | 132 → **0** | 126 → **0** |
| `\textbf{}` | 45 → **34** | 41 → **31** |

- **Italics:** every `\emph` was unwrapped to plain text. None were first-use term definitions
  or standard Latin (there were none of either); all were tone emphasis on ordinary words
  (`\emph{not}`×16, `\emph{refuses}`×5, `\emph{descriptive}`, `\emph{given}`, …). Result: italics
  now appear only in genuine math/technical contexts, not as prose emphasis.
- **Bold:** removed 11 (canonical) / 10 (TGRS) mid-sentence emphasis bolds — the "AI tone"
  offenders: bold *sentences* mid-paragraph ("The oriented case is not a mechanical port.",
  "Coverage is evaluated out-of-sample", "The preregistered test thus certifies…", "At the pooled
  level…", "This explains the DOTA wash…", "The prediction is reversed."), single-word bolds inside
  captions (`out-of-sample`, `Coverage-matched`, `Regime-conditional`, `matched nominal α`), and two
  bolded table column headers (`OOS R=20 coverage`, `matched`).
- **Retained bold (intentional):** the paper title; the unused `\todo{…GATED…}` macro definition
  (never rendered); and ~32 run-in paragraph headers (`\textbf{Contribution.}`,
  `\textbf{Degeneracy disclosure…}`, etc.). Run-in bold headers are a standard IEEE / journal
  convention, explicitly permitted by §1 ("where a journal convention genuinely expects it"), and
  removing them would hurt readability. Verdict-style words (FAIL etc.) are not bolded anywhere.

## §2 Source-code leakage
`\texttt{}` count 22 → **12** (canonical), 21 → **11** (TGRS). Re-expressed in natural language
(prose), source comments retained as invisible provenance:
- `mmdet.FilterAnnotations` / `min_gt_bbox_wh` / `filter_empty_gt` → "an annotation-filtering step
  that drops degenerate zero-area … ground-truth boxes (removing boxes below a minimum width or
  height, and images left with no valid ground truth)".
- `mondrian_field=class` → "grouping on object class".
- `results.json` (caption) → dropped; `dior_perclass_2026-07-15/results.json` (caption + body ×2) →
  "the frozen per-class conditional-coverage record" / "a single four-cell … record".
- exploratory `\texttt{iou}` construction → "an exploratory construction we also provide".
- `\texttt{expressway-service-area}` / `\texttt{expressway-toll-station}` (provenance note) →
  de-monospaced to plain class names (also fixed a pre-existing 20 pt overfull; see §5).

**Retained `\texttt` (permitted / not code artifacts):** `rotcert` (the shipped tool/repository name —
allowed by §2 exception), `le90`/`le135` (established mmrotate long-edge angle-convention labels, domain
terminology not source identifiers), and the `O` / `R` / `--` Table `tab:diorperclass` legend symbols
(literal cell markers). Flagged here for the record; judged in-scope to keep.

## §3 No appendix / compactness
Neither paper has an appendix — all results already live in the main body (nothing to merge). No
`\clearpage`/`\newpage`, no half-empty pages. Abstract **248 words** (≤ 250, within ISPRS cap).

## §4 Figures
**Both papers contain zero figures** — every float is a `table`/`table*`. There is nothing to
regenerate through the R/TikZ→PDF workflow, and no `figures-src/` or Makefile is required because no
figure sources exist. Not a silent skip: the paper is table-only by construction.

## §5 Tables / floats — column-span discipline
- **All `\ref`/`\cite` resolve** (0 undefined, both files); float order matches first-in-text-reference
  order (audited table-by-table in the canonical; identical logical order in TGRS).
- **TGRS `table*` conversions:** `tab:configA`, `tab:configB`, `tab:seedvar` were single-column
  `table` forced through `\resizebox{\columnwidth}` — for these 5-column CI-bearing tables that drove
  effective text below the ~7 pt floor. Converted to double-column `table*` at natural `\small` width
  (no resizebox). `tab:g1oos` (4 narrow columns) keeps its mild single-column resizebox (stays ≥8 pt).
  TGRS `table*` count 6 → 9; remaining `\resizebox{\columnwidth}` = 1 (`tab:g1oos`, intended).
- **Canonical** keeps `\resizebox{\textwidth}` on its three wide 6-column tables: their natural width
  slightly exceeds `\textwidth`, so the box scales *down* a few percent (stays ~10–11 pt, readable) —
  removing it would overflow. Left as-is by design.
- **Pre-existing overfull hboxes fixed (canonical):** `tab:g2` example-classes column overflowed 51 pt
  → gave it a wrapping `p{0.52\textwidth}` column (added `\usepackage{array}`); the provenance-note
  paragraph overflowed 20 pt → resolved by de-monospacing the two expressway class names (§2). Final:
  **0 overfull ≥10 pt** (canonical); TGRS worst overfull 1.1 pt (negligible).

## §6 References — live verification
All **20** bibliography entries independently re-verified against live sources (arXiv, doi.org,
Crossref, IEEE Xplore, Springer, PMLR, Royal Holloway portal): **20 VERIFIED, 0 fixed, 0 flagged.**
- Future-dated arXiv ids all resolve: `ding2026rtoriented` (2603.15497), `ries2026probabilistic`
  (2605.07549), `andeol2025seqcrc` (2505.24038).
- Recent journal DOIs all confirmed: `eavdetr2026` (10.1016/j.isprsjprs.2026.02.009, ISPRS J. vol 233
  pp 575–587), `uanet2024` (10.1109/TGRS.2024.3361211, art. 6261211), `ssel2024`
  (10.1109/TGRS.2024.3349415), `degrancey2022` (10.1007/978-3-031-14862-0_23).
- Sub-claim confirmed: `ding2026` Table reports Oriented R-CNN (R-50) at 64.30 AP50 on DIOR-R,
  matching the paper's disclosed same-method anchor.

## §7 Code archiving
No `docs/records/CODE-ARCHIVE-POLICY-2026-07-16.md` existed at review time, so the baseline was applied:
Data/Code Availability statement in BOTH files now states public **GitHub repository + versioned
**Zenodo DOI** on acceptance, with a per-venue mirror noted **pending** ("e.g. Code Ocean for an IEEE
submission … per the target venue's policy"). Actual URL/DOI left as `TODO-USER` (not fabricated),
consistent with the paper's existing placeholder convention. **User action:** confirm the per-venue
policy (Code Ocean for TGRS?) and fill the repo URL + Zenodo DOI.

## Venue limits (verified live, retrieved 2026-07-16)
- **ISPRS J. P&RS** (primary; Elsevier, ISSN 0924-2716): abstract **≤ 250 words**; **no hard page/word
  limit** for full research articles (conciseness expected); up to 6 keywords. Paper abstract = 248
  words (OK). Source: ScienceDirect guide-for-authors
  (https://www.sciencedirect.com/journal/isprs-journal-of-photogrammetry-and-remote-sensing/publish/guide-for-authors
  — automated fetch 403; 250-word cap corroborated by the ISPRS author guidelines,
  https://www.isprs.org/documents/orangebook/doc/ISPRSguidelines_authors_abstract_final.doc).
- **IEEE TGRS** (fallback; GRSS): **no hard page cap, but mandatory Overlength Page Charge from page 11**
  ($230/pg; $200 GRSS members) — practical target ≤ 10 printed pages. Abstract per IEEE convention
  (~250 words; no numeric cap stated). Source:
  https://www.grss-ieee.org/publications/author-resources/tgrs-information-for-authors/.
  **Flag:** the TGRS build is **11 pages** — one page into overlength. User decision: trim ~1 page to
  land at 10, or budget the single-page OPC.

## Net status
Both files: latexmk exit 0, 0 undefined refs, zero result-number changes, backups in place.
Per directive §9, the DIOR-R seed-variance subsection is still to be added later; this pass was run now
and its edit classes will carry to that integration.

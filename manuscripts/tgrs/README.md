# TGRS (IEEEtran) submission package

IEEE Transactions on Geoscience and Remote Sensing two-column port of the canonical manuscript
`../paper.tex`. Ported **2026-07-13** from the review-verified `paper.tex` state (target venue per
`../VENUE-FIT-MEMO.md`: TGRS primary, JSTARS fallback — SCIE-indexed, ScholarOne, no OpenReview).

## Canonical source
`../paper.tex` (journal-agnostic `article` class) remains the single source of truth for content and
numbers. `paper_ieeetran.tex` re-expresses the **same** content in IEEEtran journal format; it is a
**packaging port, not a rewrite**, and does not modify the canonical file. Every quantitative claim
keeps its `% source:` provenance comment pointing at the on-disk result JSON (frozen
`pilot_oos_2026-07-11/`, `dior_cert_results_2026-07-11/`, `holm8_2026-07-12/`). All content claims are
byte-faithful to `../paper.tex`.

## Build
```
latexmk -pdf -interaction=nonstopmode paper_ieeetran.tex
```
Status at port time: **clean** — `latexmk` exit 0, **6 pages** two-column, 0 undefined
citations/references, all 18 bibliography entries resolved. One overfull hbox at 1.1 pt (a reference
title; well under the ~12 pt bar — cosmetic, no fix needed). One cosmetic font-shape substitution
warning (`T1/ptm/m/scit`, Times italic small-caps auto-substituted) — harmless.

## Class / style provenance
- `IEEEtran.cls` and `IEEEtran.bst` are provided **system-wide by TeX Live** (`texlive-publishers`);
  they are **not vendored** in this directory. On a system without them, install `texlive-publishers`
  or fetch from CTAN: `https://ctan.org/pkg/ieeetran`. Standard `\documentclass[journal]{IEEEtran}`.
- Bibliography style `\bibliographystyle{IEEEtran}` (numeric). Citations use `natbib` in
  `[numbers,round]` mode so the canonical `\citep`/`\citet` macros render as IEEE-style bracketed
  numbers.

## Bibliography
`refs.bib` is a **snapshot copied from `../refs.bib` on 2026-07-13** so this directory is
self-contained for submission (includes today's fixes: `degrancey2022` verified Springer LNCS /
SAFECOMP 2022 WAISE; `vovk2003mondrian` as `@techreport`; `eavdetr2026` authors, volume, pages, and
DOI **verified 2026-07-12 against the Crossref primary record**). If the upstream bib changes,
re-copy it. There is no `shared.bib` for rotcert — all references live in `refs.bib`.

## Holm-8 demotion (decision taken 2026-07-12)
The frozen single-split Holm-8 family is **demoted from confirmatory evidence to a preregistered
descriptive outcome** across the canonical manuscript, this IEEEtran port, and the cover letter. The
preregistered family was executed exactly as frozen and its outcome (GWD region smaller in 8/8 cells)
is reported, but **not** as statistical confirmation: the single-split test is structurally degenerate
(one set-size scalar per cell; permutation p deterministic at the 1/2001 floor for any negative effect;
zero-width bootstrap CI; effective n=1), so the nominal Holm-adjusted p=0.004 carries no evidence
strength. Because the frozen 8/8 and the R=20 across-split analysis both compare at matched **nominal**
α (where the union-bound baselines over-cover), the **primary quantitative characterization of
efficiency** is now the post-hoc, **coverage-fair coverage-matched ablation**
(`../coverage_matched_2026-07-13/results.json`): re-tuning each baseline to GWD's realized coverage, the
size advantage is largely baseline over-coverage — it survives only against the like-for-like naive
Bonferroni baseline on DIOR-R (≈0.8×, 20/20 splits, both detectors), is a wash on DOTA, and reverses
against the coarser axis-aligned hull throughout. The R=20 across-split result is reported as
**split-sensitivity** of the nominal contrast (GWD smaller on 20/20 splits, worst single-split ratio
0.974) **with no p-value** — `splits.py` reshuffles one fixed scene set, so the 20 splits are dependent
and an exact sign test would overstate the evidence. **No preregistration amendment is made** and no
analysis is promoted to confirmatory. The "open decision (demote vs amend) reserved to the user"
language is removed everywhere. No p-value is presented as evidence strength anywhere in the kit.

## What changed vs the single-column canonical (class change only)
- **Double-column (`table*`)**: three wide tables span both columns — G2 certified-recall
  (`tab:g2`, long example-class cells), the naive-vs-GWD angle-regime audit (`tab:naive`, 6 columns),
  and the exploratory R=20 across-split robustness (`tab:r20holm`, 5 columns).
- **`\resizebox{\columnwidth}{!}{...}`**: `tab:g1oos` (G1 out-of-sample coverage, with inline CIs) is
  scaled to column width. `tab:g1mondrian` (15-class per-class certificates) fits a single column
  natively with `\small`.
- **Preamble reflow aid**: `\renewcommand{\_}{\textunderscore\allowbreak}` lets long `\texttt{...}`
  result-JSON paths break at underscores inside the narrow columns (affects text-mode `\_` only,
  never math subscripts).
- **IEEE front matter added**: `\IEEEPARstart` opener, `\markboth`, `\begin{IEEEkeywords}`,
  `\IEEEpeerreviewmaketitle`. **Biography blocks intentionally omitted** (single-author draft; add at
  acceptance if the editor requests).
- **Abstract** carries the full Holm-8 **demotion** framing (frozen family = preregistered
  **descriptive** outcome, GWD smaller 8/8 at nominal α, **not** confirmatory; single-split degenerate at the
  1/2001 floor; post-hoc **coverage-matched ablation** as the primary efficiency characterization — the
  size gap is largely baseline over-coverage, surviving only vs the like-for-like naive baseline on DIOR-R
  ≈0.8×; no prereg amendment; no p-value on the dependent R=20 splits), the
  in-sample→out-of-sample correction, the conditional (not universal) "naive breaks" claim, and the
  executed-and-disclosed AOPG mAP item. The demotion re-framing runs the abstract slightly over the strict 250-word
  IEEE target (~270 words); **give it a final trim at submission if the target venue enforces a hard
  250-word limit.**
- Nothing content-bearing was dropped: all 5 tables, the abstract, keywords, every section, and the
  Data/Code Availability statement port intact. No figures are used by this manuscript.
- The canonical `\todo` macro was **unused** (no red `[GATED]` renders existed in the body); the AOPG
  reproduction result is honest prose in `../paper.tex` and is carried over as prose. Nothing was deleted.

## TODO-USER before submission
- **Author/affiliation block**: complete the `\author{...}`/`\thanks{...}` block in
  `paper_ieeetran.tex` — full name, affiliation (department, institution, city, country), contact
  e-mail, ORCID (0009-0001-8329-0108 present), and received/revised dates / funding. Placeholders are
  marked `TODO-USER`.
- **EAV-DETR reference (`eavdetr2026` in `refs.bib`)**: author list, volume, pages, and DOI are now
  **verified (2026-07-12) against the Crossref primary record** — Zuo, Haoyu; Ning, Minghao; Shu,
  Yiming; Huang, Shucheng; Sun, Chen; *ISPRS J. Photogramm. Remote Sens.*, vol. 233, pp. 575–587,
  2026; DOI 10.1016/j.isprsjprs.2026.02.009. The placeholder author list is retired. **Still open (the
  one user-attended item):** the Crossref record carries **no abstract**, so the PA-MCP estimand could
  not be confirmed from the primary record; the Related Work "closest neighbour" caveat keeps its
  existing hedged phrasing, and a full-text differentiation remains a named pre-submission action.
- **AOPG mAP reproduction**: executed and disclosed 2026-07-13 (`aopg_repro_2026-07-13/`). The in-house
  detectors reproduce DIOR-R test mAP 62.61 (Oriented R-CNN 1x) / 68.36 (RTMDet-R-l 3x) vs the AOPG
  published 64.41; the AOPG table has no same-method row, so the ±0.5 identity tolerance is not met
  cross-method (−1.80, +3.95). A **same-method external anchor** now disclosed: a recent benchmark table
  reports Oriented R-CNN R-50 at **64.30** mAP on DIOR-R (Ding et al. 2026, arXiv:2603.15497,
  `ding2026rtoriented`), essentially AOPG's 64.41 and within 1.7 points of our 1× 62.61 at a matching
  12-epoch schedule; the ~67–68 leaderboard band is transformer detectors, **not** Oriented R-CNN (the
  red-team's ~67.87 same-method premise does not verify). Both scores fall in the published DIOR-R band;
  disclosed, not claimed as passing; certification uses reproduced scores only. No preregistration-gated
  slot remains. Also disclosed in the manuscript training setup: the ORCNN AdamW + FilterAnnotations
  zero-area-GT fix, and RTMDet deployed at its final epoch_36 (68.36) over the higher-val epoch_24
  (69.65) to avoid checkpoint selection on the evaluation split.
- **Cover letter date and closing block** (`cover_letter.md`): fill the `TODO-USER` date, affiliation
  line, funding statement, and (optionally) three suggested reviewers.
- **Compile check**: re-run `latexmk -pdf -interaction=nonstopmode paper_ieeetran.tex` after editing
  the author block; confirm exit 0 and 0 undefined references.

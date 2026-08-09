# PROPOSED-TEX — AOPG reproduction-gate slot (drop-in for the polish agent)

Compute-only proposal. **I did not edit any manuscript.** Below is drop-in LaTeX
replacing the "preregistration-gated / not reported" placeholders now that the
gate has been executed (2026-07-13). Verdict is **FAIL under the frozen ±0.5
identity-reproduction rule**, reported as a premise-limited disclosure (the AOPG
DIOR-R table has no Oriented R-CNN / RTMDet row; reproduced scores are sane and
disclosed; no certification claim leans on a paper-quoted number). Numbers:
Oriented R-CNN 1x = **62.61**, RTMDet-R-l 3x = **68.36** (DIOR-R test split,
single-scale, no-TTA, deployed checkpoints); AOPG DIOR-R reference = **64.41**.

Both manuscripts carry near-identical placeholder sentences. Apply the matching
replacement at each cited line.

---

## A. Canonical — `reliability-commons/tools/rotcert/manuscripts/paper.tex`

### A1. Body, protocol paragraph — line **209**
Replace:
> The DIOR-R AOPG-table mAP reproduction is preregistration-gated and \emph{not} reported here.

with:
```latex
The DIOR-R AOPG-table mAP reproduction gate (preregistered, K3) was executed
post-freeze: the in-house detectors reproduce DIOR-R test-split mAP of $62.61$
(Oriented R-CNN, 1x) and $68.36$ (RTMDet-R-l, 3x), against the AOPG published
DIOR-R reference of $64.41$ \citep{cheng2022diorr}. The AOPG table lists neither
detector as a same-method row, so the $\pm0.5$ identity-reproduction tolerance is
not met by either ($-1.80$, $+3.95$); both reproduced scores nonetheless fall in
the published DIOR-R band (Oriented R-CNN between Gliding Vertex and RoI
Transformer at 1x; RTMDet-R-l above the entire 2021 table). We disclose the gap
rather than chase it, and certification uses reproduced scores only, never
paper-quoted ones.
```

### A2. Abstract-region comment — lines **59--60**
Replace:
> Ships as \texttt{rotcert}; no AOPG / mAP reproduction is claimed (gated).

with:
```latex
Ships as \texttt{rotcert}; the AOPG-table DIOR-R mAP reproduction is executed and
disclosed (no $\pm0.5$ identity match; reproduced scores in the published band),
not claimed as a passing gate.
```

### A3. Discussion — lines **474--475**
Replace:
> The DIOR-R AOPG mAP reproduction remains preregistration-gated.

with:
```latex
The DIOR-R AOPG mAP reproduction gate was executed (reproduced test mAP $62.61$ /
$68.36$ vs.\ AOPG $64.41$; $\pm0.5$ not met cross-method, disclosed).
```

### A4. Full-study — lines **484--485**
Replace:
> The one remaining preregistration-gated slot is the AOPG mAP reproduction; all required inputs are on disk.

with:
```latex
The AOPG mAP reproduction gate has now been executed from the on-disk work_dirs
(\S\ref{sec:results}); no preregistration-gated slot remains.
```

### A5. Header comments — lines **12**, **28**
Update the source-note comments (non-rendering) from "AOPG repro stays gated" /
the placeholder note to: `AOPG repro executed 2026-07-13 (aopg_repro_2026-07-13/;
FAIL vs +-0.5 cross-method, disclosed).`

---

## B. TGRS kit — `reliability-commons/tools/rotcert/manuscripts/tgrs/paper_ieeetran.tex`

### B1. DIOR-R extension paragraph — line **232**
Replace:
> The DIOR-R AOPG-table mAP reproduction is preregistration-gated and \emph{not} reported here.

with the **A1** replacement block verbatim.

### B2. Discussion — lines **498--500**
Replace:
> The DIOR-R AOPG mAP reproduction / remains preregistration-gated.

with the **A3** replacement.

### B3. Full-study — line **509**
Replace:
> The one remaining preregistration-gated slot is the AOPG mAP reproduction; all required inputs are on disk.

with the **A4** replacement.

### B4. Source-note comment — line **500**
Update `AOPG repro still gated` → `AOPG repro executed 2026-07-13
(aopg_repro_2026-07-13/RESULTS-AOPG.md; FAIL vs +-0.5 cross-method, disclosed)`.

---

## C. Notes for the polish agent

- `\citep{cheng2022diorr}` is already the DIOR-R citation key in both files; the
  AOPG number 64.41 is from AOPG (arXiv:2110.01931, Table I) / `jbwang1997/AOPG`
  DIOR-R model-zoo — if a distinct AOPG bib key is preferred over the DIOR-R key,
  add one, else the DIOR-R key is acceptable (same authors, same table).
- Keep the abstract within its word budget: if A2's replacement pushes the TGRS
  250-word abstract over, the shorter form "the AOPG-table DIOR-R reproduction is
  executed and disclosed (reproduced scores in the published band; no $\pm0.5$
  match)" is sufficient.
- Do **not** state or imply the gate "passed." The honest, sign-off-sanctioned
  statement is: executed, ±0.5 not met cross-method, reproduced scores disclosed
  and published-plausible, certification on reproduced scores only.
- Full record: `aopg_repro_2026-07-13/RESULTS-AOPG.md` +
  `aopg_repro_result.json`.

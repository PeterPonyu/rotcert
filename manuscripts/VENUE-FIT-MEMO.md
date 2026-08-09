# Venue-fit memo — rotcert (certified reliability for oriented object detection)

**Author:** write-geo (agent). **Date:** 2026-07-11.
**Constraint (user directive):** SCIE-indexed journals only; **no OpenReview** account.
This overrides the design doc's stated targets (`apps-design/05-APP-rotdet-cert.md` §Target venue:
TMLR primary, WACV secondary) — **both TMLR and WACV are OUT** (TMLR is OpenReview; WACV is a conference,
not SCIE-indexed). This memo evaluates the SCIE journal options the directive requires.

## What the paper is (for scope-matching)
An angle-aware **conformal certification** system for oriented (OBB) object detection: a GWD-based
nonconformity score that is continuous across the ±90° angle seam and safe at square aspect ratios, wrapped
into two distribution-free guarantees — **G1** per-detection localization coverage (marginal + angle-regime
conditional) and **G2** an LTT-HB-certified rotated-IoU false-negative-rate bound — plus a head-to-head
audit against naive coordinate-wise / axis-aligned-hull conformal baselines. Evaluated on DOTA (pilot
complete) and DIOR-R (in-house-trained detectors; round in progress). It is a **methods + reliability-audit**
contribution on remote-sensing detectors, not a new detector and not a mapping product.

## Candidates

### 1. IEEE TGRS — *Transactions on Geoscience and Remote Sensing* — **RECOMMENDED**
- **SCIE**, IF ≈ 8.2, ScholarOne (**no OpenReview**). The flagship remote-sensing methods journal.
- **Scope fit: strongest.** Oriented object detection on DOTA/DIOR-R is core TGRS territory — DIOR-R itself
  was introduced in TGRS 2022 (Cheng et al.), and RTMDet-R / Oriented R-CNN / GWD-loss oriented-detection
  work is routinely published and cited here. A distribution-free reliability/UQ layer for these exact
  detectors and benchmarks lands squarely in the readership. Referees will recognize every baseline and
  dataset without preamble.
- **Length:** no hard page limit (overlength charges beyond ~10–12 pages typical); comfortably fits the
  two-guarantee system + audit + appendices.
- **OA fees:** hybrid — optional Open Access APC (~US$2,645); traditional (non-OA) publication is free.
- **Review latency:** first decision typically **12–18 weeks** (3–4 month median); slower than JSTARS but
  the prestige/scope premium is worth it for a methods paper the detection community will cite.
- **Empirical bar:** multi-dataset + multi-detector expected. The DOTA+DIOR-R × RTMDet-R+Oriented-R-CNN
  grid meets it; a single-dataset pilot alone would be under-scoped here (see "what DIOR-R adds").

### 2. IEEE JSTARS — *J. Selected Topics in Applied EO and Remote Sensing* — **alternate (faster)**
- **SCIE**, IF ≈ 5, ScholarOne, fully **Open Access**, first decision ≈ **2–4 months** (faster than TGRS).
- **Scope fit: good but slightly off-center.** JSTARS rewards *applied* EO with clear downstream utility;
  a statistics-forward conformal-certification methods paper is publishable but sits a notch above its
  typical applied remit. Best used as the fallback if TGRS review latency becomes a problem or a TGRS
  decision pushes toward an applications reframe.
- **OA fee:** ~US$2,000 APC (OA is mandatory — it is a fully-OA title).

### 3. ISPRS J. P&RS — *ISPRS Journal of Photogrammetry and Remote Sensing* — **poor fit**
- **SCIE**, highest IF of the three (≈ 12), but the empirical bar is **application-science / mapping-product
  impact** (geoscience outcome, not a statistical-methods contribution). A conformal-certificate paper with
  no mapping deliverable is a weak match; slower cycles compound the risk. Not recommended.

## Recommendation: **IEEE TGRS (primary), JSTARS (fallback)**
Reasoning: TGRS is the community home of oriented detection and of DIOR-R itself, is SCIE + ScholarOne
(satisfies the no-OpenReview constraint), imposes no page limit, and treats a reliability/certification
methods contribution on DOTA/DIOR-R detectors as in-scope and citable. Its heavier empirical bar is exactly
what the completed multi-dataset grid satisfies. JSTARS is the faster, lower-bar fallback (mandatory OA,
~2–4 month decisions) if latency or an applications reframe is preferred. ISPRS is a poor fit for a
methods/audit paper. Packaging note: TGRS and JSTARS both use `IEEEtran`; the manuscript is drafted in the
journal-agnostic `article` class (like the portfolio's other canonical sources) and an `IEEEtran` port is a
one-pass packaging step at submission, as done for the geospatial-FM paper.

## What the DIOR-R round adds to venue strength
- **Turns the single-dataset pilot into the multi-dataset claim TGRS referees expect.** The DOTA pilot alone
  (one detector, one dataset) is a proof-of-concept; DOTA **+** DIOR-R across **two** detectors
  (RTMDet-R-l, Oriented R-CNN) is the breadth a TGRS methods paper needs to argue generality rather than a
  single-benchmark artifact.
- **Demonstrates the certificate transfers across dataset/sensor regimes** (DOTA aerial 1024-crops vs DIOR-R
  800×800 optical single-tiles), directly answering the "does the GWD certificate generalize?" referee
  question.
- **Establishes an independent second detector family** (two-stage Oriented R-CNN vs single-stage RTMDet-R),
  testing score-agnosticism of the conformal wrapper.
- **Note (honest):** DIOR-R needed **in-house training** (no license-clean DIOR-R checkpoint exists
  anywhere); its **AOPG-table reproduction gate stays prereg-gated** and is not presumed here. The venue
  strength comes from the certification results on the trained detectors, not from an mAP-leaderboard claim.

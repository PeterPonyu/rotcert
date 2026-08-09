# RED-TEAM REPORT — rotcert manuscript skeleton (`manuscripts/paper.tex`)

**Reviewer stance:** adversarial, goal = reject. **Scope:** attacks what EXISTS in the current draft
(DOTA-v1.0 pilot, real numbers) — the `\todo{}`-marked DIOR-R slots are not attacked as absent, but
the DIOR-R *substrate that has since landed* is used to check whether the draft's central number
survives contact with a proper held-out protocol. No paper edits made.

---

## FATAL-1: The headline G1 coverage number (0.900) is not measured on held-out data — it is a
tautological echo of the calibration formula, and this is unstated anywhere in the paper

**Claim under attack.** Abstract / §4.1 / Table 1: "the GWD certificate meets its target: marginal
coverage is 0.900 (scene-clustered 95% CI [0.879, 0.916]; n=54,131 detections, 455 scenes) at
α=0.10." §Experimental Setup states: "the pilot reports **a single scene-level calibration/
evaluation split**." This is the paper's single most-quoted empirical result — it's in the abstract,
the intro, and the results.

**Evidence.**
1. The boot script that produced this number (`orchestration/next_boot_rotcert.sh:155,163,170`)
   defines **one** file, `matched="$pilot_dir/matched.jsonl"`, and passes it as `--matched` to BOTH
   `rotcert calibrate` (which writes `cert_gwd.json`) and `rotcert audit` (which writes
   `audit.json`, the source of the 0.900 number). There is no cal/eval split anywhere in the pilot
   invocation — the same 54,131 matched true-positive detections are used for both roles.
2. `rotcert/cli.py::_cmd_audit` (lines 267-301) confirms this is not an accident of the boot
   script alone: the function itself takes ONE `--matched` file, calibrates a quantile from it
   (`_certify.g1_calibrate(matched, ...)`), and then computes "coverage" by checking that SAME
   `matched` list against the calibrator it was just fit from. There is no internal split, no
   held-out subset, nothing. `cert_gwd.json`'s per-class `n_cal` column sums to exactly 54,131 —
   the identical number reported as the evaluation-set `n` in Table 1 and the abstract.
3. **Arithmetic proof this is not approximately but *exactly* tautological.** Split conformal's
   quantile is the k-th order statistic with k = ⌈(1-α)(n+1)⌉. For n=54,131, α=0.10:
   k = ⌈0.9 × 54,132⌉ = 48,719. Checking what fraction of the SAME n=54,131 points that quantile
   covers gives 48,719/54,131 = **0.9000203210729526** — which matches the paper's reported point
   estimate **to all 16 reported significant figures**. This is not "coverage that happens to be
   near target because split conformal generally works" — it is the mechanical, data-independent
   output of evaluating an empirical quantile's own defining order statistic against the sample it
   was computed from. **Any nonconformity score whatsoever — GWD, naive-coordinate, or literally
   Gaussian noise — would produce a number this close to (1-α) under this procedure**, because the
   check has no dependence on whether the score is any good. As currently measured, this number has
   **zero statistical power to distinguish the GWD score from an arbitrary/broken one.**
4. The per-stratum breakdown (Table 1's interior/boundary/square rows) is *somewhat* less vacuous —
   the global order statistic doesn't mechanically force each subgroup individually to ≈90% — but it
   is still computed via `covered_rows` built from the same in-sample `matched` list (same
   `_cmd_audit` call), so it reflects self-consistency of the calibration sample with itself, not
   generalization to unseen scenes.
5. **The paper's own Limitations section (§5) does not disclose this.** It discusses "Pilot scale...
   a single scene-level split," "Frozen-checkpoint variance," and "Exchangeability approximation" —
   all honest caveats about *other* things — but never states that calibration and "evaluation" in
   this pilot are literally the same 54,131 detections. A reader is left to believe a genuine
   held-out check was performed.

**Hostile argument.** "Your headline number, quoted in the abstract as evidence the method 'meets its
target,' is definitionally forced by the calibration formula and carries no information about
whether GWD is a well-designed score. I can compute this exact number without running your detector,
your matcher, or your GWD distance at all — I only need n and α. This is either a serious
methodological oversight or, if intentional, a materially misleading presentation of what was
measured. Either way, the paper's stated protocol ('a single scene-level calibration/evaluation
split') does not match what the code did."

**Is this fixable, and does the method actually survive?** Yes, on both counts, per evidence already
sitting in the repo (not yet in the paper): `dior_cert_results_2026-07-11/orcnn/r20_coverage.json`
runs a **genuine out-of-sample R=20 scene-blocked cal/match/eval protocol** (40/20/40 split,
split-seed = repeat index) on the newly-landed DIOR-R substrate and reports **mean coverage 0.9001,
BCa 95% CI [0.8985, 0.9019]** — i.e., the *held-out* number lands essentially on top of the in-sample
number. That is genuinely good news for the method, but it is evidence that does **not yet appear
anywhere in this manuscript**, and it was produced for DIOR-R/Oriented R-CNN, not for the DOTA/
RTMDet-R-l pilot this draft actually reports. The paper cannot claim a validated 0.900 result until
an equivalent held-out check is run for the arm it is reporting.

**Severity: FATAL.** As written, the paper's central quantitative claim is not evidence of anything
about the method — it is an algebraic identity misdescribed as an empirical measurement, and the
paper does not disclose this. This must be fixed (either by re-running the DOTA pilot's audit through
a genuine cal/eval split, matching the `three_way_scene_split`/R=20 machinery that already exists in
`rotcert/splits.py` but was never invoked for this pilot, or by explicitly re-deriving the DOTA number
via the same protocol `r20_coverage.json` used for DIOR-R) before any of Table 1's numbers can be
presented as validating anything.

**Remediation.** (1) Re-run `rotcert calibrate` / `rotcert audit` through `three_way_scene_split`
(already implemented, `cal_frac=0.4`/`match_frac=0.2`, eval=0.4 — just never wired into the pilot
boot script) or the R=20 protocol DIOR-R's `r20_analysis.py` already demonstrates; report the
held-out number as the headline, with the in-sample number (if kept at all) explicitly labeled as a
sanity check, never as the validation. (2) Add an explicit sentence to §Experimental Setup and
§Limitations stating what was actually done in the current pilot, if the in-sample number is kept in
any form pending the re-run. (3) Do this for the DOTA pilot specifically — the DIOR-R fix does not
retroactively fix the DOTA claim the abstract makes.

---

## MAJOR-2: The paper's central motivating claim — naive coordinate-wise CP under-covers near the
angle seam / at square boxes — is asserted in the Introduction and never tested with data

**Claim under attack.** §1: "A conformal interval built coordinate-wise on θ neither wraps nor stays
defined at squares, so it under-covers wrapped ground truth while paying for wraparound outliers with
an inflated, Bonferroni-corrected quantile." This is the entire reason GWD is needed instead of a
simpler score — it's the paper's motivating hook.

**Evidence.** §4.3 ("efficiency vs. naive baselines") only reports **average SET SIZE** for
naive-coordinate vs. GWD (360.7 px² vs. 375.5 px², an efficiency/tightness comparison), not
coverage broken down by angle regime for the naive baseline. The regime-conditional coverage
breakdown that WOULD test the under-coverage claim ("the boundary-conditional coverage audit tests
whether naive CP under-covers the near-seam and near-square strata") is explicitly named in §3
("Baselines and the head-to-head") as part of the **confirmatory** audit, and deferred to §6
("Full-study design (in progress)") / the `\todo{}`-marked Holm-8 table. So the paper states the
naive-CP failure mode as established fact in the Introduction, then never shows the number that would
demonstrate it in this draft.

**Hostile argument.** "You've told me why I should care about this paper in your first page, and then
never shown me the experiment that would prove it. Is 'naive CP under-covers near the seam' a logical
consequence of coordinate-wise CP ignoring periodicity (in which case, make the argument rigorously
and cite it as a structural fact, not an empirical claim awaiting confirmation), or is it an empirical
claim your own Methods section says needs a dedicated audit (in which case, you cannot assert it in
the Introduction as if already shown)? You can't have it both ways."

**Severity: MAJOR** (compounds with FATAL-1: even the descriptive GWD-vs-naive comparison that does
appear is subject to the same in-sample/self-referential measurement, so a reader cannot even
partially credit the efficiency numbers as held-out evidence).

**Remediation.** Either (a) demote the Introduction's claim to "a structural argument, formalized
below" and give the periodicity argument as a short proof/construction rather than an assertion of
fact, or (b) pull the boundary-conditional naive-coverage number forward from the deferred confirmatory
audit into this draft (cheap: it's the same pipeline already run for GWD, just pointed at
`cert_naive-coord.json`) so the paper's hook is actually evidenced.

---

## MAJOR-3: A near-scoop was not found in the current Related Work — EAV-DETR (ISPRS J. Photogramm.
Remote Sens., 2026) combines oriented detection with Mondrian conformal prediction

**Evidence.** A 2026 scoop search (WebSearch, this pass) surfaced **EAV-DETR: Efficient
Arbitrary-View Oriented object detection with probabilistic guarantees for UAV imagery**
(ScienceDirect, `S0924271626000602`; code at `github.com/zzzhak/EAV-DETR`). Its third contribution is
explicitly named **"Pose-Aware Mondrian Conformal Prediction (PA-MCP)"**: "utilizes the UAV's flight
pose as a physical prior to generate prediction sets with conditional coverage guarantees" for
oriented object detection. This is thematically very close to rotcert's core framing — oriented
(arbitrary-orientation) boxes + Mondrian-stratified conformal prediction + explicit coverage
guarantees — and is **not cited anywhere in the current Related Work**, which only cites axis-aligned
conformal-detection work (`andeol2023conformal`, `timans2024twostep`, `copley2024copa`,
`angelopoulos2022crc`, etc.).

**What could not be confirmed (paywalled full text; GitHub README gives no method detail):** whether
PA-MCP's "prediction sets" cover box **localization/geometry** (position, size, angle — rotcert's G1
estimand) or something narrower (e.g., existence/classification prediction sets conditioned on pose,
a different guarantee). This materially changes how serious the overlap is:
- If PA-MCP certifies localization regions conditioned on Mondrian strata, it is a genuine
  concurrent/prior system in the same specific niche rotcert claims as novel ("no found prior work
  states a three-way triage certificate... on MVTec AD" is inspect-gate's language, but rotcert's own
  novelty claim — "the angle-aware OBB conformal score" — would need explicit differentiation against
  a paper that already does oriented + Mondrian + conformal).
- If PA-MCP only produces classification-style prediction sets, the overlap is more "shared broad
  theme" than a specific scoop, and rotcert's GWD-score + coupled-G1/G2 specifics likely still stand
  as differentiated.

**Hostile argument.** "Your K6-equivalent scoop search did not surface the one paper in your own
sub-field (oriented aerial detection, 2026, conformal prediction, Mondrian stratification) that shares
the most surface area with your claimed novelty. I don't need to know exactly what PA-MCP guarantees
to flag that you haven't looked hard enough, and I will not take 'we didn't find anything' at face
value until you've actually read this paper."

**Severity: MAJOR**, potentially escalating to a novelty-blocking finding depending on what full-text
review reveals. Cannot be resolved without the paywalled text.

**Remediation.** Obtain the EAV-DETR paper (institutional access or a paid retrieval) before
submission; if PA-MCP does certify localization geometry, add it to Related Work with an explicit
differentiation (GWD's continuity-across-the-seam property vs. PA-MCP's pose-as-Mondrian-stratum
design; DOTA/DIOR-R vs. UAV-specific imagery; the coupled G1+G2 system vs. a single guarantee) —
mirroring exactly how the paper already differentiates against Shen & Liu and CRC-SGAD. If PA-MCP
turns out to be classification-only, add one sentence noting the distinction and move on.

---

## MODERATE-4: G1's per-box guarantee rests on an admitted-but-unstress-tested exchangeability
approximation

**Claim under attack.** §3 ("Scene-level discipline") + §5 ("Exchangeability approximation"): scenes
are the exchangeable/splitting unit; the paper explicitly concedes "G1's per-box guarantee treats the
matched true-positive score as box-level exchangeable, an approximation." This is disclosed, which is
to the authors' credit — the design doc (`apps-design/05-APP-rotdet-cert.md:304-309`) states the same
thing even more explicitly.

**Hostile argument.** "You've named the approximation and defended it by widening the CI (scene
clustering), which fixes variance, not necessarily bias. With ~119 detections per scene on average
(54,131/455), if within-scene detections share a systematic error mode (e.g., one scene's sensor
angle or resolution makes ALL its objects harder to localize), scene-clustering the CI doesn't tell me
whether the POINT ESTIMATE itself is trustworthy, only how much it might vary across re-splits. Show
me a permutation or block-bootstrap check that within-scene correlation doesn't bias the marginal
point estimate, not just widen its interval."

**Severity: MODERATE** — already disclosed, which substantially blunts the attack, but the disclosure
is qualitative ("an approximation we pay for by clustering") rather than quantitatively stress-tested.

**Remediation.** A scene-permutation or leave-one-scene-out sensitivity check (cheap, already-computed
data) showing the point estimate is stable under scene-level perturbation would close this.

---

## MODERATE-5: G2's honest-refusal rate (13/15 DOTA classes refused) is defensible but under-supported
by the evidence actually in this draft

**Claim under attack.** §4.4: G2 certifies only 2/15 DOTA classes (small-vehicle, tennis-court),
refusing the rest, "most... power-floor refusals... the intended behavior at pilot scale."

**Hostile argument.** "13/15 refused is a 13% success rate for your second headline guarantee. 'This
is intentional honesty' is a real defense in principle — refusal-over-silent-rounding is a legitimate
design philosophy — but a reviewer will still ask: at what dataset scale does this system actually
certify a useful FRACTION of classes? If the answer is 'basically never, on any real aerial dataset,'
the honesty framing doesn't rescue G2 from being decorative."

**What the record actually shows (not yet in the paper).** `dior_cert_results_2026-07-11/orcnn/`
already has the answer, and it's a real point in the paper's favor: DIOR-R (far more images/class)
certifies **4/20** classes vs. DOTA's 2/15 — a modest but real increase consistent with "more images
per class → the power floor moves down." This is exactly the evidence that would strengthen the
refusal-discipline defense (showing the floor is data-scale-driven and closes with more data, not a
permanent ceiling on the method), but it is not in this draft.

**Severity: MODERATE.**

**Remediation.** Pull the DIOR-R vs. DOTA G2 certified-class-count comparison forward (even as a
single sentence/footnote ahead of the full DIOR-R table) to preempt the "is G2 useless" objection with
the scaling evidence that already exists.

---

## MINOR-6: The abstract under-signals G1's conditional-on-detection scope

The body text (§3, twice, and §5) is explicit and correct that "G1 is conditional on detection" —
this is well-handled where it counts. The abstract states G1 as "per-detection localization coverage
regions," which an expert reader parses correctly but a skimming reviewer might not. One clause
("G1 says nothing about missed detections; G2 covers that separately") in the abstract would remove
any ambiguity at the point most reviewers actually read.

---

## Scoop search summary (§6 of the assignment)

Two searches: (1) "conformal prediction rotated oriented bounding box object detection Gaussian
Wasserstein" and (2) "conformal prediction oriented object detection aerial DOTA angle coverage
guarantee." No paper doing GWD-as-conformal-score, or the specific coupled G1(localization)+G2(recall)
system on DOTA/DIOR-R, was found. The one substantive near-neighbor is **EAV-DETR** (MAJOR-3, above) —
flagged, not resolved, pending full-text access. Existing cited near-neighbors
(`andeol2023conformal`, `timans2024twostep`, `copley2024copa`, `angelopoulos2022crc`, `shen2025...`
via rotcert's own citations) remain axis-aligned or single-guarantee, correctly differentiated in the
current Related Work.

## Does the Limitations section pre-empt these attacks?

**No, for FATAL-1** — the self-referential audit issue is not mentioned anywhere in §5 despite that
section otherwise being candid about comparable-severity caveats (frozen-checkpoint variance,
exchangeability approximation, pilot scale). **Partially, for MODERATE-4** (exchangeability is named,
just not stress-tested). **No, for MAJOR-2** (the undemonstrated naive-CP-failure claim isn't flagged
as unshown — it's stated as established fact in §1 with no forward pointer clarifying it's untested in
this draft, only that the *confirmatory* head-to-head is pending, which reads as "the comparison is
pending" rather than "the specific under-coverage claim is asserted, not shown"). **N/A for MAJOR-3**
(a scoop can't be pre-empted by a Limitations section; it's a Related Work / literature-currency
issue).

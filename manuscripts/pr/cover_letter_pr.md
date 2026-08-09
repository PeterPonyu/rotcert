Dear Prof. Zoran Duric and Prof. Petia Radeva, Editors-in-Chief,

We submit "Angle-aware conformal certification for oriented object detection in aerial imagery" for consideration in Pattern Recognition.

Oriented boxes have a periodic +/-90-degree angle seam and an unidentifiable angle at square aspect ratios, so coordinate-wise conformalization misrepresents their geometry. We use the established Gaussian-Wasserstein representation as a seam-continuous, square-safe conformal nonconformity score and build two post-hoc certificates around frozen detectors: out-of-sample localization coverage (G1) and a Learn-then-Test false-negative-rate bound (G2) with an explicit power-floor refusal.

The headline contribution is the actionable per-class certificate: for every class the tool either certifies or returns a planning readout — the exact images still needed (a finite power floor), or a statement that no data budget certifies at the requested risk and which looser risk the current data already supports. Around it: a geometry-appropriate set representation and an inspectable two-certificate system with readouts in center-offset pixels, orientation half-extent, and rotated-IoU floor. We do not claim novelty for conformal prediction, Learn-then-Test, Gaussian box representations, or the detectors.

Evaluation spans DIOR-R, DOTA-v1.0, and HRSC2016-MS, five detector architectures, and 20 scene-split recalibrations. All 20 DIOR-R classes certify for G1 across four architectures. The manuscript also reports the limits rather than averaging them away: DOTA's overlapping crops violate the exchangeable-unit assumption and coverage is 1-1.2 points below nominal; G2 refuses most dataset-class cells that do not clear its power floor; coverage-matched compactness is DIOR-R-specific, near parity on DOTA, and reverses on HRSC. The preregistered Holm-8 size comparison is retained only as descriptive because its effective sample size is one scalar pair per cell.

This manuscript fits Pattern Recognition as a representation-and-certification method with explicit validity conditions, refusal behavior, and reproducible evaluation. Detector reproduction gaps and single-seed DIOR-R training are disclosed as limitations.

The manuscript is original, is not under consideration elsewhere, and includes the required declarations.

Sincerely,
Zeyu Fu
fuzeyu09@gmail.com

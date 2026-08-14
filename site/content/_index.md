---
title: "Angle-aware conformal certification for oriented object detection in aerial imagery"
---

<ol class="thesis-lines">
  <li>Oriented detectors emit a point OBB and say nothing distribution-free about how far the true box may lie, or how many objects they silently miss.</li>
  <li>An OBB angle lives on ℝ/πℤ: a ±90° seam and square-box unidentifiability make coordinate-wise conformalization geometrically misspecified, not merely “a bit loose.”</li>
  <li>RotCert conformalizes a <strong>seam-continuous, square-safe Gaussian–Wasserstein (GWD) nonconformity</strong> (GWD is consumed as a score, not claimed as a new detector loss).</li>
  <li><strong>G1</strong> is a Mondrian per-class GWD-ball localization certificate with a finite-sample coverage guarantee, validated out-of-sample over R=20 scene splits.</li>
  <li><strong>G2</strong> is a Learn-then-Test Hoeffding–Bentkus certified rotated-IoU false-negative-rate bound that <strong>refuses</strong> below its power floor instead of emitting a number it cannot back.</li>
  <li>The operator-facing object is a <strong>per-class certificate card</strong>: center-offset radius (px), orientation half-extent (deg) or unconstrained, rotated-IoU floor — or an explicit refuse with a planning readout.</li>
  <li>Evidence is frozen and scoped: ten G1 cells, three aerial datasets (DIOR-R, DOTA-v1.0, HRSC2016-MS), five detector architectures; all 20 DIOR-R classes certify for G1; DOTA overlapping crops are a diagnosed exchangeability stress case (~1–1.2 points below nominal).</li>
  <li>The objects here are those <strong>certificates and refusals</strong> — GWD balls, G1/G2 cards, and a dashed refuse gate — not a detector and not a data portal.</li>
</ol>

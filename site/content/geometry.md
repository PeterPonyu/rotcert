---
title: Why GWD
kicker: Geometry
lede: The angle of an oriented box is not a Euclidean coordinate. Coordinate-wise conformalization is misspecified at the seam and at squares.
---

An oriented box reports heading on ℝ/πℤ. Adding 90° can leave the same rectangle. Near-square boxes make heading unidentifiable. Treating (x, y, w, h, θ) as five independent axes therefore opens a **coordinate gap**: the nonconformity can jump when the representation wraps, even if the physical box barely moved.

Gaussian–Wasserstein distance (GWD) is used here **only as a nonconformity score**. It is seam-continuous and square-safe. RotCert does not claim a new detector loss.

{{< geometry-widget >}}

<figure class="web-fig">
  <img src="{{< static "web-figures/obb-schematic.svg" >}}" width="640" height="280" alt="Two oriented boxes, a GWD disk around the predicted center, and an orange seam marker on the heading circle.">
  <figcaption>Static fallback: elongated seam (left) versus near-square unconstrained heading (right). The green disk is the GWD ball reported as a center-offset radius; the wedge is omitted when heading is unconstrained.</figcaption>
</figure>

Naive coordinate-wise coverage is **regime-dependent** (angle strata), not a universal failure of every baseline. That comparison is scoped; this page does not promote “GWD always smaller.”

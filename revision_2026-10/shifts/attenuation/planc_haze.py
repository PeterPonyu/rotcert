"""Declared synthetic attenuation inserted into a frozen test pipeline right after image loading.

Arithmetic is identical to rotcert/orchestration/expansion_20261001/haze.py (haze_rgb): depth proxy 0.5+0.5*y/(H-1), transmission
t = exp(-beta*depth), airlight 255 on every channel, float64, rint ties-to-even, uint8. It is channel independent, so it
gives the same result on the BGR array mmcv hands over as on RGB. NOT physical haze and NOT measured depth; ground truth,
splits, checkpoints and every other pipeline step are untouched.
"""
import hashlib
import json
import math
import os

import numpy as np

BETAS = (0.6, 1.2, 1.8)
_AUDITED = [0]
POLICY = {
    "schema": "rotcert.synthetic-haze.v1", "betas": list(BETAS), "A": [255, 255, 255],
    "proxy_depth": "0.5+0.5*y/max(H-1,1)", "depth_units": "dimensionless proxy, NOT measured depth",
    "arithmetic": "float64 decoded intensity, rint ties-to-even, uint8", "physical_haze_claim": False,
    "GT_or_split_modified": False, "applied": "in memory, directly after mmdet.LoadImageFromFile, native resolution",
}


def haze_array(img, beta):
    if type(beta) not in (float, int) or not math.isfinite(beta) or beta not in BETAS:
        raise ValueError("Only frozen beta .6/1.2/1.8 is allowed")
    a = np.asarray(img)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3 or min(a.shape[:2]) < 1:
        raise ValueError("Decoded nonempty uint8 HxWx3 image required")
    depth = .5 + .5 * np.arange(a.shape[0], dtype=np.float64) / max(a.shape[0] - 1, 1)
    t = np.exp(-float(beta) * depth)[:, None, None]
    return np.rint(a.astype(np.float64) * t + 255.0 * (1.0 - t)).clip(0, 255).astype(np.uint8)


def source_sha256():
    with open(__file__, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


try:  # registered only inside the detector environment
    from mmdet.registry import TRANSFORMS

    @TRANSFORMS.register_module()
    class HazeSynthetic:
        def __init__(self, beta):
            if beta not in BETAS:
                raise ValueError("Only frozen beta .6/1.2/1.8 is allowed")
            self.beta = float(beta)

        def __call__(self, results):
            before = results["img"]
            results["img"] = haze_array(before, self.beta)
            audit = os.environ.get("PLANC_HAZE_AUDIT")
            if audit and _AUDITED[0] < 50:      # evidence that the transform really ran inside the test pipeline
                _AUDITED[0] += 1
                with open(audit, "a") as handle:
                    handle.write(json.dumps({"img_path": str(results.get("img_path")), "beta": self.beta, "shape": list(before.shape),
                                             "mean_before": float(before.mean()), "mean_after": float(results["img"].mean())}) + "\n")
            return results

        def __repr__(self):
            return "HazeSynthetic(beta=%r)" % self.beta
except ImportError:
    pass

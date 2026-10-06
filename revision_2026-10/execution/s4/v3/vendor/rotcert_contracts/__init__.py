"""Isolated W1 numerical and probability contracts; no shared-source mutations."""
from .exact import (conformal_rank, pooled_threshold, hcp_threshold, scene_max_threshold,
                    crc_object_threshold, crc_object_adaptive_threshold,
                    crc_object_adaptive_thresholds)
from .bounds import empirical_bernstein, tolerance_rank
from .geometry import angle_projection, orientation_half_extent, critical_radius
from .pac import Provenance, representative_pac_g1, fixed_sequence_g1, fixed_lambda_g2, compose_pac
from .pac import fixed_sequence_e2e

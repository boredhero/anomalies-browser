"""Sky-View Factor detection pass — low SVF indicates enclosed/concave features.

SVF = proportion of visible sky from each cell. Concave features like dolines
and pits appear as dark spots (low SVF). Based on Zakšek et al. (2011).
A 30m search radius reveals dolines up to 50m diameter.
"""

import numpy as np
from scipy.ndimage import label as ndimage_label

from magic_eyes.detection.base import Candidate, DetectionPass, FeatureType, PassInput
from magic_eyes.detection.registry import register_pass
from magic_eyes.processing.derivatives import compute_svf


@register_pass
class SkyViewFactorPass(DetectionPass):
    """Detect enclosed features via low Sky-View Factor."""

    @property
    def name(self) -> str:
        return "sky_view_factor"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def required_derivatives(self) -> list[str]:
        return ["svf"]

    def run(self, input_data: PassInput) -> list[Candidate]:
        config = input_data.config
        threshold = config.get("threshold", 0.75)
        radius_m = config.get("radius_m", 30.0)
        n_directions = config.get("n_directions", 16)
        min_area_pixels = config.get("min_area_pixels", 4)

        resolution = abs(input_data.transform[0])

        # Use pre-computed SVF if available
        if "svf" in input_data.derivatives:
            svf = input_data.derivatives["svf"]
        else:
            svf = compute_svf(input_data.dem, resolution, radius_m, n_directions)

        # Low SVF = enclosed feature
        enclosed_mask = svf < threshold

        if not np.any(enclosed_mask):
            return []

        labeled, num_features = ndimage_label(enclosed_mask)

        candidates = []
        for i in range(1, num_features + 1):
            region = labeled == i
            if np.sum(region) < min_area_pixels:
                continue

            rows, cols = np.where(region)
            cy, cx = float(np.mean(rows)), float(np.mean(cols))
            geo_x, geo_y = input_data.transform * (cx, cy)

            min_svf = float(np.min(svf[region]))
            # Lower SVF = higher confidence
            score = max(0, 1.0 - min_svf)

            area_m2 = float(np.sum(region)) * resolution * resolution

            from shapely.geometry import Point

            candidates.append(
                Candidate(
                    geometry=Point(geo_x, geo_y),
                    score=min(score, 1.0),
                    feature_type=FeatureType.DEPRESSION,
                    morphometrics={
                        "min_svf": min_svf,
                        "area_m2": area_m2,
                    },
                )
            )

        return candidates

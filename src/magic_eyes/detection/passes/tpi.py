"""Topographic Position Index detection pass — negative TPI indicates depressions.

TPI = elevation - mean(elevation in neighborhood). Multi-scale analysis at
5m, 15m, and 50m radii. Negative values at all scales = high-confidence depression.
"""

import numpy as np
from scipy.ndimage import label as ndimage_label

from magic_eyes.detection.base import Candidate, DetectionPass, FeatureType, PassInput
from magic_eyes.detection.registry import register_pass
from magic_eyes.processing.derivatives import compute_tpi


@register_pass
class TPIPass(DetectionPass):
    """Detect depressions via negative Topographic Position Index."""

    @property
    def name(self) -> str:
        return "tpi"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def required_derivatives(self) -> list[str]:
        return ["tpi_15m"]

    def run(self, input_data: PassInput) -> list[Candidate]:
        config = input_data.config
        threshold = config.get("threshold", -1.0)
        radius_m = config.get("radius_m", 15.0)
        min_area_pixels = config.get("min_area_pixels", 4)

        resolution = abs(input_data.transform[0])
        radius_px = max(1, int(round(radius_m / resolution)))

        # Use pre-computed TPI if available
        tpi_key = f"tpi_{int(radius_m)}m"
        if tpi_key in input_data.derivatives:
            tpi = input_data.derivatives[tpi_key]
        else:
            tpi = compute_tpi(input_data.dem, radius_px)

        # Negative TPI = depression
        depression_mask = tpi < threshold

        if not np.any(depression_mask):
            return []

        labeled, num_features = ndimage_label(depression_mask)

        candidates = []
        for i in range(1, num_features + 1):
            region = labeled == i
            if np.sum(region) < min_area_pixels:
                continue

            rows, cols = np.where(region)
            cy, cx = float(np.mean(rows)), float(np.mean(cols))
            geo_x, geo_y = input_data.transform * (cx, cy)

            min_tpi = float(np.min(tpi[region]))
            score = min(abs(min_tpi) / 5.0, 1.0)

            area_m2 = float(np.sum(region)) * resolution * resolution

            from shapely.geometry import Point

            candidates.append(
                Candidate(
                    geometry=Point(geo_x, geo_y),
                    score=score,
                    feature_type=FeatureType.SINKHOLE,
                    morphometrics={
                        "min_tpi": min_tpi,
                        "area_m2": area_m2,
                        "tpi_radius_m": radius_m,
                    },
                )
            )

        return candidates

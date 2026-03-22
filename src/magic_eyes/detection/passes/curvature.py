"""Curvature detection pass — profile and plan curvature for concavity detection.

Strongly negative profile curvature indicates concavities (depressions, cave mouths).
Uses Zevenbergen & Thorne (1987) method from the derivatives module.
"""

import numpy as np
from scipy.ndimage import label as ndimage_label

from magic_eyes.detection.base import Candidate, DetectionPass, FeatureType, PassInput
from magic_eyes.detection.registry import register_pass
from magic_eyes.processing.derivatives import compute_curvature


@register_pass
class CurvaturePass(DetectionPass):
    """Detect concavities via profile/plan curvature thresholding."""

    @property
    def name(self) -> str:
        return "curvature"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def required_derivatives(self) -> list[str]:
        return []

    def run(self, input_data: PassInput) -> list[Candidate]:
        config = input_data.config
        curvature_type = config.get("type", "total")
        threshold = config.get("threshold", -0.02)
        min_area_pixels = config.get("min_area_pixels", 4)

        resolution = abs(input_data.transform[0])

        # Use pre-computed derivative if available
        key = f"{curvature_type}_curvature"
        if key in input_data.derivatives:
            curv = input_data.derivatives[key]
        else:
            curv = compute_curvature(input_data.dem, resolution, curvature_type)

        # Threshold — looking for strongly negative curvature (concavities)
        concave_mask = curv < threshold

        if not np.any(concave_mask):
            return []

        labeled, num_features = ndimage_label(concave_mask)

        candidates = []
        for i in range(1, num_features + 1):
            region = labeled == i
            if np.sum(region) < min_area_pixels:
                continue

            rows, cols = np.where(region)
            cy, cx = float(np.mean(rows)), float(np.mean(cols))
            geo_x, geo_y = input_data.transform * (cx, cy)

            # Strength: how negative the curvature is
            min_curv = float(np.min(curv[region]))
            strength = min(abs(min_curv) / 0.1, 1.0)

            area_m2 = float(np.sum(region)) * resolution * resolution

            from shapely.geometry import Point

            candidates.append(
                Candidate(
                    geometry=Point(geo_x, geo_y),
                    score=strength,
                    feature_type=FeatureType.DEPRESSION,
                    morphometrics={
                        "min_curvature": min_curv,
                        "area_m2": area_m2,
                        "curvature_type": curvature_type,
                    },
                )
            )

        return candidates

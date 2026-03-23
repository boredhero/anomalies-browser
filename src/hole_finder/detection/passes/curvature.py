"""Curvature detection pass — consumes pre-computed curvature rasters."""

import numpy as np
from scipy.ndimage import label as ndimage_label
from shapely.geometry import Point

from hole_finder.detection.base import Candidate, DetectionPass, FeatureType, PassInput
from hole_finder.detection.registry import register_pass


@register_pass
class CurvaturePass(DetectionPass):

    @property
    def name(self) -> str:
        return "curvature"

    @property
    def version(self) -> str:
        return "0.2.0"

    @property
    def required_derivatives(self) -> list[str]:
        return ["profile_curvature"]

    def run(self, input_data: PassInput) -> list[Candidate]:
        config = input_data.config
        threshold = config.get("threshold", -0.02)
        min_area_pixels = config.get("min_area_pixels", 4)

        resolution = abs(input_data.transform[0])

        curv = input_data.derivatives.get("profile_curvature")
        if curv is None:
            return []

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

            min_curv = float(np.min(curv[region]))
            strength = min(abs(min_curv) / 0.1, 1.0)
            area_m2 = float(np.sum(region)) * resolution * resolution

            candidates.append(
                Candidate(
                    geometry=Point(geo_x, geo_y),
                    score=strength,
                    feature_type=FeatureType.DEPRESSION,
                    morphometrics={"min_curvature": min_curv, "area_m2": area_m2},
                )
            )

        return candidates

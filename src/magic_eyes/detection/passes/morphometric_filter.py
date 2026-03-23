"""Morphometric filter pass — enriches candidates with full morphometrics.

Consumes pre-computed fill_difference and slope derivatives.
Never computes derivatives itself.
"""

import numpy as np
from scipy.ndimage import label as ndimage_label
from shapely.geometry import Point

from magic_eyes.detection.base import Candidate, DetectionPass, FeatureType, PassInput
from magic_eyes.detection.postprocess.classification import classify_candidate
from magic_eyes.detection.postprocess.morphometrics import (
    compute_area,
    compute_circularity,
    compute_depth,
    compute_elongation,
    compute_k_parameter,
    compute_perimeter,
    compute_volume,
    compute_wall_slope,
)
from magic_eyes.detection.registry import register_pass


@register_pass
class MorphometricFilterPass(DetectionPass):

    @property
    def name(self) -> str:
        return "morphometric_filter"

    @property
    def version(self) -> str:
        return "0.2.0"

    @property
    def required_derivatives(self) -> list[str]:
        return ["fill_difference", "slope"]

    def run(self, input_data: PassInput) -> list[Candidate]:
        config = input_data.config
        min_depth_m = config.get("min_depth_m", 0.3)
        max_area_m2 = config.get("max_area_m2", 4000.0)
        min_area_m2 = config.get("min_area_m2", 25.0)
        min_circularity = config.get("min_circularity", 0.15)

        resolution = abs(input_data.transform[0])
        dem = input_data.dem

        fill_diff = input_data.derivatives.get("fill_difference")
        slope = input_data.derivatives.get("slope")
        if fill_diff is None or slope is None:
            return []

        # Mask nodata
        fill_diff = np.where(np.isfinite(fill_diff) & (fill_diff < 1000), fill_diff, 0)

        depression_mask = fill_diff > min_depth_m
        if not np.any(depression_mask):
            return []

        labeled, num_features = ndimage_label(depression_mask)

        candidates = []
        for i in range(1, num_features + 1):
            mask = labeled == i

            depth = compute_depth(dem, mask)
            area = compute_area(mask, resolution)
            perimeter = compute_perimeter(mask, resolution)
            circularity = compute_circularity(area, perimeter)
            volume = compute_volume(dem, mask, resolution)
            k_param = compute_k_parameter(area, depth, volume)
            elongation = compute_elongation(mask)
            wall_slope_deg = compute_wall_slope(slope, mask)

            if area < min_area_m2 or area > max_area_m2:
                continue
            if circularity < min_circularity:
                continue
            if depth < min_depth_m:
                continue

            rows, cols = np.where(mask)
            cy, cx = float(np.mean(rows)), float(np.mean(cols))
            geo_x, geo_y = input_data.transform * (cx, cy)

            depth_score = min(depth / 5.0, 1.0)
            score = (depth_score + circularity) / 2.0

            candidate = Candidate(
                geometry=Point(geo_x, geo_y),
                score=score,
                feature_type=FeatureType.UNKNOWN,
                morphometrics={
                    "depth_m": depth, "area_m2": area, "perimeter_m": perimeter,
                    "circularity": circularity, "volume_m3": volume,
                    "k_parameter": k_param, "elongation": elongation,
                    "wall_slope_deg": wall_slope_deg,
                    "depth_area_ratio": depth / area if area > 0 else 0,
                },
            )
            candidate.feature_type = classify_candidate(candidate)
            candidates.append(candidate)

        return candidates

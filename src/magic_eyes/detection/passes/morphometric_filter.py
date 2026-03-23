"""Morphometric filter pass — post-filter that enriches and filters candidates.

Not a detector per se — it computes morphometric properties on candidates
from other passes and removes those failing geometric criteria.
Discriminators from literature: depth > 0.5m, area 100-4000 m²,
circularity > 0.3 for sinkholes, depth-to-area ratio, k parameter.
"""

import numpy as np
from scipy.ndimage import label as ndimage_label

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
from magic_eyes.processing.derivatives import compute_fill_difference, compute_slope


@register_pass
class MorphometricFilterPass(DetectionPass):
    """Enrich candidates with morphometrics and filter by geometric criteria."""

    @property
    def name(self) -> str:
        return "morphometric_filter"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def required_derivatives(self) -> list[str]:
        return ["slope"]

    def run(self, input_data: PassInput) -> list[Candidate]:
        """This pass operates on the DEM directly, finding depressions
        and computing full morphometrics for each one."""
        config = input_data.config
        min_depth_m = config.get("min_depth_m", 0.3)
        max_area_m2 = config.get("max_area_m2", 4000.0)
        min_area_m2 = config.get("min_area_m2", 25.0)
        min_circularity = config.get("min_circularity", 0.15)

        resolution = abs(input_data.transform[0])
        dem = input_data.dem

        # Get fill-difference to find depressions
        fill_diff = compute_fill_difference(dem)

        # Get slope for wall slope computation
        if "slope" in input_data.derivatives:
            slope = input_data.derivatives["slope"]
        else:
            slope = compute_slope(dem, resolution)

        # Label depressions
        depression_mask = fill_diff > min_depth_m
        if not np.any(depression_mask):
            return []

        labeled, num_features = ndimage_label(depression_mask)

        candidates = []
        for i in range(1, num_features + 1):
            mask = labeled == i

            # Compute all morphometrics
            depth = compute_depth(dem, mask)
            area = compute_area(mask, resolution)
            perimeter = compute_perimeter(mask, resolution)
            circularity = compute_circularity(area, perimeter)
            volume = compute_volume(dem, mask, resolution)
            k_param = compute_k_parameter(area, depth, volume)
            elongation = compute_elongation(mask)
            wall_slope_deg = compute_wall_slope(slope, mask)

            # Filter by criteria
            if area < min_area_m2 or area > max_area_m2:
                continue
            if circularity < min_circularity:
                continue
            if depth < min_depth_m:
                continue

            # Centroid
            rows, cols = np.where(mask)
            cy, cx = float(np.mean(rows)), float(np.mean(cols))
            geo_x, geo_y = input_data.transform * (cx, cy)

            # Score based on multiple factors
            depth_score = min(depth / 5.0, 1.0)
            circ_score = circularity
            score = (depth_score + circ_score) / 2.0

            from shapely.geometry import Point

            candidate = Candidate(
                geometry=Point(geo_x, geo_y),
                score=score,
                feature_type=FeatureType.UNKNOWN,
                morphometrics={
                    "depth_m": depth,
                    "area_m2": area,
                    "perimeter_m": perimeter,
                    "circularity": circularity,
                    "volume_m3": volume,
                    "k_parameter": k_param,
                    "elongation": elongation,
                    "wall_slope_deg": wall_slope_deg,
                    "depth_area_ratio": depth / area if area > 0 else 0,
                },
            )

            # Classify based on morphometrics
            candidate.feature_type = classify_candidate(candidate)
            candidates.append(candidate)

        return candidates

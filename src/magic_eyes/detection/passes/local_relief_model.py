"""Local Relief Model detection pass — consumes pre-computed LRM rasters.

Gold standard for cave entrance detection (Moyes & Montgomery 2019).
LRM rasters are computed by WhiteboxTools in the processing pipeline.
"""

import numpy as np
from scipy.ndimage import label as ndimage_label
from shapely.geometry import Point

from magic_eyes.detection.base import Candidate, DetectionPass, FeatureType, PassInput
from magic_eyes.detection.registry import register_pass


@register_pass
class LocalReliefModelPass(DetectionPass):

    @property
    def name(self) -> str:
        return "local_relief_model"

    @property
    def version(self) -> str:
        return "0.2.0"

    @property
    def required_derivatives(self) -> list[str]:
        return ["lrm_50m", "lrm_100m", "lrm_200m"]

    def run(self, input_data: PassInput) -> list[Candidate]:
        config = input_data.config
        threshold_m = config.get("threshold_m", 0.3)
        min_area_m2 = config.get("min_area_m2", 10.0)
        max_area_m2 = config.get("max_area_m2", 5000.0)

        resolution = abs(input_data.transform[0])

        # Combine multi-scale LRM — take per-pixel minimum (most negative = deepest anomaly)
        lrm_keys = [k for k in input_data.derivatives if k.startswith("lrm_")]
        if not lrm_keys:
            return []

        lrm_stack = [input_data.derivatives[k] for k in lrm_keys]
        combined = np.minimum.reduce(lrm_stack)

        depression_mask = combined < -threshold_m
        if not np.any(depression_mask):
            return []

        labeled, num_features = ndimage_label(depression_mask)

        candidates = []
        for i in range(1, num_features + 1):
            region = labeled == i
            area_pixels = np.sum(region)
            area_m2 = area_pixels * resolution * resolution

            if area_m2 < min_area_m2 or area_m2 > max_area_m2:
                continue

            rows, cols = np.where(region)
            cy, cx = float(np.mean(rows)), float(np.mean(cols))
            geo_x, geo_y = input_data.transform * (cx, cy)

            anomaly_depth = float(-np.min(combined[region]))

            candidates.append(
                Candidate(
                    geometry=Point(geo_x, geo_y),
                    score=min(anomaly_depth / 3.0, 1.0),
                    feature_type=FeatureType.CAVE_ENTRANCE,
                    morphometrics={"lrm_anomaly_m": anomaly_depth, "area_m2": area_m2},
                )
            )

        return candidates

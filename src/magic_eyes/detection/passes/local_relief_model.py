"""Local Relief Model detection pass — gold standard for cave entrance detection.

Subtracts a smoothed (trend) surface from the DEM at multiple kernel sizes.
Negative anomalies are depressions and potential cave entrances.

Based on Moyes & Montgomery (2019) — 80% of predicted horizontal cave
entrances confirmed via field survey in Belize jungle.
"""

import numpy as np
from scipy.ndimage import label as ndimage_label

from magic_eyes.detection.base import Candidate, DetectionPass, FeatureType, PassInput
from magic_eyes.detection.registry import register_pass
from magic_eyes.processing.derivatives import compute_lrm


@register_pass
class LocalReliefModelPass(DetectionPass):
    """Detect depressions and cave entrances via Local Relief Model."""

    @property
    def name(self) -> str:
        return "local_relief_model"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def required_derivatives(self) -> list[str]:
        return []  # computes its own LRM internally

    def run(self, input_data: PassInput) -> list[Candidate]:
        config = input_data.config
        kernel_sizes_m = config.get("kernel_sizes", [50, 100, 200])
        threshold_m = config.get("threshold_m", 0.3)
        min_area_m2 = config.get("min_area_m2", 10.0)
        max_area_m2 = config.get("max_area_m2", 5000.0)

        resolution = abs(input_data.transform[0])
        dem = input_data.dem

        # Compute LRM at each scale, take per-pixel minimum (most negative = deepest anomaly)
        lrm_stack = []
        for kernel_m in kernel_sizes_m:
            lrm = compute_lrm(dem, resolution, kernel_m)
            lrm_stack.append(lrm)

        combined_lrm = np.minimum.reduce(lrm_stack)

        # Threshold: negative anomalies below threshold
        depression_mask = combined_lrm < -threshold_m

        if not np.any(depression_mask):
            return []

        # Label connected components
        labeled, num_features = ndimage_label(depression_mask)

        candidates = []
        for i in range(1, num_features + 1):
            region = labeled == i
            area_pixels = np.sum(region)
            area_m2 = area_pixels * resolution * resolution

            if area_m2 < min_area_m2 or area_m2 > max_area_m2:
                continue

            # Centroid
            rows, cols = np.where(region)
            cy, cx = float(np.mean(rows)), float(np.mean(cols))
            geo_x, geo_y = input_data.transform * (cx, cy)

            # Anomaly depth (most negative LRM value)
            anomaly_depth = float(-np.min(combined_lrm[region]))

            from shapely.geometry import Point

            candidates.append(
                Candidate(
                    geometry=Point(geo_x, geo_y),
                    score=min(anomaly_depth / 3.0, 1.0),
                    feature_type=FeatureType.CAVE_ENTRANCE,
                    morphometrics={
                        "lrm_anomaly_m": anomaly_depth,
                        "area_m2": area_m2,
                    },
                )
            )

        return candidates

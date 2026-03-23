"""Fill-difference detection pass — finds depressions from pre-computed fill-difference raster.

Consumes the fill_difference derivative (filled_DEM - original_DEM).
Does NOT compute fill-difference itself — that's done by the processing pipeline
using WhiteboxTools (compiled Rust) and GDAL.

Based on Wall et al. (2016) — 93% detection rate for known sinkholes.
"""

import numpy as np
from scipy.ndimage import label as ndimage_label
from shapely.geometry import Point

from magic_eyes.detection.base import Candidate, DetectionPass, FeatureType, PassInput
from magic_eyes.detection.registry import register_pass


@register_pass
class FillDifferencePass(DetectionPass):
    """Detect depressions from pre-computed fill-difference raster."""

    @property
    def name(self) -> str:
        return "fill_difference"

    @property
    def version(self) -> str:
        return "0.2.0"

    @property
    def required_derivatives(self) -> list[str]:
        return ["fill_difference"]

    def run(self, input_data: PassInput) -> list[Candidate]:
        config = input_data.config
        min_depth_m = config.get("min_depth_m", 0.5)
        max_area_m2 = config.get("max_area_m2", 5000.0)
        min_area_m2 = config.get("min_area_m2", 25.0)

        resolution = abs(input_data.transform[0])

        # Use pre-computed fill_difference derivative
        diff = input_data.derivatives.get("fill_difference")
        if diff is None:
            return []

        # Mask nodata (bogus huge values from DEM edges)
        diff = np.where(np.isfinite(diff) & (diff < 1000), diff, 0)

        depression_mask = diff > min_depth_m
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

            depth = float(np.max(diff[region]))

            candidates.append(
                Candidate(
                    geometry=Point(geo_x, geo_y),
                    score=min(depth / 5.0, 1.0),
                    feature_type=FeatureType.DEPRESSION,
                    morphometrics={
                        "depth_m": depth,
                        "area_m2": area_m2,
                        "area_pixels": float(area_pixels),
                    },
                )
            )

        return candidates

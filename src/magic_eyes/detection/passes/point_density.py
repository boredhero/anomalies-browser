"""Point density detection pass — finds voids where LiDAR enters openings.

Areas with anomalously low point density indicate where LiDAR pulses
penetrated into cave/mine openings rather than reflecting off terrain.
Requires raw point cloud data.
"""

import numpy as np

from magic_eyes.detection.base import Candidate, DetectionPass, FeatureType, PassInput
from magic_eyes.detection.registry import register_pass
from magic_eyes.processing.point_cloud import compute_point_density


@register_pass
class PointDensityPass(DetectionPass):
    """Detect voids via point density anomalies in point cloud."""

    @property
    def name(self) -> str:
        return "point_density"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def required_derivatives(self) -> list[str]:
        return []

    @property
    def requires_point_cloud(self) -> bool:
        return True

    def run(self, input_data: PassInput) -> list[Candidate]:
        if input_data.point_cloud is None:
            return []

        config = input_data.config
        cell_size = config.get("cell_size_m", 2.0)
        z_threshold = config.get("z_score_threshold", -2.5)

        pc = input_data.point_cloud
        try:
            x = pc["X"].astype(np.float64)
            y = pc["Y"].astype(np.float64)
            z = pc["Z"].astype(np.float64)
        except (KeyError, TypeError):
            return []

        density, z_scores, bounds = compute_point_density(x, y, z, cell_size)

        # Find void cells (strongly negative z-score)
        from scipy.ndimage import label as ndimage_label

        void_mask = z_scores < z_threshold
        if not np.any(void_mask):
            return []

        labeled, num_features = ndimage_label(void_mask)
        xmin, ymin, xmax, ymax = bounds

        candidates = []
        for i in range(1, num_features + 1):
            region = labeled == i
            if np.sum(region) < 2:
                continue

            rows, cols = np.where(region)
            cy, cx = float(np.mean(rows)), float(np.mean(cols))

            # Convert grid coords to geographic
            geo_x = xmin + cx * cell_size
            geo_y = ymax - cy * cell_size

            min_zscore = float(np.min(z_scores[region]))
            score = min(abs(min_zscore) / 5.0, 1.0)

            area_m2 = float(np.sum(region)) * cell_size * cell_size

            from shapely.geometry import Point

            candidates.append(
                Candidate(
                    geometry=Point(geo_x, geo_y),
                    score=score,
                    feature_type=FeatureType.CAVE_ENTRANCE,
                    morphometrics={
                        "density_z_score": min_zscore,
                        "void_area_m2": area_m2,
                    },
                )
            )

        return candidates

"""Random Forest classifier pass — filters true sinkholes from false positives.

Extracts 10 morphometric features per candidate depression from the DEM and
derivatives, then classifies using a trained sklearn RandomForest. AUC 0.92
per Zhu et al. (2020).

Works on CPU — no GPU required.
"""

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label as ndimage_label

from magic_eyes.detection.base import Candidate, DetectionPass, FeatureType, PassInput
from magic_eyes.detection.postprocess.morphometrics import (
    compute_area,
    compute_circularity,
    compute_depth,
    compute_elongation,
    compute_perimeter,
    compute_volume,
    compute_wall_slope,
)
from magic_eyes.detection.registry import register_pass
from magic_eyes.processing.derivatives import compute_fill_difference, compute_slope, compute_tpi, compute_svf
from magic_eyes.config import settings


def extract_features(
    dem: np.ndarray,
    mask: np.ndarray,
    slope: np.ndarray,
    tpi: np.ndarray,
    svf: np.ndarray,
    resolution: float,
) -> np.ndarray:
    """Extract 10 morphometric features for a single depression region.

    Features (matching Zhu et al. 2020):
    0. depth_m
    1. area_m2
    2. perimeter_m
    3. circularity
    4. elongation
    5. depth_area_ratio
    6. mean_slope
    7. max_slope
    8. tpi_at_centroid
    9. svf_at_centroid
    """
    depth = compute_depth(dem, mask)
    area = compute_area(mask, resolution)
    perimeter = compute_perimeter(mask, resolution)
    circularity = compute_circularity(area, perimeter)
    elongation = compute_elongation(mask)
    depth_area_ratio = depth / area if area > 0 else 0
    mean_slope_val = float(np.mean(slope[mask])) if np.any(mask) else 0
    max_slope_val = float(np.max(slope[mask])) if np.any(mask) else 0

    rows, cols = np.where(mask)
    cy, cx = int(np.mean(rows)), int(np.mean(cols))
    cy = np.clip(cy, 0, tpi.shape[0] - 1)
    cx = np.clip(cx, 0, tpi.shape[1] - 1)
    tpi_centroid = float(tpi[cy, cx])
    svf_centroid = float(svf[cy, cx])

    return np.array([
        depth, area, perimeter, circularity, elongation,
        depth_area_ratio, mean_slope_val, max_slope_val,
        tpi_centroid, svf_centroid,
    ], dtype=np.float64)


FEATURE_NAMES = [
    "depth_m", "area_m2", "perimeter_m", "circularity", "elongation",
    "depth_area_ratio", "mean_slope", "max_slope", "tpi_centroid", "svf_centroid",
]


@register_pass
class RandomForestPass(DetectionPass):
    """Classify candidate depressions using Random Forest on morphometric features."""

    @property
    def name(self) -> str:
        return "random_forest"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def required_derivatives(self) -> list[str]:
        return ["slope", "tpi_15m", "svf"]

    def _load_model(self, model_path: Path | None = None):
        """Load a trained sklearn model from disk."""
        import joblib

        if model_path is None:
            model_path = settings.models_dir / "rf_sinkhole_v1.joblib"

        if not model_path.exists():
            return None
        return joblib.load(model_path)

    def run(self, input_data: PassInput) -> list[Candidate]:
        config = input_data.config
        min_depth_m = config.get("min_depth_m", 0.5)
        min_probability = config.get("min_probability", 0.5)
        model_path = config.get("model_path")

        resolution = abs(input_data.transform[0])
        dem = input_data.dem

        # Load trained model
        model = self._load_model(Path(model_path) if model_path else None)
        if model is None:
            # No model trained yet — skip gracefully
            return []

        # Compute needed derivatives
        slope = input_data.derivatives.get("slope")
        if slope is None:
            slope = compute_slope(dem, resolution)

        tpi = input_data.derivatives.get("tpi_15m")
        if tpi is None:
            tpi = compute_tpi(dem, radius_pixels=max(1, int(15 / resolution)))

        svf = input_data.derivatives.get("svf")
        if svf is None:
            svf = compute_svf(dem, resolution, radius_m=30.0, n_directions=16)

        # Find depressions via fill-difference
        fill_diff = compute_fill_difference(dem)
        depression_mask = fill_diff > min_depth_m
        if not np.any(depression_mask):
            return []

        labeled, num_features = ndimage_label(depression_mask)

        # Extract features for each depression and classify
        candidates = []
        for i in range(1, num_features + 1):
            mask = labeled == i
            if np.sum(mask) < 4:
                continue

            features = extract_features(dem, mask, slope, tpi, svf, resolution)
            features_2d = features.reshape(1, -1)

            # Predict probability
            try:
                proba = model.predict_proba(features_2d)[0]
                # Assume class 1 = true sinkhole/feature
                prob_positive = float(proba[1]) if len(proba) > 1 else float(proba[0])
            except Exception:
                continue

            if prob_positive < min_probability:
                continue

            rows, cols = np.where(mask)
            cy, cx = float(np.mean(rows)), float(np.mean(cols))
            geo_x, geo_y = input_data.transform * (cx, cy)

            from shapely.geometry import Point

            candidates.append(
                Candidate(
                    geometry=Point(geo_x, geo_y),
                    score=prob_positive,
                    feature_type=FeatureType.SINKHOLE,
                    morphometrics={
                        name: float(val) for name, val in zip(FEATURE_NAMES, features)
                    },
                    metadata={"classifier": "random_forest", "model_version": "v1"},
                )
            )

        return candidates

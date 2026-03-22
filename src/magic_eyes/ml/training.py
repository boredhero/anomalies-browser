"""Training data pipeline for ML models.

Extracts training samples from processed DEMs using ground truth data:
- Positive samples: patches centered on known karst/mine features
- Negative samples: random patches from non-feature terrain

Supports training for:
- Random Forest (feature vectors from morphometric extraction)
- U-Net (multi-channel image patches)
- YOLO (hillshade tiles with bounding box annotations)
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from magic_eyes.detection.passes.random_forest import FEATURE_NAMES, extract_features
from magic_eyes.processing.derivatives import (
    compute_curvature,
    compute_fill_difference,
    compute_hillshade,
    compute_slope,
    compute_svf,
    compute_tpi,
)


def extract_rf_training_data(
    dem: NDArray[np.float32],
    resolution: float,
    positive_masks: list[NDArray[np.bool_]],
    n_negatives: int = 0,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Extract Random Forest training features from a DEM.

    Args:
        dem: DEM array
        resolution: cell size in meters
        positive_masks: list of boolean masks for known features
        n_negatives: number of random negative patches to generate
        rng: random number generator

    Returns:
        (features_array, labels) where labels are 1=feature, 0=non-feature
    """
    if rng is None:
        rng = np.random.default_rng(42)

    slope = compute_slope(dem, resolution)
    tpi = compute_tpi(dem, max(1, int(15 / resolution)))
    svf = compute_svf(dem, resolution, radius_m=30.0, n_directions=8)

    features_list = []
    labels_list = []

    # Positive samples from known features
    for mask in positive_masks:
        if np.sum(mask) < 4:
            continue
        feats = extract_features(dem, mask, slope, tpi, svf, resolution)
        features_list.append(feats)
        labels_list.append(1)

    # Negative samples from random locations
    h, w = dem.shape
    fill_diff = compute_fill_difference(dem)

    for _ in range(n_negatives):
        # Random patch that is NOT a depression
        cy = rng.integers(20, h - 20)
        cx = rng.integers(20, w - 20)
        radius = rng.integers(5, 15)
        mask = np.zeros((h, w), dtype=bool)
        y, x = np.mgrid[0:h, 0:w]
        mask[(y - cy) ** 2 + (x - cx) ** 2 < radius ** 2] = True

        # Skip if this happens to be a real depression
        if np.max(fill_diff[mask]) > 0.5:
            continue

        feats = extract_features(dem, mask, slope, tpi, svf, resolution)
        features_list.append(feats)
        labels_list.append(0)

    if not features_list:
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=np.int32)

    return np.array(features_list), np.array(labels_list, dtype=np.int32)


def train_random_forest(
    X: NDArray[np.float64],
    y: NDArray[np.int32],
    output_path: Path,
    n_estimators: int = 200,
    class_weight: str = "balanced",
) -> dict:
    """Train a Random Forest classifier and save to disk.

    Returns dict with training metrics.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    import joblib

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )

    # Cross-validation
    if len(X) >= 10:
        cv_scores = cross_val_score(clf, X, y, cv=min(5, len(X)), scoring="roc_auc")
        auc_mean = float(np.mean(cv_scores))
        auc_std = float(np.std(cv_scores))
    else:
        auc_mean, auc_std = 0.0, 0.0

    # Train on full data
    clf.fit(X, y)

    # Feature importance
    importances = dict(zip(FEATURE_NAMES, clf.feature_importances_.tolist()))

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output_path)

    return {
        "n_samples": len(X),
        "n_positive": int(np.sum(y == 1)),
        "n_negative": int(np.sum(y == 0)),
        "cv_auc_mean": auc_mean,
        "cv_auc_std": auc_std,
        "feature_importances": importances,
        "model_path": str(output_path),
    }


def extract_unet_patches(
    dem: NDArray[np.float32],
    resolution: float,
    positive_centers: list[tuple[int, int]],
    patch_size: int = 256,
    n_negatives: int = 0,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Extract U-Net training patches (5-channel input + binary mask).

    Returns:
        (input_patches, label_patches) each of shape (N, C, H, W) / (N, 1, H, W)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    h, w = dem.shape
    half = patch_size // 2

    # Compute derivatives for all channels
    hs = compute_hillshade(dem, resolution)
    sl = compute_slope(dem, resolution)
    curv = compute_curvature(dem, resolution, "profile")
    tpi = compute_tpi(dem, max(1, int(15 / resolution)))
    svf = compute_svf(dem, resolution, radius_m=30.0, n_directions=8)

    def normalize(arr):
        vmin, vmax = np.nanmin(arr), np.nanmax(arr)
        if vmax - vmin < 1e-10:
            return np.zeros_like(arr)
        return (arr - vmin) / (vmax - vmin)

    channels = np.stack([normalize(hs), normalize(sl), normalize(curv), normalize(tpi), normalize(svf)])

    # Create label mask from fill-difference
    fill_diff = compute_fill_difference(dem)
    label_full = (fill_diff > 0.5).astype(np.float32)

    inputs = []
    labels = []

    # Positive patches centered on known features
    for cy, cx in positive_centers:
        if cy - half < 0 or cy + half > h or cx - half < 0 or cx + half > w:
            continue
        inp = channels[:, cy - half:cy + half, cx - half:cx + half]
        lbl = label_full[cy - half:cy + half, cx - half:cx + half]
        inputs.append(inp)
        labels.append(lbl[np.newaxis])

    # Negative patches
    for _ in range(n_negatives):
        cy = rng.integers(half, h - half)
        cx = rng.integers(half, w - half)
        inp = channels[:, cy - half:cy + half, cx - half:cx + half]
        lbl = np.zeros((1, patch_size, patch_size), dtype=np.float32)
        inputs.append(inp)
        labels.append(lbl)

    if not inputs:
        return np.empty((0, 5, patch_size, patch_size), dtype=np.float32), \
               np.empty((0, 1, patch_size, patch_size), dtype=np.float32)

    return np.array(inputs, dtype=np.float32), np.array(labels, dtype=np.float32)

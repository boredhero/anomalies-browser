"""Morphometric computation for detected features."""

import numpy as np
from numpy.typing import NDArray


def compute_depth(dem: NDArray[np.float32], mask: NDArray[np.bool_]) -> float:
    """Compute depth of a depression from DEM and binary mask."""
    if not np.any(mask):
        return 0.0
    rim_elevation = np.max(dem[mask])
    floor_elevation = np.min(dem[mask])
    return float(rim_elevation - floor_elevation)


def compute_area(mask: NDArray[np.bool_], resolution_m: float) -> float:
    """Compute area in m^2 from a binary mask and pixel resolution."""
    return float(np.sum(mask) * resolution_m * resolution_m)


def compute_circularity(area_m2: float, perimeter_m: float) -> float:
    """Compute circularity index: 4*pi*area / perimeter^2.

    Perfect circle = 1.0, elongated shapes < 1.0.
    """
    if perimeter_m <= 0:
        return 0.0
    return (4.0 * np.pi * area_m2) / (perimeter_m * perimeter_m)


def compute_perimeter(mask: NDArray[np.bool_], resolution_m: float) -> float:
    """Estimate perimeter from binary mask using edge counting."""
    from scipy.ndimage import binary_erosion

    interior = binary_erosion(mask)
    edge = mask & ~interior
    return float(np.sum(edge) * resolution_m)


def compute_k_parameter(area_m2: float, depth_m: float, volume_m3: float) -> float:
    """Compute Telbisz k parameter: (area * depth) / volume.

    k ≈ 1: cylinder, k ≈ 2: bowl/calotte, k ≈ 3: cone.
    """
    if volume_m3 <= 0:
        return 0.0
    return (area_m2 * depth_m) / volume_m3


def compute_volume(
    dem: NDArray[np.float32], mask: NDArray[np.bool_], resolution_m: float
) -> float:
    """Compute volume of depression below rim elevation."""
    if not np.any(mask):
        return 0.0
    rim_elevation = np.max(dem[mask])
    depths = rim_elevation - dem[mask]
    depths = np.maximum(depths, 0)
    cell_area = resolution_m * resolution_m
    return float(np.sum(depths) * cell_area)


def compute_wall_slope(
    slope: NDArray[np.float32], mask: NDArray[np.bool_]
) -> float:
    """Compute mean slope of depression interior walls (degrees)."""
    if not np.any(mask):
        return 0.0
    return float(np.mean(slope[mask]))


def compute_elongation(mask: NDArray[np.bool_]) -> float:
    """Compute elongation ratio: minor_axis / major_axis.

    1.0 = circular, < 1.0 = elongated.
    """
    coords = np.argwhere(mask)
    if len(coords) < 3:
        return 1.0

    # PCA to find major/minor axes
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]

    if eigenvalues[0] <= 0:
        return 1.0
    return float(np.sqrt(eigenvalues[1] / eigenvalues[0]))

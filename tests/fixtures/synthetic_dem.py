"""Synthetic DEM generators for deterministic testing."""

import numpy as np
from numpy.typing import NDArray
from rasterio.transform import from_bounds

from magic_eyes.detection.base import PassInput


def make_flat_dem(
    size: int = 200,
    resolution: float = 1.0,
    elevation: float = 500.0,
) -> PassInput:
    """Create a perfectly flat DEM."""
    dem = np.full((size, size), elevation, dtype=np.float32)
    transform = from_bounds(0, 0, size * resolution, size * resolution, size, size)
    return PassInput(dem=dem, transform=transform, crs=32617, derivatives={})


def make_slope_dem(
    size: int = 200,
    resolution: float = 1.0,
    base_elevation: float = 500.0,
    slope_deg: float = 10.0,
) -> PassInput:
    """Create a uniform slope DEM (tilted in Y direction)."""
    y = np.arange(size, dtype=np.float32) * resolution
    slope_rise = np.tan(np.radians(slope_deg)) * y
    dem = np.tile(slope_rise[:, np.newaxis], (1, size)) + base_elevation
    transform = from_bounds(0, 0, size * resolution, size * resolution, size, size)
    return PassInput(dem=dem, transform=transform, crs=32617, derivatives={})


def make_sinkhole_dem(
    size: int = 200,
    resolution: float = 1.0,
    depth: float = 3.0,
    radius: float = 15.0,
    base_elevation: float = 500.0,
    center: tuple[float, float] | None = None,
) -> PassInput:
    """Create a DEM with a single gaussian sinkhole.

    Args:
        size: grid dimensions (size x size pixels)
        resolution: meters per pixel
        depth: depth of sinkhole in meters
        radius: gaussian sigma in meters
        base_elevation: surrounding terrain elevation
        center: (row, col) center in pixels, defaults to grid center
    """
    if center is None:
        center = (size / 2.0, size / 2.0)

    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    cy, cx = center
    dist_sq = (x - cx) ** 2 + (y - cy) ** 2
    dem = base_elevation - depth * np.exp(-dist_sq / (2 * (radius / resolution) ** 2))
    transform = from_bounds(0, 0, size * resolution, size * resolution, size, size)
    return PassInput(dem=dem, transform=transform, crs=32617, derivatives={})


def make_conical_pit_dem(
    size: int = 200,
    resolution: float = 1.0,
    depth: float = 5.0,
    radius: float = 10.0,
    base_elevation: float = 500.0,
) -> PassInput:
    """Create a DEM with a conical pit (steep-walled, like a cave entrance)."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy = size / 2.0, size / 2.0
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) * resolution
    dem = np.full((size, size), base_elevation, dtype=np.float32)
    pit_mask = dist < radius
    dem[pit_mask] = base_elevation - depth * (1 - dist[pit_mask] / radius)
    transform = from_bounds(0, 0, size * resolution, size * resolution, size, size)
    return PassInput(dem=dem, transform=transform, crs=32617, derivatives={})


def make_multi_depression_dem(
    size: int = 400,
    resolution: float = 1.0,
    base_elevation: float = 500.0,
) -> PassInput:
    """Create a DEM with multiple depressions of varying sizes."""
    dem = np.full((size, size), base_elevation, dtype=np.float32)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)

    depressions = [
        (100, 100, 3.0, 20.0),  # (row, col, depth, radius)
        (100, 300, 1.5, 10.0),
        (300, 100, 5.0, 30.0),
        (300, 300, 0.3, 8.0),   # too shallow for default threshold
    ]

    for cy, cx, depth, radius in depressions:
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        dem -= depth * np.exp(-dist_sq / (2 * (radius / resolution) ** 2))

    transform = from_bounds(0, 0, size * resolution, size * resolution, size, size)
    return PassInput(dem=dem, transform=transform, crs=32617, derivatives={})

"""Geographic utility functions."""

import numpy as np
from shapely.geometry import Polygon, box


def bbox_to_polygon(west: float, south: float, east: float, north: float) -> Polygon:
    """Create a Shapely polygon from bounding box coordinates."""
    return box(west, south, east, north)


def degrees_to_meters(lat: float, lon_delta: float, lat_delta: float) -> tuple[float, float]:
    """Approximate conversion of degree deltas to meters at a given latitude."""
    lat_m = lat_delta * 111_320.0
    lon_m = lon_delta * 111_320.0 * np.cos(np.radians(lat))
    return lon_m, lat_m


def meters_to_degrees(lat: float, x_m: float, y_m: float) -> tuple[float, float]:
    """Approximate conversion of meter deltas to degrees at a given latitude."""
    lat_deg = y_m / 111_320.0
    lon_deg = x_m / (111_320.0 * np.cos(np.radians(lat)))
    return lon_deg, lat_deg

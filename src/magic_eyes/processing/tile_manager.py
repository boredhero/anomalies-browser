"""Tile manager — spatial indexing and tile grid management.

Maintains an R-tree index of all processed tiles for fast spatial queries.
Handles overlap buffers for edge-effect mitigation during detection.
"""

from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from rtree import index as rtree_index
from shapely.geometry import Polygon, box


@dataclass
class ManagedTile:
    """A tile tracked by the TileManager."""

    tile_id: UUID
    bbox: Polygon
    dem_path: Path | None = None
    derivative_paths: dict[str, Path] = field(default_factory=dict)
    point_cloud_path: Path | None = None
    crs: int = 32617
    resolution_m: float = 1.0


class TileManager:
    """Spatial index and metadata for all processed tiles."""

    def __init__(self):
        self._idx = rtree_index.Index()
        self._tiles: dict[int, ManagedTile] = {}
        self._counter = 0

    def add_tile(self, tile: ManagedTile) -> int:
        """Add a tile to the spatial index. Returns internal index ID."""
        idx_id = self._counter
        self._counter += 1
        self._tiles[idx_id] = tile
        bounds = tile.bbox.bounds  # (minx, miny, maxx, maxy)
        self._idx.insert(idx_id, bounds)
        return idx_id

    def query_bbox(
        self,
        west: float,
        south: float,
        east: float,
        north: float,
    ) -> list[ManagedTile]:
        """Find all tiles intersecting a bounding box."""
        hits = list(self._idx.intersection((west, south, east, north)))
        return [self._tiles[i] for i in hits]

    def query_polygon(self, polygon: Polygon) -> list[ManagedTile]:
        """Find all tiles intersecting an arbitrary polygon."""
        bounds = polygon.bounds
        candidates = self.query_bbox(*bounds)
        return [t for t in candidates if t.bbox.intersects(polygon)]

    def get_neighbors(self, tile: ManagedTile, buffer_m: float = 100.0) -> list[ManagedTile]:
        """Find tiles neighboring a given tile (for overlap processing).

        Args:
            tile: the reference tile
            buffer_m: buffer distance in approximate meters (degrees approximation)
        """
        # Approximate meters to degrees at typical latitude (~40°N)
        buffer_deg = buffer_m / 111_320.0
        buffered = tile.bbox.buffer(buffer_deg)
        neighbors = self.query_polygon(buffered)
        return [t for t in neighbors if t.tile_id != tile.tile_id]

    def count(self) -> int:
        return len(self._tiles)

    def all_tiles(self) -> list[ManagedTile]:
        return list(self._tiles.values())

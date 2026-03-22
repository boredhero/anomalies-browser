"""Abstract base class for LiDAR data sources."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from shapely.geometry import Polygon


@dataclass
class TileInfo:
    """Metadata for a discoverable LiDAR tile."""

    source_id: str
    filename: str
    url: str
    bbox: Polygon
    crs: int
    file_size_bytes: int | None = None
    acquisition_year: int | None = None
    format: str = "laz"


class DataSource(ABC):
    """Abstract base for LiDAR data sources."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this data source."""
        ...

    @abstractmethod
    async def discover_tiles(self, bbox: Polygon) -> AsyncIterator[TileInfo]:
        """Find available tiles intersecting the bounding box."""
        ...

    @abstractmethod
    async def download_tile(self, tile: TileInfo, dest_dir: Path) -> Path:
        """Download a single tile. Return local file path."""
        ...

    async def download_region(self, bbox: Polygon, dest_dir: Path) -> list[Path]:
        """Download all tiles in a region."""
        paths = []
        async for tile_info in self.discover_tiles(bbox):
            path = await self.download_tile(tile_info, dest_dir)
            paths.append(path)
        return paths

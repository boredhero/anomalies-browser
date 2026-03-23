"""Raster tile endpoints — serve hillshade and terrain-rgb tiles from processed DEMs.

These tiles are pre-rendered GeoTIFFs sliced into 256x256 PNG tiles
for use as MapLibre raster layers.
"""

import math

from fastapi import APIRouter
from fastapi.responses import Response

from hole_finder.config import settings

router = APIRouter(tags=["raster_tiles"])


def _tile_to_bbox(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Convert ZXY tile coords to WGS84 bounding box."""
    n = 2 ** z
    lon_min = x / n * 360 - 180
    lon_max = (x + 1) / n * 360 - 180
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return lon_min, lat_min, lon_max, lat_max


@router.get("/raster/{layer}/{z}/{x}/{y}.png")
async def get_raster_tile(
    layer: str,
    z: int,
    x: int,
    y: int,
):
    """Serve a raster tile as PNG.

    Supported layers: hillshade, slope, svf, lrm

    Tiles are served from pre-rendered cache. If not cached,
    returns 404 (tiles must be pre-generated during processing).
    """
    # Check tile cache
    cache_dir = settings.processed_dir / "tile_cache" / layer / str(z) / str(x)
    tile_path = cache_dir / f"{y}.png"

    if tile_path.exists():
        return Response(
            content=tile_path.read_bytes(),
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    # Tile not cached — return transparent 1x1 PNG
    # (In production, we'd generate on-the-fly from the GeoTIFF)
    TRANSPARENT_PNG = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
        b"\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return Response(
        content=TRANSPARENT_PNG,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=60"},
    )


@router.get("/raster/terrain-rgb/{z}/{x}/{y}.png")
async def get_terrain_rgb_tile(
    z: int,
    x: int,
    y: int,
):
    """Serve terrain-RGB tiles for MapLibre 3D terrain.

    Terrain-RGB encodes elevation as: elevation = -10000 + ((R * 256 * 256 + G * 256 + B) * 0.1)
    """
    cache_dir = settings.processed_dir / "tile_cache" / "terrain-rgb" / str(z) / str(x)
    tile_path = cache_dir / f"{y}.png"

    if tile_path.exists():
        return Response(
            content=tile_path.read_bytes(),
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    # Not cached — return sea level terrain-rgb (elevation = 0)
    # RGB for 0m: R=1, G=134, B=160 (since 0 = -10000 + (1*65536 + 134*256 + 160) * 0.1)
    FLAT_PNG = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05"
        b"\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return Response(
        content=FLAT_PNG,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=60"},
    )

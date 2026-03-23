"""DEM/DTM generation from LiDAR point clouds via PDAL.

Supports COPC, LAZ, and LAS input formats. Uses SMRF (Simple Morphological
Filter) for ground classification and IDW interpolation for DEM generation.

PDAL is a system dependency — this module only runs on machines with PDAL installed
(i.e., the remote worker at 192.168.1.111, not the dev laptop).
"""

import json
import subprocess
from pathlib import Path

from hole_finder.utils.logging import log


def build_dem_pipeline(
    input_path: str,
    output_path: str,
    resolution: float = 1.0,
    target_srs: str | None = None,
) -> dict:
    """Build a PDAL pipeline JSON for ground-classified DEM generation.

    Steps:
    1. Read point cloud (auto-detects format)
    2. Optionally reproject to target CRS
    3. SMRF ground classification
    4. Filter to ground-only points (classification 2)
    5. Write DEM via GDAL IDW interpolation
    """
    pipeline: list[dict] = []

    # Reader — PDAL auto-detects format from extension
    reader = {"type": "readers.copc" if input_path.endswith(".copc.laz") else "readers.las",
              "filename": input_path}
    pipeline.append(reader)

    # Reproject if target SRS specified
    if target_srs:
        pipeline.append({
            "type": "filters.reprojection",
            "out_srs": target_srs,
        })

    # SMRF ground classification
    pipeline.append({
        "type": "filters.smrf",
        "slope": 0.15,
        "window": 18,
        "threshold": 0.5,
        "scalar": 1.25,
    })

    # Filter to ground-classified points only
    pipeline.append({
        "type": "filters.range",
        "limits": "Classification[2:2]",
    })

    # Write DEM via GDAL
    pipeline.append({
        "type": "writers.gdal",
        "filename": output_path,
        "resolution": resolution,
        "output_type": "idw",
        "gdalopts": "COMPRESS=DEFLATE,TILED=YES,BLOCKXSIZE=256,BLOCKYSIZE=256",
        "data_type": "float32",
    })

    return {"pipeline": pipeline}


def build_full_return_dem_pipeline(
    input_path: str,
    output_path: str,
    resolution: float = 1.0,
    target_srs: str | None = None,
) -> dict:
    """Build pipeline for full-return DEM preserving low-point classifications.

    Critical for cave entrance detection — standard DEMs filter out the very
    returns that indicate cave openings (class 7 = low point/noise).
    """
    pipeline: list[dict] = []

    reader = {"type": "readers.copc" if input_path.endswith(".copc.laz") else "readers.las",
              "filename": input_path}
    pipeline.append(reader)

    if target_srs:
        pipeline.append({
            "type": "filters.reprojection",
            "out_srs": target_srs,
        })

    # Keep ground (2) + low point (7) + unclassified (1)
    pipeline.append({
        "type": "filters.range",
        "limits": "Classification[1:2],Classification[7:7]",
    })

    pipeline.append({
        "type": "writers.gdal",
        "filename": output_path,
        "resolution": resolution,
        "output_type": "idw",
        "gdalopts": "COMPRESS=DEFLATE,TILED=YES",
        "data_type": "float32",
    })

    return {"pipeline": pipeline}


def run_pdal_pipeline(pipeline_dict: dict) -> dict:
    """Execute a PDAL pipeline via subprocess (pdal pipeline command).

    Returns metadata from the pipeline execution.
    """
    pipeline_json = json.dumps(pipeline_dict)

    try:
        result = subprocess.run(
            ["pdal", "pipeline", "--stdin", "--metadata", "/dev/stdout"],
            input=pipeline_json,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per tile
        )

        if result.returncode != 0:
            log.error("pdal_failed", stderr=result.stderr[:500])
            raise RuntimeError(f"PDAL pipeline failed: {result.stderr[:500]}")

        # Try to parse metadata
        try:
            return json.loads(result.stdout) if result.stdout.strip() else {}
        except json.JSONDecodeError:
            return {}

    except FileNotFoundError:
        raise RuntimeError(
            "PDAL not found. Install system PDAL: pacman -S pdal (Arch) or apt install pdal (Debian)"
        )


def generate_dem(
    input_path: Path,
    output_dir: Path,
    resolution: float = 1.0,
    target_srs: str | None = None,
) -> tuple[Path, Path | None]:
    """Generate ground-classified DEM and optionally full-return DEM.

    Args:
        input_path: path to LAZ/COPC point cloud
        output_dir: directory for output GeoTIFFs
        resolution: DEM resolution in meters
        target_srs: target CRS (e.g., "EPSG:32617")

    Returns:
        (dem_path, full_return_dem_path) — full_return may be None
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem.replace(".copc", "")

    # Ground-classified DEM
    dem_path = output_dir / f"{stem}_dem_{resolution}m.tif"
    if not dem_path.exists():
        log.info("generating_dem", input=str(input_path), output=str(dem_path))
        pipeline = build_dem_pipeline(
            str(input_path), str(dem_path), resolution, target_srs
        )
        run_pdal_pipeline(pipeline)

    # Full-return DEM (for cave detection)
    full_dem_path = output_dir / f"{stem}_fullreturn_{resolution}m.tif"
    if not full_dem_path.exists():
        try:
            log.info("generating_full_return_dem", output=str(full_dem_path))
            pipeline = build_full_return_dem_pipeline(
                str(input_path), str(full_dem_path), resolution, target_srs
            )
            run_pdal_pipeline(pipeline)
        except Exception as e:
            log.warning("full_return_dem_failed", error=str(e))
            full_dem_path = None

    return dem_path, full_dem_path

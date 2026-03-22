"""Celery task definitions."""

from pathlib import Path

from magic_eyes.workers.celery_app import app
from magic_eyes.config import settings


@app.task(bind=True, queue="ingest", max_retries=3)
def download_tile(self, source_name: str, tile_info_dict: dict, dest_dir: str):
    """Download a single LiDAR tile from the given source."""
    import asyncio
    from magic_eyes.ingest.manager import get_source
    from magic_eyes.ingest.sources.base import TileInfo
    from shapely.geometry import shape

    self.update_state(state="PROGRESS", meta={"percent": 0, "message": "Starting download"})

    source = get_source(source_name)
    tile = TileInfo(
        source_id=tile_info_dict["source_id"],
        filename=tile_info_dict["filename"],
        url=tile_info_dict["url"],
        bbox=shape(tile_info_dict["bbox"]),
        crs=tile_info_dict.get("crs", 4326),
        file_size_bytes=tile_info_dict.get("file_size_bytes"),
        format=tile_info_dict.get("format", "laz"),
    )

    dest = Path(dest_dir)
    result_path = asyncio.run(source.download_tile(tile, dest))

    self.update_state(state="PROGRESS", meta={"percent": 100, "message": "Complete"})
    return str(result_path)


@app.task(bind=True, queue="process")
def process_tile(self, tile_path: str, output_dir: str | None = None):
    """Generate DEM and derivatives for a tile.

    Args:
        tile_path: path to LAZ/COPC file OR existing DEM GeoTIFF
        output_dir: output directory (default: settings.processed_dir)
    """
    from magic_eyes.processing.pipeline import ProcessingPipeline

    self.update_state(state="PROGRESS", meta={"percent": 0, "message": "Starting processing"})

    input_path = Path(tile_path)
    out_dir = Path(output_dir) if output_dir else settings.processed_dir

    pipeline = ProcessingPipeline(output_dir=out_dir)

    if input_path.suffix in (".laz", ".las"):
        result = pipeline.process_point_cloud(input_path)
    elif input_path.suffix in (".tif", ".tiff"):
        result = pipeline.process_dem_file(input_path)
    else:
        raise ValueError(f"Unsupported file type: {input_path.suffix}")

    self.update_state(state="PROGRESS", meta={"percent": 100, "message": "Complete"})
    return {
        "dem_path": str(result.dem_path),
        "derivative_paths": {k: str(v) for k, v in result.derivative_paths.items()},
        "resolution_m": result.resolution_m,
        "crs": result.crs,
    }


@app.task(bind=True, queue="detect")
def run_detection(self, tile_id: str, pass_names: list, config: dict):
    """Run classical detection passes on a processed tile."""
    # TODO: implement in Phase 4
    raise NotImplementedError


@app.task(bind=True, queue="gpu")
def run_ml_pass(self, tile_id: str, pass_name: str, config: dict):
    """Run a single ML-based detection pass (GPU required)."""
    # TODO: implement in Phase 7
    raise NotImplementedError

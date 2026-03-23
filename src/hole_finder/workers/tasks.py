"""Celery task definitions."""

from pathlib import Path

from hole_finder.config import settings
from hole_finder.workers.celery_app import app


@app.task(bind=True, queue="ingest", max_retries=3)
def download_tile(self, source_name: str, tile_info_dict: dict, dest_dir: str):
    """Download a single LiDAR tile from the given source."""
    import asyncio

    from shapely.geometry import shape

    from hole_finder.ingest.manager import get_source
    from hole_finder.ingest.sources.base import TileInfo

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
    """Generate DEM and derivatives for a tile."""
    from hole_finder.processing.pipeline import ProcessingPipeline

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
def run_detection(self, dem_path: str, pass_names: list, config: dict):
    """Run detection passes on a processed tile's DEM."""
    from hole_finder.detection.runner import PassRunner

    self.update_state(state="PROGRESS", meta={"percent": 0, "message": "Running detection"})

    runner = PassRunner(
        pass_names=pass_names,
        config=config,
        weights=config.get("weights"),
    )

    candidates = runner.run_on_dem(Path(dem_path))

    self.update_state(state="PROGRESS", meta={"percent": 100, "message": "Complete"})
    return {
        "num_detections": len(candidates),
        "detections": [
            {
                "lon": c.geometry.x,
                "lat": c.geometry.y,
                "score": c.score,
                "feature_type": c.feature_type.value,
                "morphometrics": c.morphometrics,
            }
            for c in candidates
        ],
    }


@app.task(bind=True, queue="gpu")
def run_ml_pass(self, dem_path: str, pass_name: str, config: dict):
    """Run a single ML detection pass (GPU queue)."""

    from hole_finder.detection.base import PassInput
    from hole_finder.detection.registry import PassRegistry
    from hole_finder.utils.raster_io import read_dem

    self.update_state(state="PROGRESS", meta={"percent": 0, "message": f"Running {pass_name}"})

    dem, transform, crs = read_dem(Path(dem_path))

    pass_cls = PassRegistry.get(pass_name)
    detection_pass = pass_cls()

    pass_input = PassInput(
        dem=dem,
        transform=transform,
        crs=crs,
        derivatives={},
        config=config.get(f"passes.{pass_name}", {}),
    )

    candidates = detection_pass.run(pass_input)

    self.update_state(state="PROGRESS", meta={"percent": 100, "message": "Complete"})
    return {
        "pass_name": pass_name,
        "num_detections": len(candidates),
        "detections": [
            {
                "lon": c.geometry.x,
                "lat": c.geometry.y,
                "score": c.score,
                "feature_type": c.feature_type.value,
            }
            for c in candidates
        ],
    }

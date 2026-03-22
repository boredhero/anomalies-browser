"""Celery task definitions."""

from magic_eyes.workers.celery_app import app


@app.task(bind=True, queue="ingest", max_retries=3)
def download_tile(self, source_name: str, tile_info_dict: dict, dest_dir: str):
    """Download a single LiDAR tile from the given source."""
    # TODO: implement
    self.update_state(state="PROGRESS", meta={"percent": 0, "message": "Starting"})
    raise NotImplementedError


@app.task(bind=True, queue="process")
def process_tile(self, tile_id: str):
    """Generate DEM and derivatives for a tile."""
    # TODO: implement
    raise NotImplementedError


@app.task(bind=True, queue="detect")
def run_detection(self, tile_id: str, pass_names: list, config: dict):
    """Run classical detection passes on a processed tile."""
    # TODO: implement
    raise NotImplementedError


@app.task(bind=True, queue="gpu")
def run_ml_pass(self, tile_id: str, pass_name: str, config: dict):
    """Run a single ML-based detection pass (GPU required)."""
    # TODO: implement
    raise NotImplementedError

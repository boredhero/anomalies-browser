"""Processing pipeline — orchestrates PDAL + GDAL + WhiteboxTools.

Python only orchestrates subprocesses and reads results.
All derivative computation is in derivatives.py.
"""

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from magic_eyes.processing.derivatives import compute_all_derivatives, fill_depressions
from magic_eyes.utils.logging import log


@dataclass
class ProcessedTile:
    """Immutable result of processing a tile. Stored permanently."""

    tile_dir: Path
    dem_path: Path
    filled_dem_path: Path | None = None
    derivative_paths: dict[str, Path] = field(default_factory=dict)
    resolution_m: float = 1.0
    crs: int = 32617


def generate_dem_pdal(
    input_path: Path,
    output_dir: Path,
    resolution: float = 1.0,
    target_srs: str | None = None,
) -> tuple[Path, Path]:
    """Generate ground DEM and filled DEM from point cloud using PDAL."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem.replace(".copc", "")
    dem_path = output_dir / f"{stem}_dem.tif"
    filled_path = output_dir / f"{stem}_filled.tif"

    if not dem_path.exists():
        pipeline = [
            {"type": "readers.copc" if str(input_path).endswith(".copc.laz") else "readers.las",
             "filename": str(input_path)},
        ]
        if target_srs:
            pipeline.append({"type": "filters.reprojection", "out_srs": target_srs})
        pipeline.extend([
            {"type": "filters.smrf", "slope": 0.15, "window": 18, "threshold": 0.5},
            {"type": "filters.range", "limits": "Classification[2:2]"},
            {"type": "writers.gdal", "filename": str(dem_path), "resolution": resolution,
             "output_type": "idw", "gdalopts": "COMPRESS=DEFLATE,TILED=YES,BLOCKXSIZE=256,BLOCKYSIZE=256",
             "data_type": "float32"},
        ])
        log.info("pdal_dem", input=str(input_path))
        proc = subprocess.run(
            ["pdal", "pipeline", "--stdin"],
            input=json.dumps({"pipeline": pipeline}),
            capture_output=True, text=True, timeout=900,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"PDAL DEM failed: {proc.stderr[:500]}")

    if not filled_path.exists():
        fill_depressions(str(dem_path), str(filled_path))

    return dem_path, filled_path


class ProcessingPipeline:
    """Full tile processing — PDAL + GDAL + WhiteboxTools.

    All outputs are written to persistent storage. Once computed,
    derivatives are cached permanently and never recomputed unless
    explicitly requested via force=True.
    """

    def __init__(
        self,
        output_dir: Path,
        resolution: float = 1.0,
        target_srs: str | None = None,
    ):
        self.output_dir = output_dir
        self.resolution = resolution
        self.target_srs = target_srs

    def process_point_cloud(self, input_path: Path, force: bool = False) -> ProcessedTile:
        """Process a LAZ/COPC point cloud through the full pipeline."""
        stem = input_path.stem.replace(".copc", "")
        tile_dir = self.output_dir / stem
        tile_dir.mkdir(parents=True, exist_ok=True)
        deriv_dir = tile_dir / "derivatives"

        marker = tile_dir / ".processed"
        if marker.exists() and not force:
            return self._load_existing(tile_dir, deriv_dir)

        log.info("pipeline_stage1_dem", input=str(input_path))
        dem_path, filled_path = generate_dem_pdal(
            input_path, tile_dir, self.resolution, self.target_srs
        )

        log.info("pipeline_stage2_derivatives")
        derivative_paths = compute_all_derivatives(dem_path, filled_path, deriv_dir)

        marker.write_text(f"processed\nderivatives: {len(derivative_paths)}\n")
        log.info("pipeline_complete", tile_dir=str(tile_dir), derivatives=len(derivative_paths))

        return ProcessedTile(
            tile_dir=tile_dir, dem_path=dem_path, filled_dem_path=filled_path,
            derivative_paths=derivative_paths, resolution_m=self.resolution,
        )

    def process_dem_file(self, dem_path: Path, force: bool = False) -> ProcessedTile:
        """Process from an existing DEM file (no PDAL needed)."""
        stem = dem_path.stem
        tile_dir = self.output_dir / stem
        tile_dir.mkdir(parents=True, exist_ok=True)
        deriv_dir = tile_dir / "derivatives"

        marker = tile_dir / ".processed"
        if marker.exists() and not force:
            return self._load_existing(tile_dir, deriv_dir)

        filled_path = tile_dir / f"{stem}_filled.tif"
        if not filled_path.exists():
            fill_depressions(str(dem_path), str(filled_path))

        log.info("pipeline_derivatives", dem=str(dem_path))
        derivative_paths = compute_all_derivatives(dem_path, filled_path, deriv_dir)

        marker.write_text(f"processed\nderivatives: {len(derivative_paths)}\n")
        return ProcessedTile(
            tile_dir=tile_dir, dem_path=dem_path, filled_dem_path=filled_path,
            derivative_paths=derivative_paths, resolution_m=self.resolution,
        )

    def _load_existing(self, tile_dir: Path, deriv_dir: Path) -> ProcessedTile:
        """Load an already-processed tile from disk."""
        dem_files = list(tile_dir.glob("*_dem.tif")) + list(tile_dir.glob("dem_*.tif"))
        dem_path = dem_files[0] if dem_files else tile_dir / "dem.tif"

        filled_files = list(tile_dir.glob("*_filled.tif"))
        filled_path = filled_files[0] if filled_files else None

        derivative_paths = {}
        if deriv_dir.exists():
            for f in deriv_dir.glob("*.tif"):
                derivative_paths[f.stem] = f

        log.info("pipeline_cached", tile_dir=str(tile_dir), derivatives=len(derivative_paths))
        return ProcessedTile(
            tile_dir=tile_dir, dem_path=dem_path, filled_dem_path=filled_path,
            derivative_paths=derivative_paths, resolution_m=self.resolution,
        )

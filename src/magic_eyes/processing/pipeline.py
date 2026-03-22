"""ProcessingPipeline — orchestrates full tile processing chain.

Given a point cloud file, produces:
1. Ground-classified DEM (via PDAL)
2. Full-return DEM (preserves cave-opening returns)
3. All terrain derivatives (slope, curvature, TPI, SVF, LRM, etc.)
4. Point cloud density and multi-return analysis

For machines without PDAL (dev laptops), can start from an existing DEM file.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from magic_eyes.processing.derivatives import compute_all_derivatives
from magic_eyes.utils.logging import log
from magic_eyes.utils.raster_io import read_dem, write_raster


@dataclass
class ProcessedTile:
    """Result of processing a tile through the pipeline."""

    dem_path: Path
    full_return_dem_path: Path | None
    derivative_paths: dict[str, Path]
    resolution_m: float
    crs: int


class ProcessingPipeline:
    """Full processing chain for a single LiDAR tile."""

    def __init__(
        self,
        output_dir: Path,
        resolution: float = 1.0,
        target_srs: str | None = None,
    ):
        self.output_dir = output_dir
        self.resolution = resolution
        self.target_srs = target_srs

    def process_point_cloud(self, input_path: Path) -> ProcessedTile:
        """Process a point cloud file through the full pipeline.

        Requires PDAL to be installed (remote worker only).
        """
        from magic_eyes.processing.dem import generate_dem

        stem = input_path.stem.replace(".copc", "")
        tile_dir = self.output_dir / stem
        tile_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Generate DEMs
        log.info("pipeline_dem", input=str(input_path))
        dem_path, full_return_path = generate_dem(
            input_path, tile_dir, self.resolution, self.target_srs
        )

        # Step 2: Compute derivatives from ground DEM
        return self._process_dem(dem_path, full_return_path, tile_dir)

    def process_dem_file(self, dem_path: Path) -> ProcessedTile:
        """Process from an existing DEM file (no PDAL needed).

        Useful for development/testing on machines without PDAL.
        """
        stem = dem_path.stem
        tile_dir = self.output_dir / stem
        tile_dir.mkdir(parents=True, exist_ok=True)
        return self._process_dem(dem_path, None, tile_dir)

    def _process_dem(
        self,
        dem_path: Path,
        full_return_path: Path | None,
        tile_dir: Path,
    ) -> ProcessedTile:
        """Compute all derivatives from a DEM."""
        log.info("pipeline_derivatives", dem=str(dem_path))

        dem, transform, crs = read_dem(dem_path)

        # Handle nodata
        nodata_mask = np.isnan(dem) | (dem < -9000)
        if np.any(nodata_mask):
            dem[nodata_mask] = np.nanmean(dem[~nodata_mask])

        # Compute all derivatives
        derivatives = compute_all_derivatives(dem, self.resolution)

        # Write each derivative to disk
        derivative_paths: dict[str, Path] = {}
        for name, array in derivatives.items():
            out_path = tile_dir / f"{name}.tif"
            write_raster(out_path, array, transform, crs)
            derivative_paths[name] = out_path

        log.info(
            "pipeline_complete",
            tile_dir=str(tile_dir),
            n_derivatives=len(derivative_paths),
        )

        return ProcessedTile(
            dem_path=dem_path,
            full_return_dem_path=full_return_path,
            derivative_paths=derivative_paths,
            resolution_m=self.resolution,
            crs=crs,
        )

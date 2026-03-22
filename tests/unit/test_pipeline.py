"""Unit tests for the processing pipeline and tile manager."""

import tempfile
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box

from magic_eyes.processing.pipeline import ProcessingPipeline
from magic_eyes.processing.tile_manager import ManagedTile, TileManager
from magic_eyes.processing.point_cloud import (
    compute_multi_return_ratio,
    compute_point_density,
)


def _write_test_dem(path: Path, size: int = 100, depression: bool = True) -> None:
    """Write a test DEM GeoTIFF with an optional depression."""
    dem = np.full((size, size), 500.0, dtype=np.float32)
    if depression:
        y, x = np.mgrid[0:size, 0:size].astype(np.float32)
        dist = np.sqrt((x - size / 2) ** 2 + (y - size / 2) ** 2)
        dem -= 3.0 * np.exp(-dist ** 2 / (2 * 15 ** 2))

    transform = from_bounds(0, 0, size, size, size, size)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="float32",
        crs="EPSG:32617",
        transform=transform,
    ) as dst:
        dst.write(dem, 1)


class TestProcessingPipeline:
    def test_process_dem_file(self):
        """Pipeline should produce all derivatives from a DEM GeoTIFF."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            dem_path = tmpdir / "test_dem.tif"
            _write_test_dem(dem_path, size=80)

            output_dir = tmpdir / "output"
            pipeline = ProcessingPipeline(output_dir=output_dir, resolution=1.0)
            result = pipeline.process_dem_file(dem_path)

            assert result.dem_path == dem_path
            assert len(result.derivative_paths) >= 10

            # Check key derivatives exist on disk
            for name in ["hillshade", "slope", "svf", "fill_difference"]:
                assert name in result.derivative_paths
                assert result.derivative_paths[name].exists()

    def test_derivatives_are_valid_geotiffs(self):
        """Each derivative should be a readable GeoTIFF with correct shape."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            dem_path = tmpdir / "test.tif"
            _write_test_dem(dem_path, size=60)

            output_dir = tmpdir / "output"
            pipeline = ProcessingPipeline(output_dir=output_dir)
            result = pipeline.process_dem_file(dem_path)

            for name, path in result.derivative_paths.items():
                with rasterio.open(path) as src:
                    data = src.read(1)
                    assert data.shape == (60, 60), f"{name} has wrong shape"
                    assert data.dtype == np.float32, f"{name} has wrong dtype"


class TestTileManager:
    def test_add_and_query(self):
        tm = TileManager()
        tile = ManagedTile(
            tile_id=uuid4(),
            bbox=box(-79.8, 39.7, -79.7, 39.8),
        )
        tm.add_tile(tile)
        assert tm.count() == 1

        results = tm.query_bbox(-80.0, 39.5, -79.5, 40.0)
        assert len(results) == 1
        assert results[0].tile_id == tile.tile_id

    def test_no_results_outside_bbox(self):
        tm = TileManager()
        tm.add_tile(ManagedTile(tile_id=uuid4(), bbox=box(-79.8, 39.7, -79.7, 39.8)))

        results = tm.query_bbox(-75.0, 40.0, -74.0, 41.0)
        assert len(results) == 0

    def test_multiple_tiles(self):
        tm = TileManager()
        for i in range(5):
            tm.add_tile(ManagedTile(
                tile_id=uuid4(),
                bbox=box(-80 + i * 0.1, 39.7, -79.9 + i * 0.1, 39.8),
            ))
        assert tm.count() == 5

        # Query that intersects first 3
        results = tm.query_bbox(-80.1, 39.6, -79.75, 39.9)
        assert len(results) >= 2

    def test_get_neighbors(self):
        tm = TileManager()
        center = ManagedTile(tile_id=uuid4(), bbox=box(-79.8, 39.7, -79.7, 39.8))
        neighbor = ManagedTile(tile_id=uuid4(), bbox=box(-79.7, 39.7, -79.6, 39.8))
        far = ManagedTile(tile_id=uuid4(), bbox=box(-75.0, 40.0, -74.9, 40.1))

        tm.add_tile(center)
        tm.add_tile(neighbor)
        tm.add_tile(far)

        neighbors = tm.get_neighbors(center, buffer_m=5000)
        neighbor_ids = {t.tile_id for t in neighbors}
        assert neighbor.tile_id in neighbor_ids
        assert far.tile_id not in neighbor_ids

    def test_query_polygon(self):
        tm = TileManager()
        tm.add_tile(ManagedTile(tile_id=uuid4(), bbox=box(-79.8, 39.7, -79.7, 39.8)))

        poly = box(-80.0, 39.5, -79.5, 40.0)
        results = tm.query_polygon(poly)
        assert len(results) == 1


class TestPointDensity:
    def test_uniform_density(self):
        rng = np.random.default_rng(42)
        n = 10000
        x = rng.uniform(0, 100, n)
        y = rng.uniform(0, 100, n)
        z = rng.uniform(0, 10, n)

        density, z_scores, bounds = compute_point_density(x, y, z, cell_size=5.0)
        # Uniform distribution should have z-scores near 0
        assert abs(np.mean(z_scores[density > 0])) < 1.0

    def test_void_detected(self):
        """An area with no points should have strongly negative z-score."""
        rng = np.random.default_rng(42)
        n = 10000
        x = rng.uniform(0, 100, n)
        y = rng.uniform(0, 100, n)
        z = rng.uniform(0, 10, n)

        # Remove points from a 10x10 area (simulating a void/cave opening)
        void_mask = (x > 45) & (x < 55) & (y > 45) & (y < 55)
        x = x[~void_mask]
        y = y[~void_mask]
        z = z[~void_mask]

        density, z_scores, bounds = compute_point_density(x, y, z, cell_size=5.0)
        # The void cells should have low/zero density
        void_row = int((100 - 50) / 5.0)
        void_col = int(50 / 5.0)
        assert density[void_row, void_col] < density.mean() * 0.3


class TestMultiReturnRatio:
    def test_single_return_zero_ratio(self):
        """All single-return points should give ratio ~0."""
        n = 1000
        x = np.random.uniform(0, 100, n)
        y = np.random.uniform(0, 100, n)
        rn = np.ones(n, dtype=np.int32)
        nr = np.ones(n, dtype=np.int32)

        ratio, _ = compute_multi_return_ratio(x, y, rn, nr, cell_size=10.0)
        assert np.max(ratio) < 0.01

    def test_all_multi_return_high_ratio(self):
        """All multi-return points should give ratio ~1."""
        n = 1000
        x = np.random.uniform(0, 100, n)
        y = np.random.uniform(0, 100, n)
        rn = np.ones(n, dtype=np.int32)
        nr = np.full(n, 3, dtype=np.int32)  # 3 returns per pulse

        ratio, _ = compute_multi_return_ratio(x, y, rn, nr, cell_size=10.0)
        assert np.mean(ratio[ratio > 0]) > 0.9

    def test_vegetation_excluded(self):
        """Vegetation-classified points should be excluded."""
        n = 1000
        x = np.random.uniform(0, 100, n)
        y = np.random.uniform(0, 100, n)
        rn = np.ones(n, dtype=np.int32)
        nr = np.full(n, 3, dtype=np.int32)
        # All points are vegetation (class 3)
        classification = np.full(n, 3, dtype=np.int32)

        ratio, _ = compute_multi_return_ratio(
            x, y, rn, nr, classification=classification, cell_size=10.0
        )
        # All excluded, ratio should be 0 everywhere
        assert np.max(ratio) < 0.01

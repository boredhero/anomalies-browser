"""Comprehensive tests for all detection passes against synthetic DEMs."""

from pathlib import Path

import numpy as np
from rasterio.transform import from_bounds

from magic_eyes.detection.base import FeatureType, PassInput
from magic_eyes.detection.passes.curvature import CurvaturePass
from magic_eyes.detection.passes.local_relief_model import LocalReliefModelPass
from magic_eyes.detection.passes.morphometric_filter import MorphometricFilterPass
from magic_eyes.detection.passes.multi_return import MultiReturnPass
from magic_eyes.detection.passes.point_density import PointDensityPass
from magic_eyes.detection.passes.sky_view_factor import SkyViewFactorPass
from magic_eyes.detection.passes.tpi import TPIPass
from magic_eyes.detection.registry import PassRegistry
from magic_eyes.detection.runner import PassRunner
from tests.fixtures.synthetic_dem import (
    make_flat_dem,
    make_slope_dem,
)

# --- Helpers ---

def _make_deep_pit(size=200, depth=5.0, radius=12.0):
    """Create a DEM with a sharp conical pit for reliable detection.

    Uses a conical pit (not gaussian) to ensure well-defined edges.
    """
    dem = np.full((size, size), 500.0, dtype=np.float32)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    dist = np.sqrt((x - size / 2) ** 2 + (y - size / 2) ** 2)
    # Conical pit with sharp edges
    pit_mask = dist < radius
    dem[pit_mask] = 500.0 - depth * (1 - dist[pit_mask] / radius)
    transform = from_bounds(0, 0, size, size, size, size)
    return PassInput(dem=dem, transform=transform, crs=32617, derivatives={})


# --- Registry Tests ---

class TestAllPassesRegistered:
    def test_all_8_passes_registered(self):
        passes = PassRegistry.list_passes()
        expected = [
            "fill_difference", "local_relief_model", "curvature",
            "sky_view_factor", "tpi", "point_density", "multi_return",
            "morphometric_filter",
        ]
        for name in expected:
            assert name in passes, f"Pass {name!r} not registered"

    def test_pass_count(self):
        assert len(PassRegistry.list_passes()) == 11  # 8 classical + 3 ML


# --- LocalReliefModelPass ---

class TestLocalReliefModelPass:
    def setup_method(self):
        self.p = LocalReliefModelPass()

    def test_detects_sinkhole(self):
        inp = _make_deep_pit(depth=4.0, radius=20.0)
        candidates = self.p.run(inp)
        assert len(candidates) >= 1
        assert candidates[0].morphometrics["lrm_anomaly_m"] > 1.0

    def test_no_false_pos_flat(self):
        inp = make_flat_dem()
        assert len(self.p.run(inp)) == 0

    def test_no_false_pos_slope(self):
        inp = make_slope_dem(slope_deg=10.0)
        assert len(self.p.run(inp)) == 0

    def test_feature_type_is_cave(self):
        inp = _make_deep_pit(depth=6.0, radius=20.0)
        candidates = self.p.run(inp)
        assert len(candidates) >= 1, "LRM should detect the deep pit"
        assert candidates[0].feature_type == FeatureType.CAVE_ENTRANCE


# --- CurvaturePass ---

class TestCurvaturePass:
    def setup_method(self):
        self.p = CurvaturePass()

    def test_detects_depression(self):
        inp = _make_deep_pit(depth=5.0, radius=15.0)
        inp.config = {"threshold": -0.005}
        candidates = self.p.run(inp)
        assert len(candidates) >= 1

    def test_no_false_pos_flat(self):
        inp = make_flat_dem()
        assert len(self.p.run(inp)) == 0


# --- SkyViewFactorPass ---

class TestSkyViewFactorPass:
    def setup_method(self):
        self.p = SkyViewFactorPass()

    def test_detects_enclosed_pit(self):
        # Create a steep-walled pit for SVF detection
        size = 100
        dem = np.full((size, size), 500.0, dtype=np.float32)
        y, x = np.mgrid[0:size, 0:size].astype(np.float32)
        dist = np.sqrt((x - 50) ** 2 + (y - 50) ** 2)
        # Steep conical pit: 15m deep, 15m radius
        pit_mask = dist < 15
        dem[pit_mask] = 500.0 - 15.0 * (1 - dist[pit_mask] / 15.0)
        transform = from_bounds(0, 0, size, size, size, size)
        inp = PassInput(dem=dem, transform=transform, crs=32617, derivatives={})
        inp.config = {"threshold": 0.9, "radius_m": 15.0, "n_directions": 8}
        candidates = self.p.run(inp)
        assert len(candidates) >= 1

    def test_no_false_pos_flat(self):
        inp = make_flat_dem(size=80)
        inp.config = {"radius_m": 10.0, "n_directions": 8}
        assert len(self.p.run(inp)) == 0


# --- TPIPass ---

class TestTPIPass:
    def setup_method(self):
        self.p = TPIPass()

    def test_detects_depression(self):
        inp = _make_deep_pit(depth=5.0, radius=20.0)
        inp.config = {"threshold": -0.5, "radius_m": 15.0}
        candidates = self.p.run(inp)
        assert len(candidates) >= 1
        assert candidates[0].feature_type == FeatureType.SINKHOLE

    def test_no_false_pos_flat(self):
        inp = make_flat_dem()
        assert len(self.p.run(inp)) == 0


# --- MorphometricFilterPass ---

class TestMorphometricFilterPass:
    def setup_method(self):
        self.p = MorphometricFilterPass()

    def test_computes_full_morphometrics(self):
        inp = _make_deep_pit(depth=6.0, radius=25.0)
        inp.config = {"min_depth_m": 0.3, "min_area_m2": 5.0, "min_circularity": 0.1}
        candidates = self.p.run(inp)
        assert len(candidates) >= 1
        m = candidates[0].morphometrics
        assert "depth_m" in m
        assert "area_m2" in m
        assert "circularity" in m
        assert "volume_m3" in m
        assert "k_parameter" in m
        assert "elongation" in m
        assert "wall_slope_deg" in m
        assert "depth_area_ratio" in m

    def test_depth_approximately_correct(self):
        inp = _make_deep_pit(depth=6.0, radius=25.0)
        inp.config = {"min_depth_m": 0.3, "min_area_m2": 5.0, "min_circularity": 0.1}
        candidates = self.p.run(inp)
        depth = candidates[0].morphometrics["depth_m"]
        assert 2.0 < depth < 8.0, f"Expected ~6m depth, got {depth}"

    def test_circularity_high_for_circular_pit(self):
        inp = _make_deep_pit(depth=6.0, radius=25.0)
        inp.config = {"min_depth_m": 0.3, "min_area_m2": 5.0, "min_circularity": 0.1}
        candidates = self.p.run(inp)
        circ = candidates[0].morphometrics["circularity"]
        assert circ > 0.3, f"Circular pit should have high circularity, got {circ}"

    def test_no_false_pos_flat(self):
        inp = make_flat_dem()
        assert len(self.p.run(inp)) == 0

    def test_classifies_feature_type(self):
        inp = _make_deep_pit(depth=6.0, radius=25.0)
        inp.config = {"min_depth_m": 0.3, "min_area_m2": 5.0, "min_circularity": 0.1}
        candidates = self.p.run(inp)
        # Should be classified (not UNKNOWN)
        assert candidates[0].feature_type != FeatureType.UNKNOWN


# --- PointDensityPass ---

class TestPointDensityPass:
    def setup_method(self):
        self.p = PointDensityPass()

    def test_returns_empty_without_point_cloud(self):
        inp = make_flat_dem()
        assert len(self.p.run(inp)) == 0

    def test_detects_void_in_point_cloud(self):
        rng = np.random.default_rng(42)
        n = 5000
        x = rng.uniform(0, 100, n)
        y = rng.uniform(0, 100, n)
        z = rng.uniform(0, 10, n)

        # Create void at center
        void_mask = (x > 40) & (x < 60) & (y > 40) & (y < 60)
        pc = {
            "X": x[~void_mask],
            "Y": y[~void_mask],
            "Z": z[~void_mask],
        }

        inp = make_flat_dem(size=100)
        inp.point_cloud = pc
        inp.config = {"cell_size_m": 5.0, "z_score_threshold": -1.5}
        candidates = self.p.run(inp)
        assert len(candidates) >= 1


# --- MultiReturnPass ---

class TestMultiReturnPass:
    def setup_method(self):
        self.p = MultiReturnPass()

    def test_returns_empty_without_point_cloud(self):
        inp = make_flat_dem()
        assert len(self.p.run(inp)) == 0

    def test_detects_anomalous_returns(self):
        rng = np.random.default_rng(42)
        n = 5000
        x = rng.uniform(0, 100, n)
        y = rng.uniform(0, 100, n)
        rn = np.ones(n, dtype=np.int32)
        nr = np.ones(n, dtype=np.int32)

        # Make a cluster of multi-return points in one area
        cluster = (x > 40) & (x < 60) & (y > 40) & (y < 60)
        nr[cluster] = 4  # multi-return

        pc = {
            "X": x,
            "Y": y,
            "ReturnNumber": rn,
            "NumberOfReturns": nr,
        }

        inp = make_flat_dem(size=100)
        inp.point_cloud = pc
        inp.config = {"search_radius_m": 10.0, "min_multi_return_ratio": 0.3}
        candidates = self.p.run(inp)
        assert len(candidates) >= 1


# --- PassRunner with TOML ---

class TestPassRunnerToml:
    def test_load_cave_hunting_config(self):
        toml_path = Path(__file__).parent.parent.parent / "configs" / "passes" / "cave_hunting.toml"
        runner = PassRunner.from_toml(toml_path)
        assert len(runner.passes) >= 5  # point_density/multi_return skip without PC

    def test_load_sinkhole_survey_config(self):
        toml_path = Path(__file__).parent.parent.parent / "configs" / "passes" / "sinkhole_survey.toml"
        runner = PassRunner.from_toml(toml_path)
        assert len(runner.passes) >= 4

    def test_run_sinkhole_config_on_synthetic(self):
        toml_path = Path(__file__).parent.parent.parent / "configs" / "passes" / "sinkhole_survey.toml"
        runner = PassRunner.from_toml(toml_path)
        inp = _make_deep_pit(depth=4.0, radius=20.0)
        candidates = runner.run_on_array(
            inp.dem, inp.transform, inp.crs, inp.derivatives
        )
        assert len(candidates) >= 1, "Sinkhole config should detect a deep pit"

    def test_run_cave_config_on_synthetic(self):
        toml_path = Path(__file__).parent.parent.parent / "configs" / "passes" / "cave_hunting.toml"
        runner = PassRunner.from_toml(toml_path)
        inp = _make_deep_pit(depth=5.0, radius=15.0)
        candidates = runner.run_on_array(
            inp.dem, inp.transform, inp.crs, inp.derivatives
        )
        assert len(candidates) >= 1, "Cave config should detect a deep pit"

    def test_no_false_pos_flat_with_config(self):
        toml_path = Path(__file__).parent.parent.parent / "configs" / "passes" / "sinkhole_survey.toml"
        runner = PassRunner.from_toml(toml_path)
        inp = make_flat_dem()
        candidates = runner.run_on_array(
            inp.dem, inp.transform, inp.crs, inp.derivatives
        )
        assert len(candidates) == 0, "Flat terrain should produce no detections"

"""Unit tests for terrain derivative computations."""

import numpy as np
import pytest

from magic_eyes.processing.derivatives import (
    compute_all_derivatives,
    compute_curvature,
    compute_fill_difference,
    compute_hillshade,
    compute_lrm,
    compute_lrm_multiscale,
    compute_slope,
    compute_svf,
    compute_tpi,
    compute_tpi_multiscale,
)


class TestHillshade:
    def test_flat_terrain_uniform(self):
        dem = np.full((50, 50), 100.0, dtype=np.float32)
        hs = compute_hillshade(dem)
        # Flat terrain should have uniform hillshade
        assert np.std(hs) < 1.0

    def test_output_range(self):
        dem = np.random.rand(50, 50).astype(np.float32) * 100
        hs = compute_hillshade(dem)
        assert hs.min() >= 0
        assert hs.max() <= 255

    def test_shape_preserved(self):
        dem = np.zeros((100, 80), dtype=np.float32)
        hs = compute_hillshade(dem)
        assert hs.shape == (100, 80)


class TestSlope:
    def test_flat_terrain_zero_slope(self):
        dem = np.full((50, 50), 500.0, dtype=np.float32)
        slope = compute_slope(dem)
        assert np.allclose(slope, 0, atol=0.01)

    def test_45_degree_slope(self):
        # Create a surface that rises 1m per 1m horizontal
        dem = np.arange(50, dtype=np.float32).reshape(1, -1).repeat(50, axis=0)
        slope = compute_slope(dem, resolution=1.0)
        # Interior should be ~45 degrees
        interior = slope[5:-5, 5:-5]
        assert np.mean(interior) == pytest.approx(45.0, abs=5.0)

    def test_slope_degrees_vs_radians(self):
        dem = np.arange(50, dtype=np.float32).reshape(1, -1).repeat(50, axis=0) * 0.5
        slope_deg = compute_slope(dem, degrees=True)
        slope_rad = compute_slope(dem, degrees=False)
        np.testing.assert_allclose(
            np.radians(slope_deg[10:-10, 10:-10]),
            slope_rad[10:-10, 10:-10],
            atol=0.01,
        )


class TestCurvature:
    def test_flat_zero_curvature(self):
        dem = np.full((50, 50), 500.0, dtype=np.float32)
        curv = compute_curvature(dem, curvature_type="profile")
        assert np.allclose(curv, 0, atol=1e-6)

    def test_convex_hilltop_vs_concave_depression(self):
        """Profile curvature should differ between hilltops and depressions."""
        size = 80
        y, x = np.mgrid[0:size, 0:size].astype(np.float32)
        dist = np.sqrt((x - 40) ** 2 + (y - 40) ** 2)

        # Hilltop (convex)
        hilltop = 500 + 10 * np.exp(-dist ** 2 / (2 * 15 ** 2))
        curv_hill = compute_curvature(hilltop.astype(np.float32), curvature_type="total")

        # Depression (concave)
        depression = 500 - 10 * np.exp(-dist ** 2 / (2 * 15 ** 2))
        curv_dep = compute_curvature(depression.astype(np.float32), curvature_type="total")

        # Sign should flip between hilltop and depression at center
        assert curv_hill[40, 40] != pytest.approx(curv_dep[40, 40], abs=0.001), \
            "Curvature should differ between hilltop and depression"
        # They should have opposite signs
        assert curv_hill[40, 40] * curv_dep[40, 40] < 0, \
            "Hilltop and depression should have opposite curvature signs"

    def test_plan_curvature_exists(self):
        dem = np.random.rand(50, 50).astype(np.float32) * 100
        curv = compute_curvature(dem, curvature_type="plan")
        assert curv.shape == (50, 50)
        assert np.any(curv != 0)


class TestTPI:
    def test_flat_zero_tpi(self):
        dem = np.full((100, 100), 500.0, dtype=np.float32)
        tpi = compute_tpi(dem, radius_pixels=10)
        assert np.allclose(tpi, 0, atol=0.01)

    def test_depression_negative_tpi(self):
        """A pit should have negative TPI at center."""
        dem = np.full((100, 100), 500.0, dtype=np.float32)
        # Create a pit at center
        y, x = np.mgrid[0:100, 0:100].astype(np.float32)
        dist = np.sqrt((x - 50) ** 2 + (y - 50) ** 2)
        dem[dist < 10] = 495.0  # 5m deep pit

        tpi = compute_tpi(dem, radius_pixels=15)
        assert tpi[50, 50] < -1.0, "Pit center should have strongly negative TPI"

    def test_ridge_positive_tpi(self):
        """A mound should have positive TPI at peak."""
        dem = np.full((100, 100), 500.0, dtype=np.float32)
        y, x = np.mgrid[0:100, 0:100].astype(np.float32)
        dist = np.sqrt((x - 50) ** 2 + (y - 50) ** 2)
        dem[dist < 10] = 505.0  # 5m mound

        tpi = compute_tpi(dem, radius_pixels=15)
        assert tpi[50, 50] > 1.0, "Mound peak should have positive TPI"

    def test_multiscale(self):
        dem = np.random.rand(100, 100).astype(np.float32) * 10 + 500
        result = compute_tpi_multiscale(dem, resolution=1.0)
        assert "tpi_5m" in result
        assert "tpi_15m" in result
        assert "tpi_50m" in result


class TestSVF:
    def test_flat_terrain_high_svf(self):
        """Flat terrain should have SVF close to 1.0 (fully open sky)."""
        dem = np.full((60, 60), 500.0, dtype=np.float32)
        svf = compute_svf(dem, radius_m=10.0, n_directions=8)
        # Interior should be ~1.0
        interior = svf[15:-15, 15:-15]
        assert np.mean(interior) > 0.9

    def test_depression_low_svf(self):
        """A deep pit should have low SVF at center."""
        dem = np.full((100, 100), 500.0, dtype=np.float32)
        y, x = np.mgrid[0:100, 0:100].astype(np.float32)
        dist = np.sqrt((x - 50) ** 2 + (y - 50) ** 2)
        dem -= 10.0 * np.exp(-dist ** 2 / (2 * 10 ** 2))  # deep gaussian pit

        svf = compute_svf(dem, resolution=1.0, radius_m=20.0, n_directions=8)
        center_svf = svf[50, 50]
        edge_svf = svf[5, 5]
        assert center_svf < edge_svf, "Pit center should have lower SVF than edge"

    def test_output_range(self):
        dem = np.random.rand(50, 50).astype(np.float32) * 50 + 500
        svf = compute_svf(dem, radius_m=10.0, n_directions=8)
        assert svf.min() >= 0
        assert svf.max() <= 1.1  # slight numerical tolerance


class TestLRM:
    def test_flat_zero_lrm(self):
        dem = np.full((100, 100), 500.0, dtype=np.float32)
        lrm = compute_lrm(dem, kernel_m=20.0)
        assert np.allclose(lrm, 0, atol=0.1)

    def test_depression_negative_lrm(self):
        """A sinkhole should show as negative LRM."""
        dem = np.full((200, 200), 500.0, dtype=np.float32)
        y, x = np.mgrid[0:200, 0:200].astype(np.float32)
        dist = np.sqrt((x - 100) ** 2 + (y - 100) ** 2)
        dem -= 5.0 * np.exp(-dist ** 2 / (2 * 15 ** 2))

        lrm = compute_lrm(dem, resolution=1.0, kernel_m=50.0)
        assert lrm[100, 100] < -1.0, "Depression should have negative LRM"

    def test_multiscale(self):
        dem = np.random.rand(200, 200).astype(np.float32) * 10 + 500
        result = compute_lrm_multiscale(dem, resolution=1.0)
        assert "lrm_50m" in result
        assert "lrm_100m" in result
        assert "lrm_200m" in result


class TestFillDifference:
    def test_flat_zero_fill(self):
        dem = np.full((50, 50), 500.0, dtype=np.float32)
        fd = compute_fill_difference(dem)
        assert np.allclose(fd, 0, atol=0.01)

    def test_depression_positive_fill(self):
        """Fill-difference should be positive where depressions exist."""
        dem = np.full((100, 100), 500.0, dtype=np.float32)
        dem[40:60, 40:60] = 495.0  # square pit
        fd = compute_fill_difference(dem)
        assert fd[50, 50] > 3.0, "Pit center should have positive fill-difference"

    def test_no_depression_zero_fill(self):
        # Uniform slope — no depressions
        dem = np.arange(100, dtype=np.float32).reshape(1, -1).repeat(100, axis=0)
        fd = compute_fill_difference(dem)
        assert np.max(fd) < 0.1


class TestComputeAll:
    def test_all_derivatives_returned(self):
        dem = np.random.rand(80, 80).astype(np.float32) * 50 + 500
        derivs = compute_all_derivatives(dem, resolution=1.0)

        expected_keys = [
            "hillshade", "slope", "profile_curvature", "plan_curvature",
            "svf", "tpi_5m", "tpi_15m", "tpi_50m",
            "lrm_50m", "lrm_100m", "lrm_200m", "fill_difference",
        ]
        for key in expected_keys:
            assert key in derivs, f"Missing derivative: {key}"
            assert derivs[key].shape == dem.shape, f"Shape mismatch for {key}"
            assert derivs[key].dtype == np.float32, f"Wrong dtype for {key}"

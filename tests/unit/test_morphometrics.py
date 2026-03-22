"""Unit tests for morphometric computations."""

import numpy as np
import pytest

from magic_eyes.detection.postprocess.morphometrics import (
    compute_area,
    compute_circularity,
    compute_depth,
    compute_elongation,
    compute_k_parameter,
    compute_volume,
)


class TestMorphometrics:
    def test_depth_of_simple_pit(self):
        dem = np.array([[10, 10, 10], [10, 5, 10], [10, 10, 10]], dtype=np.float32)
        mask = np.array(
            [[True, True, True], [True, True, True], [True, True, True]]
        )
        assert compute_depth(dem, mask) == pytest.approx(5.0)

    def test_depth_empty_mask(self):
        dem = np.ones((3, 3), dtype=np.float32)
        mask = np.zeros((3, 3), dtype=bool)
        assert compute_depth(dem, mask) == 0.0

    def test_area_computation(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True  # 4x4 = 16 pixels
        assert compute_area(mask, resolution_m=2.0) == pytest.approx(64.0)

    def test_circularity_perfect_circle(self):
        # Approximate: a large filled circle should approach 1.0
        size = 100
        y, x = np.mgrid[0:size, 0:size]
        mask = (x - 50) ** 2 + (y - 50) ** 2 < 30**2
        area = float(np.sum(mask))
        perimeter = float(np.sum(mask & ~np.roll(mask, 1, axis=0)))  # rough
        circ = compute_circularity(area, perimeter)
        assert circ > 0.5  # not perfect due to discrete approximation

    def test_circularity_zero_perimeter(self):
        assert compute_circularity(100.0, 0.0) == 0.0

    def test_k_parameter_cylinder(self):
        # k ≈ 1 for cylinder: volume = area * depth
        area, depth = 100.0, 5.0
        volume = area * depth  # cylinder
        assert compute_k_parameter(area, depth, volume) == pytest.approx(1.0)

    def test_k_parameter_cone(self):
        # k ≈ 3 for cone: volume = area * depth / 3
        area, depth = 100.0, 5.0
        volume = area * depth / 3.0
        assert compute_k_parameter(area, depth, volume) == pytest.approx(3.0)

    def test_volume_computation(self):
        dem = np.array([[10, 10, 10], [10, 7, 10], [10, 10, 10]], dtype=np.float32)
        mask = np.ones((3, 3), dtype=bool)
        vol = compute_volume(dem, mask, resolution_m=1.0)
        # Rim = 10, depths: 0,0,0,0,3,0,0,0,0 => volume = 3 * 1 = 3
        assert vol == pytest.approx(3.0)

    def test_elongation_circular(self):
        size = 50
        y, x = np.mgrid[0:size, 0:size]
        mask = (x - 25) ** 2 + (y - 25) ** 2 < 10**2
        elong = compute_elongation(mask)
        assert elong > 0.8  # approximately circular

    def test_elongation_elongated(self):
        mask = np.zeros((50, 200), dtype=bool)
        mask[20:30, 10:190] = True  # very elongated
        elong = compute_elongation(mask)
        assert elong < 0.4  # clearly elongated

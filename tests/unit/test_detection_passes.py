"""Unit tests for detection passes against synthetic DEMs."""

import pytest

from magic_eyes.detection.passes.fill_difference import FillDifferencePass
from tests.fixtures.synthetic_dem import (
    make_conical_pit_dem,
    make_flat_dem,
    make_multi_depression_dem,
    make_sinkhole_dem,
    make_slope_dem,
)


class TestFillDifferencePass:
    def setup_method(self):
        self.pass_instance = FillDifferencePass()

    def test_detects_gaussian_sinkhole(self):
        input_data = make_sinkhole_dem(depth=3.0, radius=15.0)
        candidates = self.pass_instance.run(input_data)
        assert len(candidates) >= 1, "Should detect the sinkhole"
        best = max(candidates, key=lambda c: c.score)
        assert abs(best.morphometrics["depth_m"] - 3.0) < 1.0
        assert best.morphometrics["area_m2"] > 0

    def test_detects_conical_pit(self):
        input_data = make_conical_pit_dem(depth=5.0, radius=10.0)
        candidates = self.pass_instance.run(input_data)
        assert len(candidates) >= 1, "Should detect the conical pit"
        best = max(candidates, key=lambda c: c.score)
        assert best.morphometrics["depth_m"] > 2.0

    def test_no_false_positives_on_flat(self):
        input_data = make_flat_dem()
        candidates = self.pass_instance.run(input_data)
        assert len(candidates) == 0, "Should not detect anything on flat terrain"

    def test_no_false_positives_on_slope(self):
        input_data = make_slope_dem(slope_deg=15.0)
        candidates = self.pass_instance.run(input_data)
        assert len(candidates) == 0, "Should not detect anything on uniform slope"

    def test_rejects_shallow_depression(self):
        input_data = make_sinkhole_dem(depth=0.2, radius=15.0)
        input_data.config = {"min_depth_m": 0.5}
        candidates = self.pass_instance.run(input_data)
        assert len(candidates) == 0, "Should reject depression shallower than threshold"

    def test_detects_multiple_depressions(self):
        input_data = make_multi_depression_dem()
        candidates = self.pass_instance.run(input_data)
        # Should detect at least the 3 depressions above min_depth (0.5m)
        assert len(candidates) >= 2, f"Expected >=2 detections, got {len(candidates)}"

    def test_respects_max_area_filter(self):
        input_data = make_sinkhole_dem(depth=3.0, radius=80.0)  # very large
        input_data.config = {"max_area_m2": 1000.0}
        candidates = self.pass_instance.run(input_data)
        for c in candidates:
            assert c.morphometrics["area_m2"] <= 1000.0


class TestPassRegistry:
    def test_fill_difference_registered(self):
        from magic_eyes.detection.registry import PassRegistry

        passes = PassRegistry.list_passes()
        assert "fill_difference" in passes

    def test_get_pass_chain(self):
        from magic_eyes.detection.registry import PassRegistry

        chain = PassRegistry.get_pass_chain(["fill_difference"])
        assert len(chain) == 1
        assert chain[0].name == "fill_difference"

    def test_unknown_pass_raises(self):
        from magic_eyes.detection.registry import PassRegistry

        with pytest.raises(KeyError, match="nonexistent"):
            PassRegistry.get("nonexistent")

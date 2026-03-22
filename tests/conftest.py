"""Shared test fixtures."""

import json
from pathlib import Path

import pytest

from magic_eyes.detection.registry import PassRegistry


@pytest.fixture(autouse=True)
def ensure_passes_registered():
    """Ensure all passes are registered before each test."""
    import importlib
    import magic_eyes.detection.passes as passes_mod
    from magic_eyes.detection.passes import (
        fill_difference, local_relief_model, curvature,
        sky_view_factor, tpi, point_density, multi_return,
        morphometric_filter, random_forest, unet_segmentation, yolo_detector,
    )

    # Force re-registration if registry was cleared
    for mod in [fill_difference, local_relief_model, curvature, sky_view_factor,
                tpi, point_density, multi_return, morphometric_filter,
                random_forest, unet_segmentation, yolo_detector]:
        importlib.reload(mod)
    importlib.reload(passes_mod)
    yield


@pytest.fixture
def known_sites():
    """Load known validation sites from JSON fixture."""
    path = Path(__file__).parent / "fixtures" / "known_sites.json"
    with open(path) as f:
        data = json.load(f)
    return data["validation_sites"]

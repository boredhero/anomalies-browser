"""Shared test fixtures."""

import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def ensure_passes_registered():
    """Ensure all passes are registered before each test."""
    import importlib

    import hole_finder.detection.passes as passes_mod
    from hole_finder.detection.passes import (
        curvature,
        fill_difference,
        local_relief_model,
        morphometric_filter,
        multi_return,
        point_density,
        random_forest,
        sky_view_factor,
        tpi,
        unet_segmentation,
        yolo_detector,
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

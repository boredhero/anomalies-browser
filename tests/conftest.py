"""Shared test fixtures."""

import json
from pathlib import Path

import pytest

from magic_eyes.detection.registry import PassRegistry


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear pass registry before each test to avoid cross-contamination."""
    PassRegistry.clear()
    # Re-import to re-register passes
    import magic_eyes.detection.passes  # noqa: F401
    yield
    PassRegistry.clear()


@pytest.fixture
def known_sites():
    """Load known validation sites from JSON fixture."""
    path = Path(__file__).parent / "fixtures" / "known_sites.json"
    with open(path) as f:
        data = json.load(f)
    return data["validation_sites"]

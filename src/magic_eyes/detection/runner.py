"""Pass runner: orchestrates detection pass chains on tiles."""

from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import rasterio

from magic_eyes.detection.base import Candidate, DetectionPass, PassInput
from magic_eyes.detection.fusion import ResultFuser
from magic_eyes.detection.registry import PassRegistry


class PassRunner:
    """Executes a configured chain of detection passes on a tile."""

    def __init__(
        self,
        pass_names: list[str],
        config: dict[str, Any] | None = None,
        weights: dict[str, float] | None = None,
    ):
        self.passes = PassRegistry.get_pass_chain(pass_names)
        self.config = config or {}
        self.fuser = ResultFuser(weights=weights)

    def run_on_dem(
        self,
        dem_path: Path,
        derivatives: dict[str, Path] | None = None,
        point_cloud: Any | None = None,
    ) -> list[Candidate]:
        """Run all passes on a DEM file and return fused candidates."""
        with rasterio.open(dem_path) as src:
            dem = src.read(1).astype(np.float32)
            transform = src.transform
            crs = src.crs.to_epsg() or 32617

        # Load derivative rasters
        loaded_derivatives: dict[str, np.ndarray] = {}
        if derivatives:
            for name, path in derivatives.items():
                with rasterio.open(path) as src:
                    loaded_derivatives[name] = src.read(1).astype(np.float32)

        # Run each pass
        all_candidates: list[tuple[str, Candidate]] = []
        for detection_pass in self.passes:
            pass_config = self.config.get(f"passes.{detection_pass.name}", {})
            pass_input = PassInput(
                dem=dem,
                transform=transform,
                crs=crs,
                derivatives=loaded_derivatives,
                point_cloud=point_cloud if detection_pass.requires_point_cloud else None,
                config=pass_config,
            )
            candidates = detection_pass.run(pass_input)
            for candidate in candidates:
                all_candidates.append((detection_pass.name, candidate))

        # Fuse results from multiple passes
        return self.fuser.fuse(all_candidates)

"""Processing pipeline — PDAL + GDAL + WhiteboxTools subprocesses.

Python does NOT compute derivatives. Everything is done by compiled
native tools (C++/Rust) via subprocess calls that saturate all cores:

- PDAL: point cloud → ground-classified DEM (C++, multi-threaded)
- GDAL: hillshade, slope, aspect, TPI, roughness (C, multi-threaded)
- WhiteboxTools: fill_depressions, SVF, LRM (Rust, single-threaded but fast)

Python only orchestrates subprocesses and reads the resulting GeoTIFFs.
For hundreds of tiles, Celery distributes tiles across workers.
"""

import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from magic_eyes.utils.logging import log


@dataclass
class ProcessedTile:
    """Immutable result of processing a tile. Stored permanently."""

    tile_dir: Path
    dem_path: Path
    filled_dem_path: Path | None = None
    derivative_paths: dict[str, Path] = field(default_factory=dict)
    resolution_m: float = 1.0
    crs: int = 32617


def _run(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    """Run a subprocess, raise on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {result.stderr[:500]}")
    return result


# ---------------------------------------------------------------------------
# Stage 1: Point cloud → DEM via PDAL (C++, uses all cores for I/O)
# ---------------------------------------------------------------------------

def generate_dem_pdal(
    input_path: Path,
    output_dir: Path,
    resolution: float = 1.0,
    target_srs: str | None = None,
) -> tuple[Path, Path]:
    """Generate ground DEM and filled DEM from point cloud using PDAL.

    Returns (dem_path, filled_dem_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem.replace(".copc", "")
    dem_path = output_dir / f"{stem}_dem.tif"
    filled_path = output_dir / f"{stem}_filled.tif"

    if not dem_path.exists():
        pipeline = [
            {"type": "readers.copc" if str(input_path).endswith(".copc.laz") else "readers.las",
             "filename": str(input_path)},
        ]
        if target_srs:
            pipeline.append({"type": "filters.reprojection", "out_srs": target_srs})
        pipeline.extend([
            {"type": "filters.smrf", "slope": 0.15, "window": 18, "threshold": 0.5},
            {"type": "filters.range", "limits": "Classification[2:2]"},
            {"type": "writers.gdal", "filename": str(dem_path), "resolution": resolution,
             "output_type": "idw", "gdalopts": "COMPRESS=DEFLATE,TILED=YES,BLOCKXSIZE=256,BLOCKYSIZE=256",
             "data_type": "float32"},
        ])
        log.info("pdal_dem", input=str(input_path))
        _run(["pdal", "pipeline", "--stdin"], timeout=900)
        # pdal pipeline reads from stdin
        proc = subprocess.run(
            ["pdal", "pipeline", "--stdin"],
            input=json.dumps({"pipeline": pipeline}),
            capture_output=True, text=True, timeout=900,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"PDAL DEM failed: {proc.stderr[:500]}")

    # Generate filled DEM using WhiteboxTools (compiled Rust)
    if not filled_path.exists():
        _wbt_fill_depressions(dem_path, filled_path)

    return dem_path, filled_path


def _wbt_fill_depressions(dem_path: Path, filled_path: Path) -> None:
    """Fill depressions using WhiteboxTools (compiled Rust)."""
    import whitebox
    wbt = whitebox.WhiteboxTools()
    wbt.set_verbose_mode(False)
    wbt.fill_depressions(str(dem_path), str(filled_path))


# ---------------------------------------------------------------------------
# Stage 2: DEM → derivatives via GDAL + WhiteboxTools (compiled, parallel)
# ---------------------------------------------------------------------------

def _gdal_hillshade(dem: str, out: str) -> str:
    _run(["gdaldem", "hillshade", dem, out, "-az", "315", "-alt", "45",
          "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES", "-q"])
    return out


def _gdal_slope(dem: str, out: str) -> str:
    _run(["gdaldem", "slope", dem, out, "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES", "-q"])
    return out


def _gdal_tpi(dem: str, out: str) -> str:
    _run(["gdaldem", "TPI", dem, out, "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES", "-q"])
    return out


def _gdal_roughness(dem: str, out: str) -> str:
    _run(["gdaldem", "roughness", dem, out, "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES", "-q"])
    return out


def _wbt_svf(dem: str, out: str) -> str:
    import whitebox
    wbt = whitebox.WhiteboxTools()
    wbt.set_verbose_mode(False)
    # WhiteboxTools sky_view_factor — compiled Rust, much faster than Python
    wbt.sky_view_factor(dem, out)
    return out


def _wbt_lrm(dem: str, out: str, kernel: int = 100) -> str:
    """Local Relief Model via WhiteboxTools — subtract mean filter from DEM."""
    import whitebox
    wbt = whitebox.WhiteboxTools()
    wbt.set_verbose_mode(False)
    # Use deviation_from_mean as proxy for LRM
    wbt.deviation_from_mean(dem, out, filterx=kernel, filtery=kernel)
    return out


def _gdal_fill_diff(dem: str, filled: str, out: str) -> str:
    """Compute fill_difference = filled - original using gdal_calc."""
    _run(["gdal_calc.py", "-A", filled, "-B", dem,
          "--outfile=" + out, "--calc=A-B", "--type=Float32",
          "--co=COMPRESS=DEFLATE", "--co=TILED=YES", "--quiet"])
    return out


def _gdal_curvature(dem: str, out_profile: str, out_plan: str) -> tuple[str, str]:
    """Compute curvature via gdaldem or GDAL VRT trick.

    gdaldem doesn't directly support curvature, so we use WhiteboxTools.
    """
    import whitebox
    wbt = whitebox.WhiteboxTools()
    wbt.set_verbose_mode(False)
    wbt.profile_curvature(dem, out_profile)
    wbt.plan_curvature(dem, out_plan)
    return out_profile, out_plan


def compute_all_derivatives_native(
    dem_path: Path,
    filled_dem_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Compute all derivatives using GDAL + WhiteboxTools subprocesses.

    Runs independent derivatives in parallel via ProcessPoolExecutor.
    Each task is a subprocess call to a compiled native tool.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dem = str(dem_path)
    filled = str(filled_dem_path)

    # Define all tasks: name → (function, args)
    tasks: dict[str, tuple] = {
        "hillshade": (_gdal_hillshade, dem, str(output_dir / "hillshade.tif")),
        "slope": (_gdal_slope, dem, str(output_dir / "slope.tif")),
        "tpi": (_gdal_tpi, dem, str(output_dir / "tpi.tif")),
        "roughness": (_gdal_roughness, dem, str(output_dir / "roughness.tif")),
        "svf": (_wbt_svf, dem, str(output_dir / "svf.tif")),
        "lrm_50m": (_wbt_lrm, dem, str(output_dir / "lrm_50m.tif"), 50),
        "lrm_100m": (_wbt_lrm, dem, str(output_dir / "lrm_100m.tif"), 100),
        "lrm_200m": (_wbt_lrm, dem, str(output_dir / "lrm_200m.tif"), 200),
        "fill_difference": (_gdal_fill_diff, dem, filled, str(output_dir / "fill_difference.tif")),
    }

    results: dict[str, Path] = {}

    # Run ALL derivatives in parallel as separate processes
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {}
        for name, args in tasks.items():
            fn = args[0]
            fn_args = args[1:]
            out_path = fn_args[-1] if name != "fill_difference" else fn_args[-1]

            # Skip if already computed (persistent cache)
            if Path(out_path).exists():
                results[name] = Path(out_path)
                log.info("derivative_cached", name=name)
                continue

            futures[executor.submit(fn, *fn_args)] = (name, out_path)

        for future in as_completed(futures):
            name, out_path = futures[future]
            try:
                future.result()
                results[name] = Path(out_path)
                log.info("derivative_computed", name=name)
            except Exception as e:
                log.error("derivative_failed", name=name, error=str(e))

    # Curvature requires sequential WhiteboxTools calls (can't parallelize same WBT instance)
    profile_path = output_dir / "profile_curvature.tif"
    plan_path = output_dir / "plan_curvature.tif"
    if not profile_path.exists() or not plan_path.exists():
        try:
            _gdal_curvature(dem, str(profile_path), str(plan_path))
            results["profile_curvature"] = profile_path
            results["plan_curvature"] = plan_path
        except Exception as e:
            log.error("curvature_failed", error=str(e))
    else:
        results["profile_curvature"] = profile_path
        results["plan_curvature"] = plan_path

    return results


# ---------------------------------------------------------------------------
# Full pipeline: point cloud → permanent processed tile
# ---------------------------------------------------------------------------

class ProcessingPipeline:
    """Full tile processing — PDAL + GDAL + WhiteboxTools.

    All outputs are written to persistent storage (SSD). Once computed,
    derivatives are cached permanently and never recomputed unless
    explicitly requested via force=True.
    """

    def __init__(
        self,
        output_dir: Path,
        resolution: float = 1.0,
        target_srs: str | None = None,
    ):
        self.output_dir = output_dir
        self.resolution = resolution
        self.target_srs = target_srs

    def process_point_cloud(self, input_path: Path, force: bool = False) -> ProcessedTile:
        """Process a LAZ/COPC point cloud through the full pipeline.

        Results are permanently cached. Set force=True to recompute.
        """
        stem = input_path.stem.replace(".copc", "")
        tile_dir = self.output_dir / stem
        tile_dir.mkdir(parents=True, exist_ok=True)

        deriv_dir = tile_dir / "derivatives"

        # Check if already fully processed
        marker = tile_dir / ".processed"
        if marker.exists() and not force:
            return self._load_existing(tile_dir, deriv_dir)

        # Stage 1: Point cloud → DEM + filled DEM
        log.info("pipeline_stage1_dem", input=str(input_path))
        dem_path, filled_path = generate_dem_pdal(
            input_path, tile_dir, self.resolution, self.target_srs
        )

        # Stage 2: DEM → all derivatives (parallel native subprocesses)
        log.info("pipeline_stage2_derivatives")
        derivative_paths = compute_all_derivatives_native(dem_path, filled_path, deriv_dir)

        # Mark as processed
        marker.write_text(f"processed\nderivatives: {len(derivative_paths)}\n")

        log.info("pipeline_complete", tile_dir=str(tile_dir), derivatives=len(derivative_paths))
        return ProcessedTile(
            tile_dir=tile_dir,
            dem_path=dem_path,
            filled_dem_path=filled_path,
            derivative_paths=derivative_paths,
            resolution_m=self.resolution,
        )

    def process_dem_file(self, dem_path: Path, force: bool = False) -> ProcessedTile:
        """Process from an existing DEM file (no PDAL needed).

        Generates a filled DEM via WhiteboxTools, then all derivatives.
        """
        stem = dem_path.stem
        tile_dir = self.output_dir / stem
        tile_dir.mkdir(parents=True, exist_ok=True)
        deriv_dir = tile_dir / "derivatives"

        marker = tile_dir / ".processed"
        if marker.exists() and not force:
            return self._load_existing(tile_dir, deriv_dir)

        # Generate filled DEM
        filled_path = tile_dir / f"{stem}_filled.tif"
        if not filled_path.exists():
            _wbt_fill_depressions(dem_path, filled_path)

        # Compute all derivatives
        log.info("pipeline_derivatives", dem=str(dem_path))
        derivative_paths = compute_all_derivatives_native(dem_path, filled_path, deriv_dir)

        marker.write_text(f"processed\nderivatives: {len(derivative_paths)}\n")

        return ProcessedTile(
            tile_dir=tile_dir,
            dem_path=dem_path,
            filled_dem_path=filled_path,
            derivative_paths=derivative_paths,
            resolution_m=self.resolution,
        )

    def _load_existing(self, tile_dir: Path, deriv_dir: Path) -> ProcessedTile:
        """Load an already-processed tile from disk."""
        dem_files = list(tile_dir.glob("*_dem.tif")) + list(tile_dir.glob("dem_*.tif"))
        dem_path = dem_files[0] if dem_files else tile_dir / "dem.tif"

        filled_files = list(tile_dir.glob("*_filled.tif"))
        filled_path = filled_files[0] if filled_files else None

        derivative_paths = {}
        if deriv_dir.exists():
            for f in deriv_dir.glob("*.tif"):
                derivative_paths[f.stem] = f

        log.info("pipeline_loaded_cached", tile_dir=str(tile_dir), derivatives=len(derivative_paths))
        return ProcessedTile(
            tile_dir=tile_dir,
            dem_path=dem_path,
            filled_dem_path=filled_path,
            derivative_paths=derivative_paths,
            resolution_m=self.resolution,
        )

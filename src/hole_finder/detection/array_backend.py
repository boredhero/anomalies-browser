"""Array backend abstraction — GPU (CuPy) or CPU (numpy/scipy).

Provides a unified API for array operations used by detection passes.
When CuPy is available and a GPU is detected, operations run on GPU.
Otherwise falls back transparently to numpy/scipy on CPU.

Usage in passes:
    from hole_finder.detection.array_backend import get_backend
    xp, xndimage = get_backend()
    # xp is cupy or numpy
    # xndimage is cupyx.scipy.ndimage or scipy.ndimage

Thread safety: The backend is determined once at module load time and is
immutable — safe for concurrent access from ThreadPoolExecutor.

Process safety: Each worker process imports this module independently
and detects its own GPU state.
"""

import numpy as np
from scipy import ndimage as scipy_ndimage

from hole_finder.utils.logging import log

# Detect CuPy at import time (once per process)
_HAS_CUPY = False
_CUPY_REASON = ""

try:
    import cupy
    import cupyx.scipy.ndimage as cupy_ndimage

    # Verify we can actually allocate on the GPU
    _test = cupy.zeros(10)
    del _test
    _HAS_CUPY = True
    log.info("cupy_available", device=str(cupy.cuda.Device(0)))
except ImportError:
    _CUPY_REASON = "cupy not installed"
except Exception as e:
    _CUPY_REASON = f"cupy init failed: {e}"

if not _HAS_CUPY and _CUPY_REASON:
    log.info("cupy_unavailable", reason=_CUPY_REASON)


def has_gpu() -> bool:
    """Check if GPU acceleration is available."""
    return _HAS_CUPY


def get_backend() -> tuple:
    """Return (array_module, ndimage_module) for the best available backend.

    Returns:
        (cupy, cupyx.scipy.ndimage) if GPU available
        (numpy, scipy.ndimage) if not
    """
    if _HAS_CUPY:
        return cupy, cupy_ndimage
    return np, scipy_ndimage


def to_device(arr: np.ndarray) -> "np.ndarray | cupy.ndarray":
    """Move a numpy array to the GPU if CuPy is available."""
    if _HAS_CUPY:
        return cupy.asarray(arr)
    return arr


def to_numpy(arr) -> np.ndarray:
    """Ensure array is a numpy array (move from GPU if needed)."""
    if _HAS_CUPY and isinstance(arr, cupy.ndarray):
        return cupy.asnumpy(arr)
    return np.asarray(arr)


def label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Connected component labeling — GPU or CPU.

    Always returns numpy arrays (transfers back from GPU) because
    downstream code (Shapely Point creation) needs numpy.
    """
    if _HAS_CUPY:
        gpu_mask = cupy.asarray(mask)
        gpu_labeled, num = cupy_ndimage.label(gpu_mask)
        return cupy.asnumpy(gpu_labeled), int(num)
    return scipy_ndimage.label(mask)


def region_stats(
    data: np.ndarray,
    labeled: np.ndarray,
    num_features: int,
    mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute bulk region statistics — GPU or CPU.

    Returns dict with numpy arrays (transferred from GPU if needed):
        areas_px, centroids, max_vals, min_vals, sum_vals, mean_vals
    """
    labels = np.arange(1, num_features + 1)

    if _HAS_CUPY:
        gpu_data = cupy.asarray(data)
        gpu_labeled = cupy.asarray(labeled)
        gpu_labels = cupy.asarray(labels)
        gpu_mask = cupy.asarray(mask) if mask is not None else (gpu_labeled > 0).astype(cupy.float32)

        areas = cupy.asnumpy(cupy_ndimage.sum(gpu_mask, gpu_labeled, gpu_labels))
        max_vals = cupy.asnumpy(cupy_ndimage.maximum(gpu_data, gpu_labeled, gpu_labels))
        min_vals = cupy.asnumpy(cupy_ndimage.minimum(gpu_data, gpu_labeled, gpu_labels))
        sum_vals = cupy.asnumpy(cupy_ndimage.sum(gpu_data, gpu_labeled, gpu_labels))
        mean_vals = cupy.asnumpy(cupy_ndimage.mean(gpu_data, gpu_labeled, gpu_labels))
        centroids = [
            tuple(float(x) for x in c) for c in
            cupy_ndimage.center_of_mass(gpu_mask, gpu_labeled, gpu_labels)
        ]
    else:
        use_mask = mask if mask is not None else (labeled > 0).astype(np.float32)
        areas = scipy_ndimage.sum(use_mask, labeled, labels).astype(np.float64)
        max_vals = np.asarray(scipy_ndimage.maximum(data, labeled, labels))
        min_vals = np.asarray(scipy_ndimage.minimum(data, labeled, labels))
        sum_vals = np.asarray(scipy_ndimage.sum(data, labeled, labels))
        mean_vals = np.asarray(scipy_ndimage.mean(data, labeled, labels))
        centroids = scipy_ndimage.center_of_mass(use_mask, labeled, labels)

    return {
        "areas_px": np.asarray(areas, dtype=np.float64),
        "max_vals": np.asarray(max_vals, dtype=np.float64),
        "min_vals": np.asarray(min_vals, dtype=np.float64),
        "sum_vals": np.asarray(sum_vals, dtype=np.float64),
        "mean_vals": np.asarray(mean_vals, dtype=np.float64),
        "centroids": centroids,
    }

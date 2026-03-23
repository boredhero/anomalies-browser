"""Detection passes — import all to trigger registration."""

# Classical passes
from hole_finder.detection.passes.curvature import CurvaturePass
from hole_finder.detection.passes.fill_difference import FillDifferencePass
from hole_finder.detection.passes.local_relief_model import LocalReliefModelPass
from hole_finder.detection.passes.morphometric_filter import MorphometricFilterPass
from hole_finder.detection.passes.multi_return import MultiReturnPass
from hole_finder.detection.passes.point_density import PointDensityPass

# ML passes
from hole_finder.detection.passes.random_forest import RandomForestPass
from hole_finder.detection.passes.sky_view_factor import SkyViewFactorPass
from hole_finder.detection.passes.tpi import TPIPass
from hole_finder.detection.passes.unet_segmentation import UNetSegmentationPass
from hole_finder.detection.passes.yolo_detector import YOLODetectorPass

__all__ = [
    "FillDifferencePass",
    "LocalReliefModelPass",
    "CurvaturePass",
    "SkyViewFactorPass",
    "TPIPass",
    "PointDensityPass",
    "MultiReturnPass",
    "MorphometricFilterPass",
    "RandomForestPass",
    "UNetSegmentationPass",
    "YOLODetectorPass",
]

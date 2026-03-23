"""Detection passes — import all to trigger registration."""

# Classical passes
from magic_eyes.detection.passes.curvature import CurvaturePass
from magic_eyes.detection.passes.fill_difference import FillDifferencePass
from magic_eyes.detection.passes.local_relief_model import LocalReliefModelPass
from magic_eyes.detection.passes.morphometric_filter import MorphometricFilterPass
from magic_eyes.detection.passes.multi_return import MultiReturnPass
from magic_eyes.detection.passes.point_density import PointDensityPass

# ML passes
from magic_eyes.detection.passes.random_forest import RandomForestPass
from magic_eyes.detection.passes.sky_view_factor import SkyViewFactorPass
from magic_eyes.detection.passes.tpi import TPIPass
from magic_eyes.detection.passes.unet_segmentation import UNetSegmentationPass
from magic_eyes.detection.passes.yolo_detector import YOLODetectorPass

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

"""Flash3D – Production-quality 3D Vision Library.

Supports Gaussian Splatting, NeRF, Depth Estimation, and 3D Reconstruction.
"""

__version__ = "1.0.0"
__author__ = "Gaurav14cs17"

from flash3d.models.flash3d_model import Flash3D
from flash3d.engine.trainer import Trainer
from flash3d.engine.predictor import Predictor
from flash3d.engine.exporter import Exporter
from flash3d.solutions.scene_reconstructor import SceneReconstructor
from flash3d.solutions.depth_estimator import DepthEstimator
from flash3d.solutions.view_synthesizer import ViewSynthesizer
from flash3d.analytics.benchmark import Benchmark

__all__ = [
    "Flash3D",
    "Trainer",
    "Predictor",
    "Exporter",
    "SceneReconstructor",
    "DepthEstimator",
    "ViewSynthesizer",
    "Benchmark",
    "__version__",
]

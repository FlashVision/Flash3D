"""Flash3D tasks – Defines high-level 3D vision tasks."""

from flash3d.tasks.depth_prediction import DepthPredictionTask
from flash3d.tasks.novel_view_synthesis import NovelViewSynthesisTask
from flash3d.tasks.point_cloud_segmentation import PointCloudSegmentationTask
from flash3d.tasks.scene_reconstruction import SceneReconstructionTask

__all__ = [
    "NovelViewSynthesisTask",
    "SceneReconstructionTask",
    "DepthPredictionTask",
    "PointCloudSegmentationTask",
]

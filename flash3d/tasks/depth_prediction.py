"""Depth Prediction task definition."""

from __future__ import annotations

import torch

from flash3d.cfg.config import Flash3DConfig
from flash3d.registry import TASKS


@TASKS.register("depth_prediction")
class DepthPredictionTask:
    """Monocular depth prediction from single images.

    Uses a DPT-style encoder-decoder or foundation model backbone.
    Evaluation metrics: AbsRel, SqRel, RMSE, delta thresholds.
    """

    def __init__(self, config: Flash3DConfig | None = None) -> None:
        self.config = config or Flash3DConfig()
        self.model = None

    def setup(self) -> None:
        """Initialize the depth estimation model."""
        from flash3d.geometry.depth import MonocularDepthEstimator

        self.model = MonocularDepthEstimator()

    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """Predict depth for a single image.

        Args:
            image: (1, 3, H, W) or (3, H, W) input image.

        Returns:
            (1, 1, H, W) predicted depth map.
        """
        if self.model is None:
            self.setup()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        return self.model(image)

    def evaluate(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Compute depth estimation metrics."""
        from flash3d.geometry.depth import compute_depth_metrics

        return compute_depth_metrics(pred, target, mask)

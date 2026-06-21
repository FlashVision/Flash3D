"""Point Cloud Segmentation task definition."""

from __future__ import annotations

import torch
import torch.nn as nn

from flash3d.cfg.config import Flash3DConfig
from flash3d.registry import TASKS


@TASKS.register("point_cloud_segmentation")
class PointCloudSegmentationTask:
    """3D point cloud semantic segmentation.

    Assigns class labels to individual points in a 3D point cloud.
    """

    def __init__(
        self,
        config: Flash3DConfig | None = None,
        num_classes: int = 40,
    ) -> None:
        self.config = config or Flash3DConfig()
        self.num_classes = num_classes
        self.model: nn.Module | None = None

    def setup(self, feature_dim: int = 64) -> None:
        """Initialize segmentation head."""
        self.model = PointSegHead(
            in_channels=3 + feature_dim,
            num_classes=self.num_classes,
        )

    def predict(self, points: torch.Tensor, features: torch.Tensor | None = None) -> torch.Tensor:
        """Predict per-point class labels.

        Args:
            points: (N, 3) point coordinates.
            features: (N, F) optional point features.

        Returns:
            (N, num_classes) class logits.
        """
        if self.model is None:
            self.setup(feature_dim=features.shape[-1] if features is not None else 0)

        if features is not None:
            x = torch.cat([points, features], dim=-1)
        else:
            x = points

        return self.model(x)


class PointSegHead(nn.Module):
    """Simple MLP-based point segmentation head."""

    def __init__(self, in_channels: int = 67, num_classes: int = 40) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

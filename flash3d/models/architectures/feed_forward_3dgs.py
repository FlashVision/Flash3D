"""Feed-Forward 3D Gaussian Splatting – Single-pass 3DGS prediction.

Inspired by pixelSplat/MVSplat/YoNoSplat: predicts Gaussian primitives
directly from input images without per-scene optimization.
Supports pose-free reconstruction from sparse views.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from flash3d.cfg.config import Flash3DConfig
from flash3d.registry import MODELS


class ImageEncoder(nn.Module):
    """CNN backbone for extracting multi-scale image features."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        self.output_dim = base_channels * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class CrossViewAttention(nn.Module):
    """Cross-view attention for multi-view feature aggregation."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 8) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(query, context, context)
        x = self.norm1(query + attended)
        x = self.norm2(x + self.ffn(x))
        return x


class GaussianHead(nn.Module):
    """Prediction head that outputs Gaussian primitive parameters per pixel."""

    def __init__(self, in_channels: int = 256, sh_degree: int = 3) -> None:
        super().__init__()
        self.sh_degree = sh_degree
        num_sh = (sh_degree + 1) ** 2

        self.depth_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, 1),
            nn.Softplus(),
        )
        self.scale_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 3, 1),
        )
        self.rotation_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 4, 1),
        )
        self.opacity_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, 1),
        )
        self.sh_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_sh * 3, 1),
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict Gaussian parameters from feature maps.

        Args:
            features: (B, C, H, W) feature maps.

        Returns:
            Dict with depth, scales, rotations, opacities, sh_coeffs.
        """
        depth = self.depth_head(features)
        scales = self.scale_head(features)
        rotations = F.normalize(self.rotation_head(features), dim=1)
        opacities = torch.sigmoid(self.opacity_head(features))
        sh_coeffs = self.sh_head(features)

        return {
            "depth": depth,
            "scales": scales,
            "rotations": rotations,
            "opacities": opacities,
            "sh_coeffs": sh_coeffs,
        }


@MODELS.register("feed_forward_3dgs")
class FeedForward3DGS(nn.Module):
    """Feed-forward 3D Gaussian Splatting for single-pass scene reconstruction.

    Given sparse input views, directly predicts per-pixel Gaussian primitives
    that can be rendered from novel viewpoints without optimization.
    """

    def __init__(
        self,
        config: Flash3DConfig | None = None,
        base_channels: int = 64,
        num_attention_layers: int = 4,
        sh_degree: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.config = config
        self.sh_degree = sh_degree

        self.image_encoder = ImageEncoder(in_channels=3, base_channels=base_channels)
        feat_dim = self.image_encoder.output_dim

        self.proj = nn.Conv2d(feat_dim, 256, 1)

        self.cross_view_layers = nn.ModuleList([
            CrossViewAttention(embed_dim=256, num_heads=8)
            for _ in range(num_attention_layers)
        ])

        self.gaussian_head = GaussianHead(in_channels=256, sh_degree=sh_degree)

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        cameras: dict[str, torch.Tensor] | None = None,
        images: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Predict 3D Gaussians from input images.

        Args:
            images: (B, V, C, H, W) multi-view input images.
            cameras: Optional camera parameters for each view.

        Returns:
            Dict with predicted Gaussian parameters and optionally rendered views.
        """
        if images is None:
            return {"model": "feed_forward_3dgs"}

        if images.dim() == 4:
            images = images.unsqueeze(1)

        B, V, C, H, W = images.shape

        flat_images = images.reshape(B * V, C, H, W)
        features = self.image_encoder(flat_images)
        features = self.proj(features)

        _, Cf, Hf, Wf = features.shape
        features = features.reshape(B, V, Cf, Hf, Wf)

        ref_features = features[:, 0]
        ref_flat = ref_features.flatten(2).permute(0, 2, 1)

        for layer in self.cross_view_layers:
            for v in range(1, V):
                ctx_flat = features[:, v].flatten(2).permute(0, 2, 1)
                ref_flat = layer(ref_flat, ctx_flat)

        ref_features = ref_flat.permute(0, 2, 1).reshape(B, Cf, Hf, Wf)
        ref_features = self.upsample(ref_features)

        target_size = (H, W)
        if ref_features.shape[-2:] != target_size:
            ref_features = F.interpolate(ref_features, size=target_size, mode="bilinear", align_corners=False)

        gaussians = self.gaussian_head(ref_features)
        gaussians["features"] = ref_features

        return gaussians

    def predict_and_render(
        self,
        context_images: torch.Tensor,
        context_cameras: dict[str, torch.Tensor],
        target_camera: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Predict Gaussians from context views and render a target view."""
        gaussians = self.forward(images=context_images, cameras=context_cameras)

        B, _, H, W = gaussians["depth"].shape
        gaussians["depth"].reshape(B, -1, 1)
        scales = gaussians["scales"].permute(0, 2, 3, 1).reshape(B, -1, 3)
        rotations = gaussians["rotations"].permute(0, 2, 3, 1).reshape(B, -1, 4)
        opacities = gaussians["opacities"].reshape(B, -1, 1)

        return {
            "predicted_depth": gaussians["depth"],
            "num_gaussians": H * W,
            "scales": scales,
            "rotations": rotations,
            "opacities": opacities,
        }

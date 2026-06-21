"""Monocular and stereo depth estimation utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthEncoder(nn.Module):
    """Encoder for monocular depth estimation using a U-Net style architecture."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        self.enc1 = self._block(in_channels, base_channels)
        self.enc2 = self._block(base_channels, base_channels * 2)
        self.enc3 = self._block(base_channels * 2, base_channels * 4)
        self.enc4 = self._block(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2)

    @staticmethod
    def _block(in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        return [e1, e2, e3, e4]


class DepthDecoder(nn.Module):
    """Decoder for monocular depth with multi-scale predictions."""

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = self._block(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = self._block(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = self._block(base_channels * 2, base_channels)
        self.depth_head = nn.Sequential(
            nn.Conv2d(base_channels, 1, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _block(in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        e1, e2, e3, e4 = features

        d3 = self.up3(e4)
        d3 = F.interpolate(d3, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        depth = self.depth_head(d1)
        return depth


class MonocularDepthEstimator(nn.Module):
    """Monocular depth estimation model.

    Predicts relative depth from a single RGB image using an encoder-decoder
    architecture. Can be used as initialization for 3DGS or NeRF.
    """

    def __init__(
        self,
        min_depth: float = 0.01,
        max_depth: float = 100.0,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth

        self.encoder = DepthEncoder(in_channels=3, base_channels=base_channels)
        self.decoder = DepthDecoder(base_channels=base_channels)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Predict depth from a single image.

        Args:
            image: (B, 3, H, W) input RGB image in [0, 1].

        Returns:
            (B, 1, H, W) predicted metric depth.
        """
        features = self.encoder(image)
        normalized_depth = self.decoder(features)
        depth = self.min_depth + (self.max_depth - self.min_depth) * normalized_depth
        return depth

    def predict_disparity(self, image: torch.Tensor) -> torch.Tensor:
        """Predict inverse depth (disparity) for better numerical stability."""
        depth = self.forward(image)
        return 1.0 / (depth + 1e-6)


def depth_to_point_cloud(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Unproject depth map to 3D point cloud.

    Args:
        depth: (H, W) or (1, H, W) depth map.
        intrinsics: (3, 3) camera intrinsic matrix.
        extrinsics: (4, 4) camera-to-world matrix (optional).
        mask: (H, W) boolean mask for valid pixels.

    Returns:
        points: (N, 3) 3D points in world (or camera) coordinates.
        colors: None (caller provides if needed).
    """
    if depth.dim() == 3:
        depth = depth.squeeze(0)

    H, W = depth.shape
    device = depth.device

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u = torch.arange(W, device=device, dtype=torch.float32)
    v = torch.arange(H, device=device, dtype=torch.float32)
    v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")

    x = (u_grid - cx) * depth / fx
    y = (v_grid - cy) * depth / fy
    z = depth

    points_cam = torch.stack([x, y, z], dim=-1).reshape(-1, 3)

    if mask is not None:
        valid = mask.reshape(-1)
        points_cam = points_cam[valid]

    if extrinsics is not None:
        R = extrinsics[:3, :3]
        t = extrinsics[:3, 3]
        points_world = (R @ points_cam.T).T + t.unsqueeze(0)
        return points_world, None

    return points_cam, None


def compute_depth_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute standard depth estimation metrics.

    Args:
        pred: Predicted depth map.
        target: Ground truth depth map.
        mask: Valid pixel mask.

    Returns:
        Dict with abs_rel, sq_rel, rmse, rmse_log, delta1, delta2, delta3.
    """
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    else:
        pred = pred.flatten()
        target = target.flatten()

    valid = target > 0
    pred = pred[valid]
    target = target[valid]

    if pred.numel() == 0:
        return {
            k: 0.0 for k in ["abs_rel", "sq_rel", "rmse", "rmse_log", "delta1", "delta2", "delta3"]
        }

    thresh = torch.max(pred / target, target / pred)
    delta1 = (thresh < 1.25).float().mean().item()
    delta2 = (thresh < 1.25**2).float().mean().item()
    delta3 = (thresh < 1.25**3).float().mean().item()

    abs_rel = ((pred - target).abs() / target).mean().item()
    sq_rel = (((pred - target) ** 2) / target).mean().item()
    rmse = ((pred - target) ** 2).mean().sqrt().item()
    rmse_log = ((pred.log() - target.log()) ** 2).mean().sqrt().item()

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
    }

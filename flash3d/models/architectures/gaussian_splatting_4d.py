"""4D Gaussian Splatting with temporal deformation fields for dynamic scenes.

Extends 3D Gaussian Splatting with a deformation network that predicts
per-Gaussian position, rotation, and scale offsets as a function of time.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flash3d.cfg.config import Flash3DConfig
from flash3d.models.architectures.gaussian_splatting import GaussianSplatting
from flash3d.registry import MODELS


class TemporalEncoding(nn.Module):
    """Fourier-based temporal encoding for time-dependent deformations."""

    def __init__(self, num_frequencies: int = 6, include_input: bool = True) -> None:
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        freqs = 2.0 ** torch.arange(num_frequencies).float()
        self.register_buffer("freqs", freqs)

    @property
    def output_dim(self) -> int:
        d = self.num_frequencies * 2
        if self.include_input:
            d += 1
        return d

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Encode scalar time values.

        Args:
            t: (...,) or (..., 1) time values in [0, 1].

        Returns:
            (..., output_dim) encoded features.
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.shape[-1] != 1:
            t = t.unsqueeze(-1)

        encoded = []
        if self.include_input:
            encoded.append(t)
        for freq in self.freqs:
            encoded.append(torch.sin(freq * math.pi * t))
            encoded.append(torch.cos(freq * math.pi * t))
        return torch.cat(encoded, dim=-1)


class DeformationNetwork(nn.Module):
    """Deformation MLP that predicts Gaussian parameter offsets given
    canonical position and time.

    Outputs per-Gaussian deltas for:
    - Position (dx, dy, dz)
    - Rotation quaternion (dw, dx, dy, dz)
    - Scale (ds_x, ds_y, ds_z)
    """

    def __init__(
        self,
        pos_dim: int = 3,
        time_dim: int = 13,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_time_frequencies: int = 6,
        num_pos_frequencies: int = 10,
    ) -> None:
        super().__init__()
        self.time_encoder = TemporalEncoding(num_time_frequencies)
        self.pos_encoder = TemporalEncoding(num_pos_frequencies)

        input_dim = self.pos_encoder.output_dim * 3 + self.time_encoder.output_dim

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(num_layers - 1):
            if i == num_layers // 2:
                layers.extend([nn.Linear(hidden_dim + input_dim, hidden_dim), nn.ReLU(inplace=True)])
            else:
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)])

        self.backbone = nn.ModuleList([l for l in layers if isinstance(l, nn.Linear)])
        self.activations = nn.ModuleList([l for l in layers if isinstance(l, nn.ReLU)])
        self.skip_layer_idx = num_layers // 2

        self.position_head = nn.Linear(hidden_dim, 3)
        self.rotation_head = nn.Linear(hidden_dim, 4)
        self.scale_head = nn.Linear(hidden_dim, 3)

        nn.init.zeros_(self.position_head.weight)
        nn.init.zeros_(self.position_head.bias)
        nn.init.zeros_(self.rotation_head.weight)
        nn.init.zeros_(self.rotation_head.bias)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)

    def forward(
        self, positions: torch.Tensor, time: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Predict deformation offsets.

        Args:
            positions: (N, 3) canonical Gaussian positions.
            time: scalar or (N, 1) time value in [0, 1].

        Returns:
            Dict with 'delta_pos' (N, 3), 'delta_rot' (N, 4), 'delta_scale' (N, 3).
        """
        if time.dim() == 0:
            time = time.unsqueeze(0).expand(positions.shape[0])
        if time.dim() == 1:
            time = time.unsqueeze(-1)

        pos_enc = self.pos_encoder(positions.reshape(-1, 1)).reshape(positions.shape[0], -1)
        time_enc = self.time_encoder(time)

        x = torch.cat([pos_enc, time_enc], dim=-1)
        x_input = x

        h = x
        for i, linear in enumerate(self.backbone):
            if i == self.skip_layer_idx and i > 0:
                h = torch.cat([h, x_input], dim=-1)
            h = F.relu(linear(h))

        return {
            "delta_pos": self.position_head(h),
            "delta_rot": self.rotation_head(h),
            "delta_scale": self.scale_head(h),
        }


@MODELS.register("gaussian_splatting_4d")
class GaussianSplatting4D(nn.Module):
    """4D Gaussian Splatting for dynamic scenes.

    Maintains a canonical 3DGS representation and a deformation network
    that warps Gaussians as a function of time. Supports:
    - Temporal deformation fields for position, rotation, and scale
    - Canonical space regularization
    - Per-timestep rendering
    - Temporal smoothness constraints
    """

    def __init__(
        self,
        config: Optional[Flash3DConfig] = None,
        num_gaussians: int = 100_000,
        sh_degree: int = 3,
        deformation_hidden_dim: int = 256,
        deformation_num_layers: int = 6,
        num_time_frequencies: int = 6,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.config = config
        self.canonical = GaussianSplatting(
            config=config, num_gaussians=num_gaussians,
            sh_degree=sh_degree, **kwargs,
        )

        self.deformation = DeformationNetwork(
            hidden_dim=deformation_hidden_dim,
            num_layers=deformation_num_layers,
            num_time_frequencies=num_time_frequencies,
        )

        self.time_range = (0.0, 1.0)

    @property
    def num_points(self) -> int:
        return self.canonical.num_points

    def deform_gaussians(
        self, time: float | torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Apply temporal deformation to canonical Gaussians.

        Args:
            time: Scalar time value in [0, 1].

        Returns:
            Dict with deformed 'means', 'rotations', 'scales', plus
            canonical 'opacities' and 'sh_coeffs'.
        """
        if isinstance(time, (int, float)):
            time = torch.tensor(time, device=self.canonical.means.device)

        deform = self.deformation(self.canonical.means, time)

        deformed_means = self.canonical.means + deform["delta_pos"]

        canonical_rot = F.normalize(self.canonical.rotations, dim=-1)
        delta_rot = F.normalize(deform["delta_rot"] + torch.tensor(
            [1, 0, 0, 0], device=canonical_rot.device, dtype=canonical_rot.dtype,
        ), dim=-1)
        deformed_rotations = self._quaternion_multiply(canonical_rot, delta_rot)

        deformed_scales = self.canonical.scales + deform["delta_scale"]

        return {
            "means": deformed_means,
            "rotations": deformed_rotations,
            "scales": deformed_scales,
            "opacities": self.canonical.get_opacity(),
            "sh_coeffs": self.canonical.sh_coeffs,
        }

    @staticmethod
    def _quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply two quaternions (w, x, y, z)."""
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dim=-1)

    def forward(
        self,
        cameras: Optional[Dict[str, torch.Tensor]] = None,
        images: Optional[torch.Tensor] = None,
        time: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Render at a given timestep.

        Args:
            cameras: Camera parameters.
            images: Ground truth images (for training).
            time: Time value in [0, 1].

        Returns:
            Rendered output dict with 'rgb', 'depth', 'alpha'.
        """
        if cameras is None:
            return {
                "means": self.canonical.means,
                "num_gaussians": self.num_points,
                "model": "4d_gaussian_splatting",
            }

        return self.render(cameras, time=time, **kwargs)

    def render(
        self,
        camera: Dict[str, torch.Tensor],
        time: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Render the deformed Gaussians at a given time."""
        from flash3d.rendering.rasterizer import rasterize_gaussians

        deformed = self.deform_gaussians(time)

        width = camera.get("image_width", 800)
        height = camera.get("image_height", 800)
        if isinstance(width, torch.Tensor):
            width, height = width.item(), height.item()

        camera_center = camera.get(
            "camera_center", torch.zeros(3, device=self.canonical.means.device),
        )

        result = rasterize_gaussians(
            means3d=deformed["means"],
            scales=torch.exp(deformed["scales"]),
            rotations=F.normalize(deformed["rotations"], dim=-1),
            opacities=deformed["opacities"],
            sh_coeffs=deformed["sh_coeffs"],
            viewmatrix=camera["viewmatrix"],
            projmatrix=camera["projmatrix"],
            camera_center=camera_center,
            image_width=int(width),
            image_height=int(height),
            sh_degree=self.canonical.sh_degree,
        )

        return result

    def temporal_smoothness_loss(
        self, time: float, dt: float = 0.01,
    ) -> torch.Tensor:
        """Compute temporal smoothness regularization loss.

        Penalizes large differences in deformation between nearby timesteps.
        """
        device = self.canonical.means.device
        t1 = torch.tensor(time, device=device)
        t2 = torch.tensor(min(time + dt, 1.0), device=device)

        d1 = self.deformation(self.canonical.means, t1)
        d2 = self.deformation(self.canonical.means, t2)

        pos_diff = (d1["delta_pos"] - d2["delta_pos"]).pow(2).mean()
        rot_diff = (d1["delta_rot"] - d2["delta_rot"]).pow(2).mean()
        scale_diff = (d1["delta_scale"] - d2["delta_scale"]).pow(2).mean()

        return pos_diff + rot_diff + scale_diff

    def initialize_from_point_cloud(
        self,
        points: torch.Tensor,
        colors: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize canonical Gaussians from a point cloud."""
        self.canonical.initialize_from_point_cloud(points, colors)

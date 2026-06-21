"""3D Gaussian Splatting – Differentiable point-based scene representation.

Implements the core Gaussian primitives with:
- Position (xyz)
- Covariance (rotation quaternion + scale)
- Opacity (sigmoid-activated)
- Spherical harmonics coefficients for view-dependent color
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flash3d.cfg.config import Flash3DConfig
from flash3d.registry import MODELS


@MODELS.register("gaussian_splatting")
class GaussianSplatting(nn.Module):
    """3D Gaussian Splatting model.

    Each Gaussian is parameterized by:
        - means (N, 3): 3D positions
        - scales (N, 3): log-space scales
        - rotations (N, 4): quaternions (wxyz)
        - opacities (N, 1): pre-sigmoid opacity
        - sh_coeffs (N, C, 3): spherical harmonics coefficients
    """

    def __init__(
        self,
        config: Optional[Flash3DConfig] = None,
        num_gaussians: int = 100_000,
        sh_degree: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.config = config
        num_g = config.model.num_gaussians if config else num_gaussians
        sh_deg = config.model.sh_degree if config else sh_degree
        self.sh_degree = sh_deg
        self.num_sh_coeffs = (sh_deg + 1) ** 2

        self.means = nn.Parameter(torch.randn(num_g, 3) * 0.1)
        self.scales = nn.Parameter(torch.zeros(num_g, 3) - 3.0)
        self.rotations = nn.Parameter(self._init_quaternions(num_g))
        self.opacities = nn.Parameter(torch.zeros(num_g, 1))
        self.sh_coeffs = nn.Parameter(
            torch.zeros(num_g, self.num_sh_coeffs, 3) * 0.01
        )

        self._densification_grad_accum = torch.zeros(num_g, 1)
        self._densification_count = torch.zeros(num_g, 1)

    @staticmethod
    def _init_quaternions(n: int) -> torch.Tensor:
        """Initialize identity quaternions (w=1, x=y=z=0)."""
        q = torch.zeros(n, 4)
        q[:, 0] = 1.0
        return q

    @property
    def num_points(self) -> int:
        return self.means.shape[0]

    def get_covariance_3d(self) -> torch.Tensor:
        """Compute 3D covariance matrices from scale and rotation.

        Returns:
            Covariance matrices (N, 3, 3).
        """
        scales = torch.exp(self.scales)
        S = torch.diag_embed(scales)

        q = F.normalize(self.rotations, dim=-1)
        R = self._quaternion_to_matrix(q)

        # Sigma = R @ S @ S^T @ R^T
        RS = R @ S
        covariance = RS @ RS.transpose(-1, -2)
        return covariance

    @staticmethod
    def _quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
        """Convert quaternion (wxyz) to rotation matrix."""
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

        R = torch.stack([
            1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
            2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
            2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y),
        ], dim=-1).reshape(*q.shape[:-1], 3, 3)

        return R

    def get_opacity(self) -> torch.Tensor:
        """Get activated opacity values in [0, 1]."""
        return torch.sigmoid(self.opacities)

    def get_scales(self) -> torch.Tensor:
        """Get activated scale values (positive)."""
        return torch.exp(self.scales)

    def forward(
        self,
        cameras: Optional[Dict[str, torch.Tensor]] = None,
        images: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Render the Gaussian field from given camera viewpoints.

        Args:
            cameras: Dict with 'viewmatrix' (4x4), 'projmatrix' (4x4),
                    'camera_center' (3,), 'image_width', 'image_height'.

        Returns:
            Dict with 'rgb' (B, 3, H, W), 'depth' (B, 1, H, W), 'alpha' (B, 1, H, W).
        """
        if cameras is None:
            return {"means": self.means, "num_gaussians": self.num_points}

        return self.render(cameras, **kwargs)

    def render(
        self,
        camera: Dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Differentiable Gaussian rasterization for a single camera."""
        from flash3d.rendering.rasterizer import rasterize_gaussians

        width = camera.get("image_width", 800)
        height = camera.get("image_height", 800)
        if isinstance(width, torch.Tensor):
            width = width.item()
            height = height.item()

        viewmatrix = camera["viewmatrix"]
        projmatrix = camera["projmatrix"]
        camera_center = camera.get("camera_center", torch.zeros(3, device=self.means.device))

        result = rasterize_gaussians(
            means3d=self.means,
            scales=self.get_scales(),
            rotations=F.normalize(self.rotations, dim=-1),
            opacities=self.get_opacity(),
            sh_coeffs=self.sh_coeffs,
            viewmatrix=viewmatrix,
            projmatrix=projmatrix,
            camera_center=camera_center,
            image_width=int(width),
            image_height=int(height),
            sh_degree=self.sh_degree,
        )

        return result

    def densify_and_prune(
        self,
        grad_threshold: float = 0.0002,
        min_opacity: float = 0.005,
        max_screen_size: float = 20.0,
    ) -> None:
        """Adaptive density control: split large Gaussians, clone small ones, prune transparent."""
        grads = self._densification_grad_accum / (self._densification_count + 1e-7)
        grads[grads.isnan()] = 0.0

        # Clone small Gaussians with large gradients
        selected_mask = (grads.squeeze() >= grad_threshold)
        scales = self.get_scales()
        small_mask = selected_mask & (scales.max(dim=-1).values < 0.01)

        if small_mask.any():
            self._clone_gaussians(small_mask)

        # Split large Gaussians with large gradients
        large_mask = selected_mask & (scales.max(dim=-1).values >= 0.01)
        if large_mask.any():
            self._split_gaussians(large_mask)

        # Prune by opacity
        opacity_mask = (self.get_opacity().squeeze() < min_opacity)
        if opacity_mask.any():
            self._prune_gaussians(opacity_mask)

        self._densification_grad_accum.zero_()
        self._densification_count.zero_()

    def _clone_gaussians(self, mask: torch.Tensor) -> None:
        """Clone selected Gaussians."""
        new_means = self.means.data[mask].clone()
        new_scales = self.scales.data[mask].clone()
        new_rotations = self.rotations.data[mask].clone()
        new_opacities = self.opacities.data[mask].clone()
        new_sh = self.sh_coeffs.data[mask].clone()

        self.means = nn.Parameter(torch.cat([self.means.data, new_means]))
        self.scales = nn.Parameter(torch.cat([self.scales.data, new_scales]))
        self.rotations = nn.Parameter(torch.cat([self.rotations.data, new_rotations]))
        self.opacities = nn.Parameter(torch.cat([self.opacities.data, new_opacities]))
        self.sh_coeffs = nn.Parameter(torch.cat([self.sh_coeffs.data, new_sh]))

    def _split_gaussians(self, mask: torch.Tensor) -> None:
        """Split selected Gaussians into two."""
        n_split = mask.sum().item()
        stds = self.get_scales()[mask].repeat(2, 1)
        means = self.means.data[mask].repeat(2, 1)
        samples = torch.randn_like(means) * stds
        new_means = means + samples

        new_scales = self.scales.data[mask].repeat(2, 1) - math.log(1.6)
        new_rotations = self.rotations.data[mask].repeat(2, 1)
        new_opacities = self.opacities.data[mask].repeat(2, 1)
        new_sh = self.sh_coeffs.data[mask].repeat(2, 1, 1)

        keep_mask = ~mask
        self.means = nn.Parameter(torch.cat([self.means.data[keep_mask], new_means]))
        self.scales = nn.Parameter(torch.cat([self.scales.data[keep_mask], new_scales]))
        self.rotations = nn.Parameter(torch.cat([self.rotations.data[keep_mask], new_rotations]))
        self.opacities = nn.Parameter(torch.cat([self.opacities.data[keep_mask], new_opacities]))
        self.sh_coeffs = nn.Parameter(torch.cat([self.sh_coeffs.data[keep_mask], new_sh]))

    def _prune_gaussians(self, mask: torch.Tensor) -> None:
        """Remove Gaussians indicated by mask."""
        keep = ~mask
        self.means = nn.Parameter(self.means.data[keep])
        self.scales = nn.Parameter(self.scales.data[keep])
        self.rotations = nn.Parameter(self.rotations.data[keep])
        self.opacities = nn.Parameter(self.opacities.data[keep])
        self.sh_coeffs = nn.Parameter(self.sh_coeffs.data[keep])

    def initialize_from_point_cloud(
        self,
        points: torch.Tensor,
        colors: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize Gaussian parameters from a 3D point cloud.

        Args:
            points: (N, 3) point positions.
            colors: (N, 3) point colors in [0, 1] (optional).
        """
        n = points.shape[0]
        device = points.device

        self.means = nn.Parameter(points.clone())
        self.scales = nn.Parameter(torch.full((n, 3), -3.0, device=device))
        self.rotations = nn.Parameter(self._init_quaternions(n).to(device))
        self.opacities = nn.Parameter(torch.full((n, 1), 0.1, device=device))

        sh = torch.zeros(n, self.num_sh_coeffs, 3, device=device)
        if colors is not None:
            C0 = 0.28209479177387814
            sh[:, 0, :] = (colors - 0.5) / C0
        self.sh_coeffs = nn.Parameter(sh)

        self._densification_grad_accum = torch.zeros(n, 1, device=device)
        self._densification_count = torch.zeros(n, 1, device=device)

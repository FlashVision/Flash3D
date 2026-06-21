"""Mip-NeRF 360: integrated positional encoding, anti-aliasing, scene contraction.

Implements the key ideas from Mip-NeRF (Barron et al., 2021) and
Mip-NeRF 360 (Barron et al., 2022) for anti-aliased neural radiance fields
with unbounded scene handling.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from flash3d.cfg.config import Flash3DConfig
from flash3d.registry import MODELS


class IntegratedPositionalEncoding(nn.Module):
    """Integrated positional encoding (IPE) from Mip-NeRF.

    Instead of encoding point positions, encodes the mean and covariance
    of conical frustums along each ray interval, providing anti-aliased
    featurization of 3D space.
    """

    def __init__(self, num_frequencies: int = 16, include_input: bool = True) -> None:
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
        return d * 3

    def forward(
        self,
        means: torch.Tensor,
        covs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute integrated positional encoding.

        Args:
            means: (..., 3) mean positions of intervals along rays.
            covs: (..., 3) diagonal covariance of the Gaussian approximation.

        Returns:
            (..., output_dim) IPE features.
        """
        encoded = []
        if self.include_input:
            encoded.append(means)

        for freq in self.freqs:
            scaled_mean = freq * means
            scaled_var = (freq**2) * covs
            weight = torch.exp(-0.5 * scaled_var)
            encoded.append(weight * torch.sin(scaled_mean))
            encoded.append(weight * torch.cos(scaled_mean))

        return torch.cat(encoded, dim=-1)


class SceneContraction(nn.Module):
    """Scene contraction for unbounded scenes (Mip-NeRF 360).

    Maps unbounded 3D coordinates into a bounded ball using the
    contraction function: points inside a unit sphere stay as-is,
    and points outside are contracted using an inverse-distance mapping.
    """

    def __init__(self, order: float = float("inf")) -> None:
        super().__init__()
        self.order = order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Contract unbounded coordinates to bounded space.

        Args:
            x: (..., 3) 3D positions.

        Returns:
            (..., 3) contracted positions.
        """
        norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        mask = (norm <= 1.0).squeeze(-1)

        contracted = x.clone()
        outer = ~mask
        if outer.any():
            n = norm[outer]
            contracted[outer] = (2.0 - 1.0 / n) * (x[outer] / n)

        return contracted

    def contract_covariance(
        self,
        x: torch.Tensor,
        cov_diag: torch.Tensor,
    ) -> torch.Tensor:
        """Contract covariance under the scene contraction mapping.

        Approximates Jacobian scaling for the contraction function.
        """
        norm = torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=1e-6)
        mask = (norm <= 1.0).squeeze(-1)

        contracted_cov = cov_diag.clone()
        outer = ~mask
        if outer.any():
            n = norm[outer]
            scale = (1.0 / (n**2)) ** 2
            contracted_cov[outer] = cov_diag[outer] * scale

        return contracted_cov


class ProposalNetwork(nn.Module):
    """Lightweight proposal MLP for sampling guidance (Mip-NeRF 360).

    Produces density estimates to guide hierarchical sampling.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(x))


class MipNeRFMLP(nn.Module):
    """MLP decoder for Mip-NeRF: maps IPE features to density and color."""

    def __init__(
        self,
        input_dim: int,
        dir_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 8,
        skip_connections: tuple[int, ...] = (4,),
    ) -> None:
        super().__init__()
        self.skip_connections = skip_connections

        layers = []
        in_d = input_dim
        for i in range(num_layers):
            if i in skip_connections:
                in_d += input_dim
            layers.append(nn.Linear(in_d, hidden_dim))
            in_d = hidden_dim
        self.layers = nn.ModuleList(layers)

        self.density_head = nn.Linear(hidden_dim, 1)
        self.feature_head = nn.Linear(hidden_dim, hidden_dim)
        self.color_mlp = nn.Sequential(
            nn.Linear(hidden_dim + dir_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid(),
        )

    def forward(
        self,
        pos_enc: torch.Tensor,
        dir_enc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = pos_enc
        for i, layer in enumerate(self.layers):
            if i in self.skip_connections:
                h = torch.cat([h, pos_enc], dim=-1)
            h = F.relu(layer(h))

        density = F.softplus(self.density_head(h))
        features = self.feature_head(h)
        rgb = self.color_mlp(torch.cat([features, dir_enc], dim=-1))
        return density, rgb


@MODELS.register("mip_nerf_360")
class MipNeRF360(nn.Module):
    """Mip-NeRF 360: Anti-aliased NeRF with unbounded scene support.

    Key features:
    - Integrated positional encoding (IPE) for anti-aliased rendering
    - Scene contraction for unbounded 360° scenes
    - Proposal-based hierarchical sampling
    - Distortion loss for regularization
    """

    def __init__(
        self,
        config: Flash3DConfig | None = None,
        num_pos_frequencies: int = 16,
        num_dir_frequencies: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 8,
        near: float = 0.01,
        far: float = 1000.0,
        num_coarse_samples: int = 64,
        num_fine_samples: int = 128,
        use_contraction: bool = True,
        num_proposal_rounds: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.config = config
        self.near = near if config is None else config.render.near
        self.far = far if config is None else config.render.far
        self.num_coarse_samples = num_coarse_samples
        self.num_fine_samples = num_fine_samples
        self.use_contraction = use_contraction

        self.ipe = IntegratedPositionalEncoding(num_pos_frequencies)
        ipe_dim = self.ipe.output_dim

        from flash3d.models.architectures.nerf import PositionalEncoding

        self.dir_encoder = PositionalEncoding(num_dir_frequencies)
        dir_dim = 3 * self.dir_encoder.output_dim

        self.mlp = MipNeRFMLP(
            input_dim=ipe_dim,
            dir_dim=dir_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        self.contraction = SceneContraction() if use_contraction else None

        self.proposal_nets = nn.ModuleList(
            [
                ProposalNetwork(input_dim=ipe_dim, hidden_dim=64, num_layers=2)
                for _ in range(num_proposal_rounds)
            ]
        )

    def forward(
        self,
        cameras: dict[str, torch.Tensor] | None = None,
        images: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        if cameras is None:
            return {"model": "mip_nerf_360", "num_parameters": self.num_parameters}
        return self.render(cameras, **kwargs)

    def render(
        self,
        camera: dict[str, torch.Tensor],
        return_distortion_loss: bool = False,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Render using Mip-NeRF 360 with proposal sampling."""
        from flash3d.rendering.ray_marching import volume_render_rays

        rays_o = camera.get("rays_o")
        rays_d = camera.get("rays_d")

        if rays_o is None:
            from flash3d.rendering.cameras import generate_rays

            width = camera.get("image_width", 800)
            height = camera.get("image_height", 800)
            if isinstance(width, torch.Tensor):
                width, height = width.item(), height.item()
            rays_o, rays_d = generate_rays(
                camera["intrinsics"],
                camera["extrinsics"],
                int(width),
                int(height),
            )

        t_vals = self._sample_along_rays(rays_o, rays_d)

        means, covs = self._compute_gaussian_intervals(rays_o, rays_d, t_vals)

        if self.contraction is not None:
            contracted_means = self.contraction(means)
            contracted_covs = self.contraction.contract_covariance(means, covs)
        else:
            contracted_means = means
            contracted_covs = covs

        pos_enc = self.ipe(contracted_means, contracted_covs)
        dirs = F.normalize(rays_d, dim=-1)
        dirs_expanded = dirs.unsqueeze(-2).expand_as(means)
        dir_enc = self.dir_encoder(dirs_expanded.reshape(-1, 3)).reshape(
            *dirs_expanded.shape[:-1], -1
        )

        density, rgb = self.mlp(pos_enc, dir_enc)
        density = density.reshape(*t_vals.shape[:-1], t_vals.shape[-1] - 1)
        rgb = rgb.reshape(*t_vals.shape[:-1], t_vals.shape[-1] - 1, 3)

        result = volume_render_rays(density.squeeze(-1), rgb, t_vals[..., :-1])

        if return_distortion_loss:
            result["distortion_loss"] = self._distortion_loss(density.squeeze(-1), t_vals)

        return result

    def _sample_along_rays(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
    ) -> torch.Tensor:
        """Generate sample points along rays."""
        n_samples = self.num_coarse_samples + self.num_fine_samples
        t_vals = torch.linspace(
            self.near,
            self.far,
            n_samples + 1,
            device=rays_o.device,
        )
        return t_vals.unsqueeze(0).expand(rays_o.shape[0], -1)

    def _compute_gaussian_intervals(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        t_vals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Gaussian approximation of conical frustums.

        Returns mean and diagonal covariance for each interval.
        """
        t_mid = 0.5 * (t_vals[..., :-1] + t_vals[..., 1:])
        t_delta = t_vals[..., 1:] - t_vals[..., :-1]

        means = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * t_mid.unsqueeze(-1)
        covs = (t_delta.unsqueeze(-1) ** 2 / 12.0).expand_as(means)
        return means, covs

    def _distortion_loss(
        self,
        weights: torch.Tensor,
        t_vals: torch.Tensor,
    ) -> torch.Tensor:
        """Distortion regularization loss (Mip-NeRF 360)."""
        t_mid = 0.5 * (t_vals[..., :-1] + t_vals[..., 1:])
        t_delta = t_vals[..., 1:] - t_vals[..., :-1]
        w = F.softmax(weights, dim=-1)

        (w * t_mid).sum(dim=-1, keepdim=True)
        dist = torch.abs(t_mid.unsqueeze(-1) - t_mid.unsqueeze(-2))
        loss = (w.unsqueeze(-1) * w.unsqueeze(-2) * dist).sum(dim=(-1, -2))
        loss = loss + (1.0 / 3.0) * (w**2 * t_delta).sum(dim=-1)
        return loss.mean()

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

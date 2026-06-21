"""Neural Radiance Fields (NeRF) – MLP-based volumetric scene representation.

Implements instant-NGP-style hash encoding with MLP decoder for
efficient training and rendering.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flash3d.cfg.config import Flash3DConfig
from flash3d.registry import MODELS


class PositionalEncoding(nn.Module):
    """Fourier feature positional encoding for NeRF inputs."""

    def __init__(self, num_frequencies: int = 10, include_input: bool = True) -> None:
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        freqs = 2.0 ** torch.linspace(0.0, num_frequencies - 1, num_frequencies)
        self.register_buffer("freqs", freqs)

    @property
    def output_dim(self) -> int:
        d = self.num_frequencies * 2
        if self.include_input:
            d += 1
        return d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input with sinusoidal positional features.

        Args:
            x: (..., D) input coordinates.
        Returns:
            (..., D * output_dim) encoded features.
        """
        encoded = []
        if self.include_input:
            encoded.append(x)

        for freq in self.freqs:
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))

        return torch.cat(encoded, dim=-1)


class HashEncoding(nn.Module):
    """Multi-resolution hash encoding (instant-NGP style).

    Provides O(1) lookup for spatial features across multiple resolution levels.
    """

    def __init__(
        self,
        num_levels: int = 16,
        features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        max_resolution: int = 2048,
    ) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.hashmap_size = 2 ** log2_hashmap_size

        growth_factor = math.exp((math.log(max_resolution) - math.log(base_resolution)) / (num_levels - 1))
        self.resolutions = [int(base_resolution * growth_factor**i) for i in range(num_levels)]

        self.hash_tables = nn.ParameterList([
            nn.Parameter(torch.randn(self.hashmap_size, features_per_level) * 1e-4)
            for _ in range(num_levels)
        ])

    @property
    def output_dim(self) -> int:
        return self.num_levels * self.features_per_level

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Look up hash features for input 3D positions.

        Args:
            x: (..., 3) positions normalized to [0, 1].
        Returns:
            (..., output_dim) hash-encoded features.
        """
        outputs = []
        for level, resolution in enumerate(self.resolutions):
            features = self._hash_lookup(x, self.hash_tables[level], resolution)
            outputs.append(features)
        return torch.cat(outputs, dim=-1)

    def _hash_lookup(
        self,
        x: torch.Tensor,
        table: nn.Parameter,
        resolution: int,
    ) -> torch.Tensor:
        """Trilinear interpolation in hash grid."""
        scaled = x * resolution
        floor_coords = scaled.floor().long()
        frac = scaled - floor_coords.float()

        # Hash the 8 corners of the voxel
        offsets = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
             [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
            device=x.device, dtype=torch.long,
        )

        corners = floor_coords.unsqueeze(-2) + offsets
        indices = self._spatial_hash(corners) % self.hashmap_size

        corner_features = table[indices.view(-1)].view(*indices.shape, -1)

        # Trilinear interpolation weights
        wx = frac[..., 0:1].unsqueeze(-2)
        wy = frac[..., 1:2].unsqueeze(-2)
        wz = frac[..., 2:3].unsqueeze(-2)

        c00 = corner_features[..., 0, :] * (1 - wx.squeeze(-2)) + corner_features[..., 1, :] * wx.squeeze(-2)
        c01 = corner_features[..., 2, :] * (1 - wx.squeeze(-2)) + corner_features[..., 4, :] * wx.squeeze(-2)
        c10 = corner_features[..., 3, :] * (1 - wx.squeeze(-2)) + corner_features[..., 5, :] * wx.squeeze(-2)
        c11 = corner_features[..., 6, :] * (1 - wx.squeeze(-2)) + corner_features[..., 7, :] * wx.squeeze(-2)

        c0 = c00 * (1 - wy.squeeze(-2)) + c01 * wy.squeeze(-2)
        c1 = c10 * (1 - wy.squeeze(-2)) + c11 * wy.squeeze(-2)

        result = c0 * (1 - wz.squeeze(-2)) + c1 * wz.squeeze(-2)
        return result

    @staticmethod
    def _spatial_hash(coords: torch.Tensor) -> torch.Tensor:
        """Spatial hashing function for 3D integer coordinates."""
        primes = torch.tensor([1, 2654435761, 805459861], device=coords.device, dtype=torch.long)
        return (coords * primes).sum(dim=-1)


class NeRFMLP(nn.Module):
    """NeRF MLP decoder: maps encoded positions to density and color."""

    def __init__(
        self,
        input_dim: int = 63,
        dir_dim: int = 27,
        hidden_dim: int = 256,
        num_layers: int = 8,
        skip_connections: Tuple[int, ...] = (4,),
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

        self.color_layers = nn.Sequential(
            nn.Linear(hidden_dim + dir_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(
        self,
        pos_encoded: torch.Tensor,
        dir_encoded: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict density and RGB color.

        Args:
            pos_encoded: Encoded positions (..., input_dim).
            dir_encoded: Encoded view directions (..., dir_dim).

        Returns:
            density: (..., 1) volume density.
            rgb: (..., 3) color values.
        """
        h = pos_encoded
        for i, layer in enumerate(self.layers):
            if i in self.skip_connections:
                h = torch.cat([h, pos_encoded], dim=-1)
            h = F.relu(layer(h))

        density = F.softplus(self.density_head(h))
        features = self.feature_head(h)

        color_input = torch.cat([features, dir_encoded], dim=-1)
        rgb = torch.sigmoid(self.color_layers(color_input))

        return density, rgb


@MODELS.register("nerf")
class NeRF(nn.Module):
    """Neural Radiance Field with optional hash encoding.

    Supports both classical positional encoding and instant-NGP hash encoding.
    """

    def __init__(
        self,
        config: Optional[Flash3DConfig] = None,
        use_hash_encoding: bool = True,
        num_pos_frequencies: int = 10,
        num_dir_frequencies: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 8,
        near: float = 0.01,
        far: float = 100.0,
        num_samples: int = 64,
        num_importance_samples: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.config = config
        self.near = near if config is None else config.render.near
        self.far = far if config is None else config.render.far
        self.num_samples = num_samples
        self.num_importance_samples = num_importance_samples
        self.use_hash_encoding = use_hash_encoding

        if use_hash_encoding:
            self.pos_encoder = HashEncoding()
            pos_dim = self.pos_encoder.output_dim
        else:
            self.pos_encoder = PositionalEncoding(num_pos_frequencies)
            pos_dim = 3 * self.pos_encoder.output_dim

        self.dir_encoder = PositionalEncoding(num_dir_frequencies)
        dir_dim = 3 * self.dir_encoder.output_dim

        self.mlp = NeRFMLP(
            input_dim=pos_dim,
            dir_dim=dir_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    def forward(
        self,
        cameras: Optional[Dict[str, torch.Tensor]] = None,
        images: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Render from camera viewpoints using volume rendering."""
        if cameras is None:
            return {"model": "nerf", "num_parameters": self.num_parameters}

        return self.render(cameras, **kwargs)

    def render(
        self,
        camera: Dict[str, torch.Tensor],
        num_rays: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Render an image using ray marching and volume rendering.

        Args:
            camera: Camera parameters with 'rays_o' and 'rays_d', or
                   intrinsics/extrinsics to generate rays.
        """
        from flash3d.rendering.ray_marching import volume_render_rays

        rays_o = camera.get("rays_o")
        rays_d = camera.get("rays_d")

        if rays_o is None:
            from flash3d.rendering.cameras import generate_rays
            width = camera.get("image_width", 800)
            height = camera.get("image_height", 800)
            if isinstance(width, torch.Tensor):
                width, height = width.item(), height.item()
            intrinsics = camera["intrinsics"]
            extrinsics = camera["extrinsics"]
            rays_o, rays_d = generate_rays(intrinsics, extrinsics, int(width), int(height))

        t_vals = torch.linspace(self.near, self.far, self.num_samples, device=rays_o.device)
        points = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * t_vals.unsqueeze(-1)

        flat_points = points.reshape(-1, 3)
        flat_dirs = rays_d.unsqueeze(-2).expand_as(points).reshape(-1, 3)
        flat_dirs = F.normalize(flat_dirs, dim=-1)

        pos_enc = self.pos_encoder(flat_points)
        dir_enc = self.dir_encoder(flat_dirs)

        density, rgb = self.mlp(pos_enc, dir_enc)

        density = density.reshape(*points.shape[:-1], 1)
        rgb = rgb.reshape(*points.shape[:-1], 3)

        result = volume_render_rays(density.squeeze(-1), rgb, t_vals)
        return result

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def query(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query the NeRF at arbitrary 3D positions.

        Args:
            positions: (N, 3) query points.
            directions: (N, 3) view directions.

        Returns:
            density: (N, 1).
            rgb: (N, 3).
        """
        pos_enc = self.pos_encoder(positions)
        dir_enc = self.dir_encoder(F.normalize(directions, dim=-1))
        return self.mlp(pos_enc, dir_enc)

"""Instant-NGP multi-resolution hash encoding with trilinear interpolation.

Implements the spatial hash encoding from "Instant Neural Graphics Primitives"
(Müller et al., 2022) with configurable resolution levels, hash table sizes,
and feature dimensions.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from flash3d.registry import MODELS


class MultiResolutionHashEncoding(nn.Module):
    """Multi-resolution hash encoding with trilinear interpolation.

    Creates a hierarchy of hash grids at different spatial resolutions.
    Each level maps 3D positions to feature vectors via spatial hashing
    and trilinear interpolation of 8-corner voxel features.

    The resolution at each level grows geometrically from base_resolution
    to max_resolution across num_levels levels.
    """

    def __init__(
        self,
        num_levels: int = 16,
        features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        max_resolution: int = 2048,
        bbox_min: tuple[float, float, float] = (0.0, 0.0, 0.0),
        bbox_max: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hashmap_size = 2**log2_hashmap_size

        self.register_buffer(
            "bbox_min",
            torch.tensor(bbox_min, dtype=torch.float32),
        )
        self.register_buffer(
            "bbox_max",
            torch.tensor(bbox_max, dtype=torch.float32),
        )

        if num_levels > 1:
            growth = math.exp(
                (math.log(max_resolution) - math.log(base_resolution)) / (num_levels - 1)
            )
        else:
            growth = 1.0

        resolutions = [int(base_resolution * growth**i) for i in range(num_levels)]
        self.register_buffer("resolutions", torch.tensor(resolutions, dtype=torch.long))

        self.hash_tables = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(self.hashmap_size, features_per_level).uniform_(-1e-4, 1e-4)
                )
                for _ in range(num_levels)
            ]
        )

        self.register_buffer(
            "_primes",
            torch.tensor([1, 2654435761, 805459861], dtype=torch.long),
        )

    @property
    def output_dim(self) -> int:
        return self.num_levels * self.features_per_level

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Encode 3D positions via multi-resolution hash lookup.

        Args:
            positions: (..., 3) world-space positions.

        Returns:
            (..., output_dim) encoded features.
        """
        orig_shape = positions.shape[:-1]
        x = positions.reshape(-1, 3)

        normalized = (x - self.bbox_min) / (self.bbox_max - self.bbox_min + 1e-8)
        normalized = normalized.clamp(0.0, 1.0)

        level_features = []
        for level in range(self.num_levels):
            resolution = self.resolutions[level].item()
            feats = self._trilinear_lookup(normalized, self.hash_tables[level], resolution)
            level_features.append(feats)

        output = torch.cat(level_features, dim=-1)
        return output.reshape(*orig_shape, self.output_dim)

    def _trilinear_lookup(
        self,
        x: torch.Tensor,
        table: nn.Parameter,
        resolution: int,
    ) -> torch.Tensor:
        """Trilinear interpolation from hash grid at a given resolution.

        Args:
            x: (N, 3) normalized positions in [0, 1].
            table: (hashmap_size, features_per_level) hash table.
            resolution: Grid resolution for this level.

        Returns:
            (N, features_per_level) interpolated features.
        """
        scaled = x * resolution
        floor_coords = scaled.floor().long()
        frac = scaled - floor_coords.float()

        offsets = torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ],
            device=x.device,
            dtype=torch.long,
        )

        corners = floor_coords.unsqueeze(1) + offsets.unsqueeze(0)
        hashed = self._spatial_hash(corners) % self.hashmap_size
        corner_feats = table[hashed.reshape(-1)].reshape(x.shape[0], 8, -1)

        fx, fy, fz = frac[:, 0:1], frac[:, 1:2], frac[:, 2:3]

        c000 = corner_feats[:, 0]
        c100 = corner_feats[:, 1]
        c010 = corner_feats[:, 2]
        c001 = corner_feats[:, 3]
        c110 = corner_feats[:, 4]
        c101 = corner_feats[:, 5]
        c011 = corner_feats[:, 6]
        c111 = corner_feats[:, 7]

        c00 = c000 * (1 - fx) + c100 * fx
        c01 = c001 * (1 - fx) + c101 * fx
        c10 = c010 * (1 - fx) + c110 * fx
        c11 = c011 * (1 - fx) + c111 * fx

        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy

        result = c0 * (1 - fz) + c1 * fz
        return result

    def _spatial_hash(self, coords: torch.Tensor) -> torch.Tensor:
        """Hash 3D integer coordinates using XOR-based spatial hashing."""
        return (coords * self._primes).sum(dim=-1).abs()


@MODELS.register("instant_ngp_encoding")
class InstantNGPHashEncoding(nn.Module):
    """Full Instant-NGP encoding module with hash grid and small MLP head.

    Combines multi-resolution hash encoding with a density and color MLP
    for efficient NeRF-style scene representation.
    """

    def __init__(
        self,
        num_levels: int = 16,
        features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        max_resolution: int = 2048,
        hidden_dim: int = 64,
        num_layers: int = 2,
        geo_feat_dim: int = 15,
        num_dir_frequencies: int = 4,
    ) -> None:
        super().__init__()
        self.hash_encoding = MultiResolutionHashEncoding(
            num_levels=num_levels,
            features_per_level=features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            max_resolution=max_resolution,
        )
        self.geo_feat_dim = geo_feat_dim
        enc_dim = self.hash_encoding.output_dim

        density_layers = [nn.Linear(enc_dim, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 1):
            density_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)])
        density_layers.append(nn.Linear(hidden_dim, 1 + geo_feat_dim))
        self.density_net = nn.Sequential(*density_layers)

        dir_enc_dim = 3 * (2 * num_dir_frequencies + 1)
        self.color_net = nn.Sequential(
            nn.Linear(geo_feat_dim + dir_enc_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )

        self.num_dir_frequencies = num_dir_frequencies

    @property
    def output_dim(self) -> int:
        return self.hash_encoding.output_dim

    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Query density and color at given positions.

        Args:
            positions: (..., 3) world-space positions.
            directions: (..., 3) view directions (optional, for color).

        Returns:
            Dict with 'density' (..., 1) and optionally 'rgb' (..., 3).
        """
        encoded = self.hash_encoding(positions)
        h = self.density_net(encoded)
        density = F.softplus(h[..., :1])
        geo_feat = h[..., 1:]

        result: dict[str, torch.Tensor] = {"density": density}

        if directions is not None:
            dir_enc = self._encode_directions(F.normalize(directions, dim=-1))
            color_input = torch.cat([geo_feat, dir_enc], dim=-1)
            rgb = self.color_net(color_input)
            result["rgb"] = rgb

        return result

    def _encode_directions(self, dirs: torch.Tensor) -> torch.Tensor:
        freqs = 2.0 ** torch.arange(
            self.num_dir_frequencies,
            device=dirs.device,
            dtype=dirs.dtype,
        )
        encoded = [dirs]
        for f in freqs:
            encoded.append(torch.sin(f * dirs))
            encoded.append(torch.cos(f * dirs))
        return torch.cat(encoded, dim=-1)

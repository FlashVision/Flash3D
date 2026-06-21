"""Point cloud processing utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class PointCloud:
    """3D point cloud with optional colors and normals.

    Supports I/O (PLY, numpy), filtering, downsampling, and conversion
    to Gaussian Splatting initialization.
    """

    def __init__(
        self,
        points: torch.Tensor,
        colors: torch.Tensor | None = None,
        normals: torch.Tensor | None = None,
    ) -> None:
        """
        Args:
            points: (N, 3) xyz coordinates.
            colors: (N, 3) RGB values in [0, 1].
            normals: (N, 3) surface normals.
        """
        self.points = points
        self.colors = colors
        self.normals = normals

    @property
    def num_points(self) -> int:
        return self.points.shape[0]

    @property
    def device(self) -> torch.device:
        return self.points.device

    @property
    def centroid(self) -> torch.Tensor:
        return self.points.mean(dim=0)

    @property
    def bounding_box(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.points.min(dim=0).values, self.points.max(dim=0).values

    def to(self, device: torch.device) -> PointCloud:
        """Move point cloud to device."""
        points = self.points.to(device)
        colors = self.colors.to(device) if self.colors is not None else None
        normals = self.normals.to(device) if self.normals is not None else None
        return PointCloud(points, colors, normals)

    def normalize(self, target_radius: float = 1.0) -> PointCloud:
        """Center and scale point cloud to fit in a sphere."""
        centroid = self.centroid
        points = self.points - centroid
        max_dist = points.norm(dim=-1).max()
        if max_dist > 0:
            points = points * (target_radius / max_dist)
        return PointCloud(points, self.colors, self.normals)

    def random_subsample(self, num_points: int) -> PointCloud:
        """Randomly subsample to a fixed number of points."""
        if self.num_points <= num_points:
            return self
        indices = torch.randperm(self.num_points)[:num_points]
        return self._index(indices)

    def voxel_downsample(self, voxel_size: float) -> PointCloud:
        """Voxel grid downsampling."""
        quantized = torch.floor(self.points / voxel_size).long()

        unique_voxels = {}
        for i in range(self.num_points):
            key = tuple(quantized[i].tolist())
            if key not in unique_voxels:
                unique_voxels[key] = []
            unique_voxels[key].append(i)

        indices = []
        for voxel_indices in unique_voxels.values():
            indices.append(voxel_indices[0])

        return self._index(torch.tensor(indices, device=self.device))

    def statistical_outlier_removal(
        self,
        k_neighbors: int = 20,
        std_ratio: float = 2.0,
    ) -> PointCloud:
        """Remove statistical outliers based on mean distance to k nearest neighbors."""

        dists = torch.cdist(self.points, self.points)
        dists.fill_diagonal_(float("inf"))
        knn_dists, _ = dists.topk(k_neighbors, largest=False, dim=-1)
        mean_dists = knn_dists.mean(dim=-1)

        global_mean = mean_dists.mean()
        global_std = mean_dists.std()
        mask = mean_dists < (global_mean + std_ratio * global_std)

        return self._index(mask.nonzero(as_tuple=False).squeeze(-1))

    def estimate_normals(self, k_neighbors: int = 30) -> PointCloud:
        """Estimate surface normals using PCA on local neighborhoods."""
        dists = torch.cdist(self.points, self.points)
        _, knn_indices = dists.topk(k_neighbors, largest=False, dim=-1)

        normals = torch.zeros_like(self.points)
        for i in range(self.num_points):
            neighbors = self.points[knn_indices[i]]
            centered = neighbors - neighbors.mean(dim=0)
            cov = centered.T @ centered / k_neighbors
            _, _, Vh = torch.linalg.svd(cov)
            normals[i] = Vh[-1]

        return PointCloud(self.points, self.colors, normals)

    def _index(self, indices: torch.Tensor) -> PointCloud:
        points = self.points[indices]
        colors = self.colors[indices] if self.colors is not None else None
        normals = self.normals[indices] if self.normals is not None else None
        return PointCloud(points, colors, normals)

    @classmethod
    def from_ply(cls, path: str | Path) -> PointCloud:
        """Load point cloud from PLY file."""
        from plyfile import PlyData

        plydata = PlyData.read(str(path))
        vertex = plydata["vertex"]

        points = torch.tensor(
            np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1),
            dtype=torch.float32,
        )

        colors = None
        if "red" in vertex.data.dtype.names:
            colors = torch.tensor(
                np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=-1),
                dtype=torch.float32,
            ) / 255.0

        normals = None
        if "nx" in vertex.data.dtype.names:
            normals = torch.tensor(
                np.stack([vertex["nx"], vertex["ny"], vertex["nz"]], axis=-1),
                dtype=torch.float32,
            )

        return cls(points, colors, normals)

    def to_ply(self, path: str | Path) -> None:
        """Save point cloud to PLY file."""
        from plyfile import PlyData, PlyElement

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        pts = self.points.cpu().numpy()
        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
        data = np.zeros(self.num_points, dtype=dtype)
        data["x"] = pts[:, 0]
        data["y"] = pts[:, 1]
        data["z"] = pts[:, 2]

        if self.colors is not None:
            colors_np = (self.colors.cpu().numpy() * 255).astype(np.uint8)
            dtype_color = [("x", "f4"), ("y", "f4"), ("z", "f4"),
                          ("red", "u1"), ("green", "u1"), ("blue", "u1")]
            data_color = np.zeros(self.num_points, dtype=dtype_color)
            data_color["x"] = pts[:, 0]
            data_color["y"] = pts[:, 1]
            data_color["z"] = pts[:, 2]
            data_color["red"] = colors_np[:, 0]
            data_color["green"] = colors_np[:, 1]
            data_color["blue"] = colors_np[:, 2]
            data = data_color

        vertex = PlyElement.describe(data, "vertex")
        PlyData([vertex]).write(str(path))

    @classmethod
    def from_numpy(cls, points: np.ndarray, colors: np.ndarray | None = None) -> PointCloud:
        """Create from numpy arrays."""
        pts = torch.from_numpy(points.astype(np.float32))
        cols = torch.from_numpy(colors.astype(np.float32)) if colors is not None else None
        return cls(pts, cols)

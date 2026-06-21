"""Mesh extraction from implicit 3D representations."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import torch


def extract_mesh_marching_cubes(
    query_fn: Callable[[torch.Tensor], torch.Tensor],
    bounds_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0),
    bounds_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    resolution: int = 128,
    threshold: float = 0.5,
    batch_size: int = 65536,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract a triangle mesh from a density field using marching cubes.

    Args:
        query_fn: Function that takes (N, 3) positions and returns (N,) densities.
        bounds_min: Minimum corner of the query volume.
        bounds_max: Maximum corner of the query volume.
        resolution: Grid resolution along each axis.
        threshold: Iso-surface threshold for marching cubes.
        batch_size: Batch size for querying the density field.
        device: Compute device.

    Returns:
        vertices: (V, 3) mesh vertices.
        faces: (F, 3) triangle face indices.
    """
    x = torch.linspace(bounds_min[0], bounds_max[0], resolution, device=device)
    y = torch.linspace(bounds_min[1], bounds_max[1], resolution, device=device)
    z = torch.linspace(bounds_min[2], bounds_max[2], resolution, device=device)

    grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing="ij")
    grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

    densities = torch.zeros(grid_points.shape[0], device=device)
    for i in range(0, grid_points.shape[0], batch_size):
        batch = grid_points[i : i + batch_size]
        with torch.no_grad():
            densities[i : i + batch_size] = query_fn(batch).squeeze(-1)

    density_grid = densities.cpu().numpy().reshape(resolution, resolution, resolution)

    try:
        from skimage.measure import marching_cubes
        vertices, faces, _, _ = marching_cubes(density_grid, level=threshold)
    except ImportError:
        vertices, faces = _simple_marching_cubes(density_grid, threshold)

    # Scale vertices back to world coordinates
    scale = np.array([
        (bounds_max[0] - bounds_min[0]) / resolution,
        (bounds_max[1] - bounds_min[1]) / resolution,
        (bounds_max[2] - bounds_min[2]) / resolution,
    ])
    offset = np.array(bounds_min)
    vertices = vertices * scale + offset

    return vertices, faces


def _simple_marching_cubes(
    grid: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simplified marching cubes fallback when skimage is not available."""
    vertices = []
    faces = []

    res = grid.shape[0]
    for i in range(res - 1):
        for j in range(res - 1):
            for k in range(res - 1):
                cube = grid[i:i+2, j:j+2, k:k+2]
                above = cube > threshold
                if above.all() or not above.any():
                    continue
                center = np.array([i + 0.5, j + 0.5, k + 0.5])
                vid = len(vertices)
                vertices.append(center)

    if not vertices:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

    return np.array(vertices), np.zeros((0, 3), dtype=np.int64)


def save_mesh_obj(
    vertices: np.ndarray,
    faces: np.ndarray,
    path: str,
    vertex_colors: Optional[np.ndarray] = None,
) -> None:
    """Save mesh to OBJ format.

    Args:
        vertices: (V, 3) vertex positions.
        faces: (F, 3) face indices (0-indexed).
        path: Output file path.
        vertex_colors: (V, 3) optional vertex colors in [0, 1].
    """
    with open(path, "w") as f:
        for i, v in enumerate(vertices):
            if vertex_colors is not None:
                c = vertex_colors[i]
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
            else:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def save_mesh_ply(
    vertices: np.ndarray,
    faces: np.ndarray,
    path: str,
) -> None:
    """Save mesh to PLY format using trimesh."""
    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(path)
    except ImportError:
        save_mesh_obj(vertices, faces, path.replace(".ply", ".obj"))

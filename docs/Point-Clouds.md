# Point Clouds

## Overview

Flash3D provides comprehensive point cloud processing utilities for 3D reconstruction workflows.

## PointCloud Class

```python
from flash3d.geometry.point_cloud import PointCloud

# Load from PLY
pc = PointCloud.from_ply("scene.ply")

# Create from tensors
import torch
points = torch.randn(10000, 3)
colors = torch.rand(10000, 3)
pc = PointCloud(points, colors)

# Operations
pc_norm = pc.normalize(target_radius=1.0)
pc_down = pc.voxel_downsample(voxel_size=0.01)
pc_sub = pc.random_subsample(5000)

# Save
pc.to_ply("output.ply")
```

## Processing Pipeline

### 1. Loading

Supports PLY files (via plyfile), numpy arrays, and COLMAP point clouds.

### 2. Filtering

- **Voxel downsampling**: Uniform spatial sampling
- **Statistical outlier removal**: Remove noise points
- **Random subsampling**: Reduce to target count

### 3. Normal Estimation

```python
pc_with_normals = pc.estimate_normals(k_neighbors=30)
```

### 4. Transformations

```python
from flash3d.geometry.transforms_3d import SE3, rotation_matrix_from_euler

R = rotation_matrix_from_euler(0.1, 0.2, 0.3)
t = torch.tensor([1.0, 0.0, 0.0])
transform = SE3.from_rotation_translation(R, t)

transformed_points = transform.transform_points(pc.points)
```

## Mesh Extraction

Extract triangle meshes from implicit fields:

```python
from flash3d.geometry.mesh import extract_mesh_marching_cubes

vertices, faces = extract_mesh_marching_cubes(
    query_fn=nerf_density_function,
    resolution=256,
    threshold=10.0,
)
```

## Metrics

- **Chamfer Distance**: Symmetric point cloud distance
- **F1-Score**: Precision/recall at distance threshold

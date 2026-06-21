"""Example: Point cloud processing and manipulation.

Demonstrates point cloud I/O, filtering, and conversion to Gaussians.
"""

import torch
import numpy as np
from flash3d.geometry.point_cloud import PointCloud
from flash3d.geometry.transforms_3d import SE3, rotation_matrix_from_euler


def main():
    # Create a synthetic point cloud
    N = 10_000
    points = torch.randn(N, 3)
    colors = torch.rand(N, 3)

    pc = PointCloud(points, colors)
    print(f"Point cloud: {pc.num_points} points")
    print(f"Centroid: {pc.centroid}")
    print(f"Bounding box: {pc.bounding_box}")

    # Normalize
    pc_norm = pc.normalize(target_radius=1.0)
    print(f"\nAfter normalization:")
    print(f"  Centroid: {pc_norm.centroid}")
    bb_min, bb_max = pc_norm.bounding_box
    print(f"  Bounding box: [{bb_min}, {bb_max}]")

    # Downsample
    pc_down = pc.random_subsample(5000)
    print(f"\nAfter random subsampling: {pc_down.num_points} points")

    pc_voxel = pc.voxel_downsample(voxel_size=0.5)
    print(f"After voxel downsampling: {pc_voxel.num_points} points")

    # Apply SE(3) transformation
    R = rotation_matrix_from_euler(0.1, 0.2, 0.3)
    t = torch.tensor([1.0, 2.0, 3.0])
    transform = SE3.from_rotation_translation(R, t)

    transformed_points = transform.transform_points(pc.points)
    print(f"\nAfter SE(3) transform:")
    print(f"  Original centroid: {pc.points.mean(dim=0)}")
    print(f"  Transformed centroid: {transformed_points.mean(dim=0)}")

    # Transform composition
    transform_inv = transform.inverse()
    identity_result = transform @ transform_inv
    print(f"\n  T @ T^-1 ≈ I: {torch.allclose(identity_result.matrix, torch.eye(4), atol=1e-5)}")


if __name__ == "__main__":
    main()

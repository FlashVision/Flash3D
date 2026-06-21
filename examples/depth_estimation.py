"""Example: Monocular depth estimation.

Demonstrates using the DepthEstimator for single-image depth prediction.
"""

import torch
import numpy as np
from flash3d import DepthEstimator
from flash3d.geometry.depth import MonocularDepthEstimator, depth_to_point_cloud


def main():
    # Initialize depth estimator
    estimator = DepthEstimator(device="cpu")
    print("DepthEstimator initialized")

    # Create a synthetic test image
    image = torch.rand(1, 3, 256, 256)

    # Predict depth
    with torch.no_grad():
        depth = estimator.predict_tensor(image)
    print(f"Input image shape: {image.shape}")
    print(f"Predicted depth shape: {depth.shape}")
    print(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}]")

    # Convert depth to point cloud
    intrinsics = torch.tensor([
        [200.0, 0.0, 128.0],
        [0.0, 200.0, 128.0],
        [0.0, 0.0, 1.0],
    ])

    points, _ = depth_to_point_cloud(depth.squeeze(), intrinsics)
    print(f"Point cloud shape: {points.shape}")
    print(f"Point cloud bounds: min={points.min(dim=0).values}, max={points.max(dim=0).values}")

    # Compute depth metrics against pseudo ground truth
    from flash3d.geometry.depth import compute_depth_metrics

    gt_depth = depth * (1.0 + 0.1 * torch.randn_like(depth))
    metrics = compute_depth_metrics(depth.squeeze(), gt_depth.squeeze())
    print(f"\nDepth metrics:")
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}")


if __name__ == "__main__":
    main()

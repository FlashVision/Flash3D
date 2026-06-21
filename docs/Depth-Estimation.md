# Depth Estimation

## Overview

Flash3D provides monocular depth estimation for predicting dense depth maps from single RGB images. This enables:

- Initialization for 3DGS/NeRF
- 3D point cloud generation from single views
- Pose-free reconstruction when combined with feed-forward methods

## Usage

### High-Level API

```python
from flash3d import DepthEstimator

estimator = DepthEstimator()

# Single image
depth = estimator.predict_single("photo.jpg")

# Batch processing
estimator.predict("images/", "depth_output/", save_colormap=True)

# Convert to point cloud
import torch
depth_tensor = estimator.predict_tensor(image_tensor)
result = estimator.depth_to_point_cloud(depth_tensor, intrinsics)
```

### Low-Level API

```python
from flash3d.geometry.depth import MonocularDepthEstimator, depth_to_point_cloud

model = MonocularDepthEstimator(min_depth=0.01, max_depth=100.0)
depth = model(image_batch)  # (B, 1, H, W)

points, _ = depth_to_point_cloud(depth[0, 0], intrinsics)
```

## Metrics

Standard depth estimation metrics:

| Metric | Description |
|--------|-------------|
| AbsRel | Mean absolute relative error |
| SqRel | Mean squared relative error |
| RMSE | Root mean squared error |
| RMSE_log | RMSE in log space |
| δ < 1.25 | Fraction within 1.25x threshold |
| δ < 1.25² | Fraction within 1.5625x threshold |
| δ < 1.25³ | Fraction within ~1.95x threshold |

## Integration with 3DGS

Depth predictions can initialize Gaussian positions:

```python
from flash3d.geometry.depth import depth_to_point_cloud
from flash3d.models.architectures.gaussian_splatting import GaussianSplatting

# Predict depth
depth = estimator.predict_tensor(image)

# Convert to point cloud
points, _ = depth_to_point_cloud(depth.squeeze(), intrinsics)

# Initialize Gaussians
model = GaussianSplatting(config=config)
model.initialize_from_point_cloud(points, colors)
```

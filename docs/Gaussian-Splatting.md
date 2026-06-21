# Gaussian Splatting

## Overview

3D Gaussian Splatting represents scenes as collections of 3D Gaussian primitives, each parameterized by:

- **Position** (xyz): 3D center of the Gaussian
- **Covariance** (scale + rotation): Shape and orientation via log-scale and quaternion
- **Opacity**: Transparency of the primitive (sigmoid-activated)
- **Spherical Harmonics**: View-dependent color representation

## Architecture

Flash3D implements the full 3DGS pipeline:

1. **Initialization**: From COLMAP point cloud or random
2. **Differentiable Rasterization**: Tile-based alpha compositing
3. **Adaptive Density Control**: Split/clone/prune Gaussians
4. **Optimization**: Per-parameter learning rates

## Usage

```python
from flash3d.models.architectures.gaussian_splatting import GaussianSplatting
from flash3d.cfg.config import Flash3DConfig

config = Flash3DConfig()
config.model.num_gaussians = 100_000
config.model.sh_degree = 3

model = GaussianSplatting(config=config)

# Initialize from point cloud
import torch
points = torch.randn(100_000, 3)
colors = torch.rand(100_000, 3)
model.initialize_from_point_cloud(points, colors)

# Render
camera = {"viewmatrix": ..., "projmatrix": ..., "camera_center": ..., ...}
result = model.render(camera)
rgb = result["rgb"]  # (3, H, W)
```

## Training Details

- L1 + SSIM loss combination (0.8/0.2 weighting)
- Densification every 100 iterations (500-15000)
- Opacity reset every 3000 iterations
- Position LR with exponential decay

## 2026 Trends

- **Feed-forward 3DGS**: Single-pass prediction without per-scene optimization
- **Pose-free reconstruction**: No COLMAP pre-processing required
- **Efficient representations**: Anchor-based and compressed Gaussians
- **4D Gaussian Splatting**: Dynamic scene modeling

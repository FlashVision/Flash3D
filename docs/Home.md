# Flash3D Documentation

Welcome to the Flash3D documentation. Flash3D is a production-quality 3D vision library supporting Gaussian Splatting, Neural Radiance Fields, Depth Estimation, and 3D Reconstruction.

## Contents

- [Installation](Installation.md)
- [Quick Start](Quick-Start.md)
- [Gaussian Splatting](Gaussian-Splatting.md)
- [NeRF](NeRF.md)
- [Depth Estimation](Depth-Estimation.md)
- [Point Clouds](Point-Clouds.md)
- [FAQ](FAQ.md)

## Architecture

Flash3D follows a modular design:

- **Models**: Unified interface wrapping 3DGS, NeRF, and feed-forward architectures
- **Rendering**: Differentiable rasterization and volume rendering
- **Geometry**: Point clouds, depth maps, meshes, and 3D transformations
- **Engine**: Training, validation, prediction, and export pipelines
- **Solutions**: High-level APIs for common workflows
- **Analytics**: Benchmarking and quality metrics

## Key Features

- 3D Gaussian Splatting with adaptive density control
- NeRF with hash encoding (instant-NGP style)
- Feed-forward 3DGS for single-pass reconstruction
- Monocular depth estimation
- COLMAP, ScanNet, RealEstate10K, DL3DV dataset support
- Export to PLY, OBJ, ONNX, and .splat formats
- Comprehensive metrics: PSNR, SSIM, LPIPS, Chamfer Distance

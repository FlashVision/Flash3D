# Changelog

All notable changes to Flash3D will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-06-21

### Added
- 3D Gaussian Splatting with adaptive density control (split/clone/prune)
- Neural Radiance Fields with hash encoding and positional encoding
- Feed-forward 3D Gaussian Splatting for single-pass reconstruction
- Monocular depth estimation with U-Net architecture
- Differentiable Gaussian rasterizer (pure PyTorch)
- Volume rendering with hierarchical sampling
- Camera models with ray generation and interpolation
- Spherical harmonics evaluation (degree 0-3)
- Point cloud processing (PLY I/O, voxel downsampling, normal estimation)
- Mesh extraction via marching cubes
- SE(3) transformations and quaternion utilities
- COLMAP, ScanNet, RealEstate10K, DL3DV dataset loaders
- Training engine with per-parameter LR for 3DGS
- Predictor with orbit trajectory rendering
- Exporter supporting PLY, OBJ, ONNX, .splat formats
- Comprehensive metrics: PSNR, SSIM, LPIPS, Chamfer Distance, F1
- LoRA adaptation for efficient fine-tuning
- YAML-based configuration system
- CLI with train, render, reconstruct, depth, export, benchmark commands
- Docker support with GPU passthrough
- CI/CD with GitHub Actions (lint, test, type-check)
- Full documentation and examples

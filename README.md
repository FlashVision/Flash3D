# Flash3D

**Production-quality 3D Vision: Gaussian Splatting, NeRF, Depth Estimation & 3D Reconstruction**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/pytorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/Gaurav14cs17/FlashVision/actions/workflows/ci.yml/badge.svg)](https://github.com/Gaurav14cs17/FlashVision/actions)

---

## Overview

Flash3D is a unified Python framework for state-of-the-art 3D vision, combining Gaussian Splatting, Neural Radiance Fields, monocular depth estimation, and 3D reconstruction into a single, modular library. Built for researchers and engineers who need production-ready 3D pipelines.

### Key Features

- **3D Gaussian Splatting** – Per-scene optimization with adaptive density control
- **Feed-Forward 3DGS** – Single-pass reconstruction without per-scene training (pixelSplat/YoNoSplat-style)
- **Neural Radiance Fields** – MLP-based volume rendering with hash encoding (instant-NGP)
- **Monocular Depth** – Foundation model-compatible depth estimation
- **Multi-format Export** – PLY, OBJ, ONNX, .splat for web viewers
- **Comprehensive Metrics** – PSNR, SSIM, LPIPS, Chamfer Distance, F1
- **LoRA Fine-tuning** – Parameter-efficient adaptation for domain transfer

---

## Installation

```bash
# Standard
pip install -e .

# Full (with visualization, metrics, experiment tracking)
pip install -e ".[full]"

# Development
pip install -e ".[dev,full]"
```

Or use the setup script:

```bash
chmod +x setup_env.sh && ./setup_env.sh
```

---

## Quick Start

### Scene Reconstruction

```python
from flash3d import SceneReconstructor

reconstructor = SceneReconstructor(method="gaussian_splatting")
result = reconstructor.reconstruct("path/to/colmap/scene/", num_iterations=30_000)
```

### Depth Estimation

```python
from flash3d import DepthEstimator

estimator = DepthEstimator()
estimator.predict("images/", "depth_output/")
```

### Novel View Synthesis

```python
from flash3d import ViewSynthesizer

synth = ViewSynthesizer.from_checkpoint("model.pth")
frames = synth.render_orbit(num_frames=120, output_dir="renders/")
```

### CLI

```bash
flash3d train --config configs/flash3d_gaussian_splatting.yaml
flash3d render --checkpoint model.pth --num-frames 120
flash3d depth --input images/ --output depth/
flash3d export --checkpoint model.pth --format ply
flash3d benchmark --checkpoint model.pth --dataset data/test/
```

---

## Supported Models

| Model | Type | Use Case | Paper |
|-------|------|----------|-------|
| 3D Gaussian Splatting | Per-scene | High-quality novel views | Kerbl et al. 2023 |
| NeRF (hash encoding) | Per-scene | Volume rendering | Müller et al. 2022 |
| Feed-Forward 3DGS | Generalizable | Single-pass reconstruction | Charatan et al. 2024 |
| Monocular Depth | Single-image | Depth from RGB | Yang et al. 2024 |

---

## Project Structure

```
Flash3D/
├── flash3d/               # Core library
│   ├── models/            # Model architectures (3DGS, NeRF, FF-3DGS)
│   ├── rendering/         # Differentiable rasterization & ray marching
│   ├── geometry/          # Point clouds, depth, meshes, transforms
│   ├── engine/            # Training, validation, prediction, export
│   ├── tasks/             # Task definitions (NVS, reconstruction, depth)
│   ├── solutions/         # High-level APIs
│   ├── analytics/         # Benchmarking & metrics
│   └── utils/             # I/O, visualization, callbacks
├── configs/               # YAML configuration files
├── examples/              # Usage examples
├── tests/                 # Pytest test suite
├── docs/                  # Documentation
└── docker/                # Containerization
```

---

## Supported Datasets

| Dataset | Format | Scenes | Use |
|---------|--------|--------|-----|
| COLMAP | Sparse reconstruction | Any | Per-scene training |
| ScanNet | RGBD sequences | 1500+ indoor | Depth & reconstruction |
| RealEstate10K | Video frames | 10K sequences | Wide-baseline NVS |
| DL3DV | Multi-view captures | 10K diverse | Large-scale training |

---

## Configuration

Flash3D uses YAML-based configuration:

```yaml
task: novel_view_synthesis
model:
  name: gaussian_splatting
  num_gaussians: 100000
  sh_degree: 3
data:
  dataset: colmap
  root_dir: data/garden/
train:
  max_iterations: 30000
  densify_grad_threshold: 0.0002
```

---

## Benchmarks

Evaluated on Mip-NeRF 360 (outdoor scenes):

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Train Time |
|--------|--------|--------|---------|------------|
| 3DGS | 27.5 | 0.815 | 0.220 | ~25 min |
| NeRF (hash) | 26.8 | 0.790 | 0.250 | ~15 min |
| FF-3DGS | 24.2 | 0.750 | 0.300 | ~0.1s/scene |

---

## Development

```bash
# Run tests
pytest tests/ -v

# Lint
ruff check flash3d/

# Type check
mypy flash3d/
```

---

## Citation

```bibtex
@software{flash3d2026,
  title={Flash3D: Production-quality 3D Vision Library},
  author={Gaurav14cs17},
  year={2026},
  url={https://github.com/Gaurav14cs17/FlashVision/tree/main/Flash3D}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).

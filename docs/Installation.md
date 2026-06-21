# Installation

## Requirements

- Python >= 3.10
- PyTorch >= 2.2.0
- CUDA >= 12.0 (recommended for GPU acceleration)

## Standard Installation

```bash
# Clone the repository
git clone https://github.com/Gaurav14cs17/FlashVision.git
cd FlashVision/Flash3D

# Install with pip
pip install -e .
```

## Full Installation (with all extras)

```bash
pip install -e ".[full]"
```

## Development Installation

```bash
pip install -e ".[dev,full]"
pre-commit install
```

## Using setup script

```bash
chmod +x setup_env.sh
./setup_env.sh
```

## Docker

```bash
cd docker
docker-compose up flash3d
```

## Verify Installation

```bash
flash3d check
flash3d settings
```

## Optional Dependencies

| Package | Purpose |
|---------|---------|
| open3d | Advanced point cloud visualization |
| lpips | Perceptual quality metric |
| pytorch3d | Differentiable rendering utils |
| tensorboard | Training visualization |
| wandb | Experiment tracking |

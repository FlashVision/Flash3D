# Quick Start

## 1. Scene Reconstruction with Gaussian Splatting

```python
from flash3d import SceneReconstructor

reconstructor = SceneReconstructor(method="gaussian_splatting")
result = reconstructor.reconstruct(
    input_path="path/to/colmap/scene/",
    output_path="outputs/my_scene/",
    num_iterations=30_000,
)
print(f"Model saved to: {result['model_path']}")
```

## 2. Depth Estimation

```python
from flash3d import DepthEstimator

estimator = DepthEstimator()
depth_maps = estimator.predict(
    input_path="path/to/images/",
    output_path="depth_results/",
)
```

## 3. Novel View Synthesis

```python
from flash3d import ViewSynthesizer

synth = ViewSynthesizer.from_checkpoint("outputs/checkpoint_final.pth")
frames = synth.render_orbit(num_frames=120, output_dir="renders/")
```

## 4. CLI Usage

```bash
# Train Gaussian Splatting
flash3d train --config configs/flash3d_gaussian_splatting.yaml

# Render novel views
flash3d render --checkpoint outputs/checkpoint_final.pth --num-frames 120

# Estimate depth
flash3d depth --input images/ --output depth_output/

# Export model
flash3d export --checkpoint model.pth --format ply

# Run benchmarks
flash3d benchmark --checkpoint model.pth --dataset data/test/
```

## 5. Configuration

All settings are managed through YAML config files:

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
  lr_position: 0.00016
```

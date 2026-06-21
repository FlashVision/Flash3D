# Neural Radiance Fields (NeRF)

## Overview

NeRF represents scenes as continuous volumetric functions mapping 3D positions and view directions to density and color, rendered via volume rendering.

## Architecture

Flash3D supports two encoding schemes:

### Hash Encoding (Default, instant-NGP style)
- Multi-resolution hash grid with learnable features
- O(1) lookup per level
- Dramatically faster convergence than positional encoding

### Classical Positional Encoding
- Sinusoidal frequency features
- Higher fidelity for simple scenes
- More parameters but well-understood

## Usage

```python
from flash3d.models.architectures.nerf import NeRF
from flash3d.cfg.config import Flash3DConfig

config = Flash3DConfig()
model = NeRF(config=config, use_hash_encoding=True)

# Query the field
positions = torch.randn(1000, 3)
directions = torch.randn(1000, 3)
density, rgb = model.query(positions, directions)

# Full render
camera = {
    "rays_o": rays_o,
    "rays_d": rays_d,
    "image_width": 800,
    "image_height": 800,
}
result = model.render(camera)
```

## Volume Rendering

The classic NeRF integral:

```
C(r) = Σᵢ Tᵢ(1 - exp(-σᵢδᵢ))cᵢ
```

Where:
- Tᵢ: Transmittance (accumulated transparency)
- σᵢ: Volume density at sample i
- δᵢ: Distance between samples
- cᵢ: Color at sample i

## Hierarchical Sampling

Flash3D implements two-pass rendering:
1. Coarse pass with uniform samples
2. Fine pass with importance sampling based on coarse weights

## Training Tips

- Use `near` and `far` bounds appropriate for your scene scale
- Hash encoding: 16 levels, 2 features per level, hashmap size 2^19
- Learning rate: 5e-4 with exponential decay
- White background for bounded synthetic scenes

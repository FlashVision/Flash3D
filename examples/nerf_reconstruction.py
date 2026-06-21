"""Example: Neural Radiance Field (NeRF) scene reconstruction.

Demonstrates NeRF training with hash encoding and volume rendering.
"""

import torch
from flash3d import Flash3D
from flash3d.cfg.config import Flash3DConfig
from flash3d.rendering.ray_marching import sample_along_rays, volume_render_rays


def main():
    config = Flash3DConfig()
    config.model.name = "nerf"
    config.render.near = 2.0
    config.render.far = 6.0

    model = Flash3D(config=config)
    print(f"Model: {model.model_name}")
    print(f"Parameters: {model.num_parameters:,}")

    # Generate sample rays
    rays_o = torch.zeros(100, 3)
    rays_o[:, 2] = -3.0
    rays_d = torch.randn(100, 3)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

    # Sample points along rays
    points, t_vals = sample_along_rays(rays_o, rays_d, near=2.0, far=6.0, num_samples=64)
    print(f"Sampled points shape: {points.shape}")

    # Query the NeRF
    model.eval()
    with torch.no_grad():
        flat_pts = points.reshape(-1, 3)
        flat_dirs = rays_d.unsqueeze(1).expand_as(points).reshape(-1, 3)
        density, rgb = model.backbone.query(flat_pts, flat_dirs)
        print(f"Density shape: {density.shape}")
        print(f"RGB shape: {rgb.shape}")

    # Volume render
    density_per_ray = density.reshape(100, 64)
    rgb_per_ray = rgb.reshape(100, 64, 3)
    result = volume_render_rays(density_per_ray, rgb_per_ray, t_vals[0])
    print(f"Rendered color shape: {result['rgb'].shape}")
    print(f"Rendered depth shape: {result['depth'].shape}")


if __name__ == "__main__":
    main()

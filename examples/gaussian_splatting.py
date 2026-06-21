"""Example: 3D Gaussian Splatting scene reconstruction.

Demonstrates training a 3DGS model from COLMAP data and rendering novel views.
"""

import torch
from flash3d import Flash3D, Trainer
from flash3d.cfg.config import Flash3DConfig
from flash3d.rendering.cameras import Camera


def main():
    config = Flash3DConfig()
    config.model.name = "gaussian_splatting"
    config.model.num_gaussians = 10_000
    config.model.sh_degree = 3
    config.train.max_iterations = 1000
    config.output_dir = "outputs/gs_example/"

    model = Flash3D(config=config)
    print(f"Model: {model.model_name}")
    print(f"Parameters: {model.num_parameters:,}")

    # Initialize from random point cloud
    points = torch.randn(10_000, 3) * 2.0
    colors = torch.rand(10_000, 3)
    model.backbone.initialize_from_point_cloud(points, colors)

    # Render a test view
    camera = Camera(
        fx=800.0, fy=800.0, cx=400.0, cy=400.0,
        width=800, height=800,
        R=torch.eye(3),
        t=torch.tensor([0.0, 0.0, -3.0]),
    )

    model.eval()
    with torch.no_grad():
        result = model.render(camera.to_dict())
        print(f"Rendered RGB shape: {result['rgb'].shape}")
        print(f"Rendered depth shape: {result['depth'].shape}")

    # Train (short demo)
    trainer = Trainer(config=config, model=model)
    metrics = trainer.train(num_iterations=100)
    print(f"Training metrics: {metrics}")


if __name__ == "__main__":
    main()

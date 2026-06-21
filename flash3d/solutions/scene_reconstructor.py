"""SceneReconstructor – End-to-end 3D scene reconstruction solution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from flash3d.cfg.config import Flash3DConfig


class SceneReconstructor:
    """High-level API for 3D scene reconstruction.

    Provides a simple interface to reconstruct scenes from images
    using Gaussian Splatting, NeRF, or feed-forward methods.

    Example:
        >>> reconstructor = SceneReconstructor(method="gaussian_splatting")
        >>> result = reconstructor.reconstruct("path/to/images/", "output/")
    """

    def __init__(
        self,
        method: str = "gaussian_splatting",
        config: Flash3DConfig | None = None,
        device: str = "cuda",
    ) -> None:
        self.method = method
        self.config = config or Flash3DConfig()
        self.config.model.name = method
        self.device = device if torch.cuda.is_available() else "cpu"
        self.config.device = self.device

    def reconstruct(
        self,
        input_path: str | Path,
        output_path: str | Path = "reconstruction/",
        num_iterations: int = 30_000,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Reconstruct a 3D scene from input images.

        Args:
            input_path: Path to directory with images (or COLMAP project).
            output_path: Directory for outputs.
            num_iterations: Training iterations.

        Returns:
            Dict with model path, metrics, and scene statistics.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self.config.data.root_dir = str(input_path)
        self.config.output_dir = str(output_path)
        self.config.train.max_iterations = num_iterations

        from flash3d.engine.exporter import Exporter
        from flash3d.engine.trainer import Trainer
        from flash3d.models.flash3d_model import Flash3D

        model = Flash3D(config=self.config)
        model = model.to(self.device)

        # Initialize from point cloud if available
        if self.method == "gaussian_splatting":
            self._initialize_from_data(model, input_path)

        trainer = Trainer(config=self.config, model=model)
        metrics = trainer.train(num_iterations=num_iterations)

        exporter = Exporter(model=model)
        model_path = exporter.export(format="ply", output_path=output_path / "scene.ply")

        return {
            "model_path": str(model_path),
            "output_dir": str(output_path),
            "method": self.method,
            "metrics": metrics,
            "num_iterations": num_iterations,
        }

    def _initialize_from_data(self, model: Any, input_path: Path) -> None:
        """Initialize Gaussians from COLMAP point cloud if available."""
        points_path = input_path / "sparse" / "0" / "points3D.bin"
        if points_path.exists():
            from flash3d.data.colmap_utils import read_points3d_binary

            points3d = read_points3d_binary(points_path)
            if points3d:
                import numpy as np

                xyz = np.array([p["xyz"] for p in points3d.values()])
                rgb = np.array([p["rgb"] for p in points3d.values()]) / 255.0

                pts_tensor = torch.from_numpy(xyz.astype(np.float32)).to(self.device)
                colors_tensor = torch.from_numpy(rgb.astype(np.float32)).to(self.device)

                model.backbone.initialize_from_point_cloud(pts_tensor, colors_tensor)

    def reconstruct_feed_forward(
        self,
        images: torch.Tensor,
        cameras: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Single-pass reconstruction using feed-forward model.

        Args:
            images: (B, V, 3, H, W) multi-view input images.
            cameras: Optional camera parameters for each view.

        Returns:
            Predicted 3D Gaussian parameters.
        """
        self.config.model.name = "feed_forward_3dgs"
        from flash3d.models.flash3d_model import Flash3D

        model = Flash3D(config=self.config).to(self.device)
        model.eval()

        with torch.no_grad():
            result = model(cameras=cameras, images=images.to(self.device))

        return result

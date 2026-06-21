"""Flash3D Predictor – Inference and novel view rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from flash3d.cfg.config import Flash3DConfig
from flash3d.models.flash3d_model import Flash3D
from flash3d.rendering.cameras import Camera


class Predictor:
    """Inference engine for rendering novel views from trained models."""

    def __init__(
        self,
        model: Flash3D,
        config: Flash3DConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or Flash3DConfig()
        self.device = next(model.parameters()).device
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, **kwargs: Any) -> Predictor:
        """Load predictor from a saved checkpoint."""
        model = Flash3D.from_pretrained(checkpoint_path, **kwargs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        return cls(model)

    @torch.no_grad()
    def render_view(self, camera: Camera) -> dict[str, torch.Tensor]:
        """Render a single novel view.

        Args:
            camera: Camera object defining the viewpoint.

        Returns:
            Dict with 'rgb', 'depth', 'alpha' tensors.
        """
        camera_dict = camera.to_dict()
        camera_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                      for k, v in camera_dict.items()}
        return self.model.render(camera_dict)

    @torch.no_grad()
    def render_trajectory(
        self,
        output_dir: str | Path = "renders/",
        num_frames: int = 120,
        cameras: list[Camera] | None = None,
        image_format: str = "png",
    ) -> list[Path]:
        """Render a camera trajectory and save frames.

        Args:
            output_dir: Directory to save rendered frames.
            num_frames: Number of frames in trajectory.
            cameras: Pre-defined camera list. If None, generates orbit.
            image_format: Output image format.

        Returns:
            List of saved frame paths.
        """
        from PIL import Image

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if cameras is None:
            cameras = self._generate_orbit_cameras(num_frames)

        saved_paths = []
        for i, cam in enumerate(tqdm(cameras, desc="Rendering")):
            result = self.render_view(cam)

            if "rgb" in result:
                rgb = result["rgb"].cpu()
                if rgb.dim() == 3:
                    rgb_np = (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                else:
                    rgb_np = (rgb.numpy() * 255).astype(np.uint8)

                frame_path = output_dir / f"frame_{i:04d}.{image_format}"
                Image.fromarray(rgb_np).save(frame_path)
                saved_paths.append(frame_path)

        return saved_paths

    @torch.no_grad()
    def predict_batch(
        self,
        images: torch.Tensor,
        cameras: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run inference on a batch of images (feed-forward models).

        Args:
            images: (B, C, H, W) or (B, V, C, H, W) input images.
            cameras: Optional camera parameters.

        Returns:
            Model predictions.
        """
        images = images.to(self.device)
        if cameras is not None:
            cameras = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                      for k, v in cameras.items()}
        return self.model(cameras=cameras, images=images)

    def _generate_orbit_cameras(self, num_frames: int) -> list[Camera]:
        """Generate orbit cameras around the scene center."""
        import math

        cameras = []
        width = self.config.render.image_width
        height = self.config.render.image_height
        fx = fy = width * 0.8

        for i in range(num_frames):
            angle = 2 * math.pi * i / num_frames
            radius = 3.0
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            y = 0.5

            eye = torch.tensor([x, y, z])
            center = torch.zeros(3)
            up = torch.tensor([0.0, 1.0, 0.0])

            from flash3d.geometry.transforms_3d import look_at
            view_mat = look_at(eye, center, up)

            cameras.append(Camera(
                fx=fx, fy=fy,
                cx=width / 2, cy=height / 2,
                width=width, height=height,
                R=view_mat[:3, :3],
                t=view_mat[:3, 3],
            ))

        return cameras

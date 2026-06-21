"""ViewSynthesizer – High-level novel view synthesis solution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from flash3d.cfg.config import Flash3DConfig


class ViewSynthesizer:
    """High-level API for novel view synthesis.

    Wraps the training and rendering pipeline into a simple interface.

    Example:
        >>> synth = ViewSynthesizer.from_checkpoint("model.pth")
        >>> frames = synth.render_orbit(num_frames=120)
    """

    def __init__(
        self,
        model: Any | None = None,
        config: Flash3DConfig | None = None,
        device: str = "cuda",
    ) -> None:
        self.config = config or Flash3DConfig()
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = model

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, **kwargs: Any) -> ViewSynthesizer:
        """Load from a trained checkpoint."""
        from flash3d.models.flash3d_model import Flash3D

        model = Flash3D.from_pretrained(checkpoint_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        return cls(model=model, device=device, **kwargs)

    @classmethod
    def train_on_scene(
        cls,
        images_path: str | Path,
        method: str = "gaussian_splatting",
        num_iterations: int = 30_000,
        **kwargs: Any,
    ) -> ViewSynthesizer:
        """Train a model on a scene and return a ready synthesizer."""
        config = Flash3DConfig()
        config.model.name = method
        config.data.root_dir = str(images_path)
        config.train.max_iterations = num_iterations

        from flash3d.engine.trainer import Trainer
        from flash3d.models.flash3d_model import Flash3D

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Flash3D(config=config).to(device)
        trainer = Trainer(config=config, model=model)
        trainer.train()

        model.eval()
        return cls(model=model, config=config, device=device)

    @torch.no_grad()
    def render_view(self, camera_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Render a single view."""
        camera_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                      for k, v in camera_dict.items()}
        return self.model.render(camera_dict)

    @torch.no_grad()
    def render_orbit(
        self,
        num_frames: int = 120,
        radius: float = 3.0,
        elevation: float = 0.3,
        output_dir: str | Path | None = None,
    ) -> list[np.ndarray]:
        """Render an orbital trajectory around the scene.

        Args:
            num_frames: Number of frames to render.
            radius: Orbit radius.
            elevation: Camera elevation.
            output_dir: If provided, save frames to disk.

        Returns:
            List of rendered RGB images as numpy arrays.
        """
        from flash3d.geometry.transforms_3d import generate_orbit_cameras
        from flash3d.rendering.cameras import Camera

        center = torch.zeros(3, device=torch.device("cpu"))
        view_matrices = generate_orbit_cameras(
            center, radius=radius, num_frames=num_frames, elevation=elevation
        )

        width = self.config.render.image_width
        height = self.config.render.image_height
        fx = fy = width * 0.8

        frames = []
        for view_mat in view_matrices:
            cam = Camera(
                fx=fx, fy=fy, cx=width / 2, cy=height / 2,
                width=width, height=height,
                R=view_mat[:3, :3], t=view_mat[:3, 3],
            )
            result = self.render_view(cam.to_dict())

            if "rgb" in result:
                rgb = result["rgb"].cpu().permute(1, 2, 0).numpy()
                rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                frames.append(rgb)

        if output_dir is not None:
            from PIL import Image

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(frames):
                Image.fromarray(frame).save(output_dir / f"frame_{i:04d}.png")

        return frames

    @torch.no_grad()
    def interpolate_views(
        self,
        camera_start: dict[str, Any],
        camera_end: dict[str, Any],
        num_frames: int = 60,
    ) -> list[np.ndarray]:
        """Smoothly interpolate between two camera viewpoints."""
        frames = []
        for i in range(num_frames):
            t = i / max(num_frames - 1, 1)
            interp_camera = {}
            for key in camera_start:
                v_start = camera_start[key]
                v_end = camera_end[key]
                if isinstance(v_start, torch.Tensor):
                    interp_camera[key] = v_start * (1 - t) + v_end * t
                else:
                    interp_camera[key] = v_start

            result = self.render_view(interp_camera)
            if "rgb" in result:
                rgb = result["rgb"].cpu().permute(1, 2, 0).numpy()
                frames.append((np.clip(rgb, 0, 1) * 255).astype(np.uint8))

        return frames

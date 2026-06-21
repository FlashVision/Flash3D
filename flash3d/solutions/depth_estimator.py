"""DepthEstimator – High-level monocular depth estimation solution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from flash3d.geometry.depth import MonocularDepthEstimator as DepthModel


class DepthEstimator:
    """High-level API for monocular depth estimation.

    Example:
        >>> estimator = DepthEstimator()
        >>> depth = estimator.predict_single("image.jpg")
        >>> estimator.predict("images_dir/", "depth_output/")
    """

    def __init__(
        self,
        model: Optional[DepthModel] = None,
        device: str = "cuda",
        min_depth: float = 0.01,
        max_depth: float = 100.0,
    ) -> None:
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = model or DepthModel(min_depth=min_depth, max_depth=max_depth)
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_single(self, image_path: str | Path) -> np.ndarray:
        """Estimate depth for a single image.

        Args:
            image_path: Path to input image.

        Returns:
            (H, W) depth map as numpy array.
        """
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)

        depth = self.model(img_tensor)
        return depth.squeeze().cpu().numpy()

    @torch.no_grad()
    def predict_tensor(self, image: torch.Tensor) -> torch.Tensor:
        """Estimate depth from a tensor.

        Args:
            image: (B, 3, H, W) or (3, H, W) input image tensor.

        Returns:
            (B, 1, H, W) depth tensor.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return self.model(image.to(self.device))

    def predict(
        self,
        input_path: str | Path,
        output_path: str | Path = "depth_output/",
        save_colormap: bool = True,
    ) -> List[Path]:
        """Batch depth estimation on a directory of images.

        Args:
            input_path: Path to image or directory of images.
            output_path: Output directory.
            save_colormap: Whether to save colorized depth maps.

        Returns:
            List of saved depth map paths.
        """
        from PIL import Image

        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if input_path.is_file():
            image_paths = [input_path]
        else:
            image_paths = sorted(
                list(input_path.glob("*.png")) +
                list(input_path.glob("*.jpg")) +
                list(input_path.glob("*.jpeg"))
            )

        saved = []
        for img_path in image_paths:
            depth = self.predict_single(img_path)

            # Save raw depth as numpy
            npy_path = output_path / f"{img_path.stem}_depth.npy"
            np.save(npy_path, depth)
            saved.append(npy_path)

            if save_colormap:
                depth_vis = self._colorize_depth(depth)
                vis_path = output_path / f"{img_path.stem}_depth_vis.png"
                Image.fromarray(depth_vis).save(vis_path)
                saved.append(vis_path)

        return saved

    @staticmethod
    def _colorize_depth(depth: np.ndarray) -> np.ndarray:
        """Convert depth map to colorized visualization."""
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 0:
            normalized = (depth - d_min) / (d_max - d_min)
        else:
            normalized = np.zeros_like(depth)

        # Turbo-like colormap approximation
        r = np.clip(1.0 - 2.0 * np.abs(normalized - 0.75), 0, 1)
        g = np.clip(1.0 - 2.0 * np.abs(normalized - 0.5), 0, 1)
        b = np.clip(1.0 - 2.0 * np.abs(normalized - 0.25), 0, 1)

        colormap = np.stack([r, g, b], axis=-1)
        return (colormap * 255).astype(np.uint8)

    def depth_to_point_cloud(
        self,
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
        image: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Convert estimated depth to 3D point cloud.

        Args:
            depth: (H, W) depth map.
            intrinsics: (3, 3) camera intrinsics.
            image: (3, H, W) optional RGB image for coloring points.

        Returns:
            Dict with 'points' (N, 3) and optionally 'colors' (N, 3).
        """
        from flash3d.geometry.depth import depth_to_point_cloud

        points, _ = depth_to_point_cloud(depth, intrinsics)
        result: Dict[str, torch.Tensor] = {"points": points}

        if image is not None:
            H, W = depth.shape[-2:]
            colors = image.permute(1, 2, 0).reshape(-1, 3)
            result["colors"] = colors

        return result

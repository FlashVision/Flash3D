"""Depth Anything v2 wrapper for monocular depth estimation.

Wraps the HuggingFace Depth Anything v2 model for use in Flash3D
pipelines (point cloud generation, 3DGS initialization, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from flash3d.geometry.depth import depth_to_point_cloud


class DepthAnythingV2(nn.Module):
    """Depth Anything v2 monocular depth estimator.

    Provides metric or relative depth predictions using the Depth Anything v2
    model family (small, base, large) from HuggingFace.
    """

    MODEL_VARIANTS = {
        "small": "depth-anything/Depth-Anything-V2-Small-hf",
        "base": "depth-anything/Depth-Anything-V2-Base-hf",
        "large": "depth-anything/Depth-Anything-V2-Large-hf",
    }

    def __init__(
        self,
        variant: str = "small",
        model_name: Optional[str] = None,
        max_depth: float = 80.0,
        device: str = "auto",
    ) -> None:
        super().__init__()
        self.max_depth = max_depth
        self._model = None
        self._processor = None
        self._variant = variant
        self._model_name = model_name or self.MODEL_VARIANTS.get(variant, self.MODEL_VARIANTS["small"])
        self._fallback = False

        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self._load_model()

    def _load_model(self) -> None:
        try:
            from transformers import AutoModelForDepthEstimation, AutoImageProcessor

            self._processor = AutoImageProcessor.from_pretrained(self._model_name)
            self._model = AutoModelForDepthEstimation.from_pretrained(
                self._model_name, torch_dtype=torch.float32,
            )
            self._model.to(self._device)
            self._model.eval()
            for param in self._model.parameters():
                param.requires_grad = False

        except Exception as e:
            print(f"Warning: Could not load Depth Anything v2 ({self._model_name}): {e}")
            print("Using fallback U-Net depth estimator.")
            from flash3d.geometry.depth import MonocularDepthEstimator
            self._model = MonocularDepthEstimator(
                min_depth=0.01, max_depth=self.max_depth,
            )
            self._model.to(self._device)
            self._fallback = True

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Predict depth from RGB image tensor.

        Args:
            image: (B, 3, H, W) RGB image in [0, 1] or ImageNet-normalized.

        Returns:
            (B, 1, H, W) predicted depth map.
        """
        if self._fallback:
            return self._model(image)

        if self._processor is not None:
            from PIL import Image as PILImage
            import numpy as np
            results = []
            for i in range(image.shape[0]):
                img_np = image[i].cpu().permute(1, 2, 0).numpy()
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)
                pil_img = PILImage.fromarray(img_np)
                inputs = self._processor(images=pil_img, return_tensors="pt")
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                outputs = self._model(**inputs)
                depth = outputs.predicted_depth.unsqueeze(1)
                depth = F.interpolate(
                    depth, size=image.shape[2:], mode="bilinear", align_corners=False,
                )
                results.append(depth)
            return torch.cat(results, dim=0)

        return self._model(image)

    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Path, "PILImage", torch.Tensor],
        output_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Predict depth from various image formats.

        Args:
            image: Image as path, PIL Image, or tensor.
            output_size: Optional (H, W) to resize depth output.

        Returns:
            (1, 1, H, W) depth tensor.
        """
        from PIL import Image as PILImage

        if isinstance(image, (str, Path)):
            image = PILImage.open(image).convert("RGB")

        if isinstance(image, PILImage.Image):
            transform = transforms.Compose([
                transforms.Resize((518, 518)),
                transforms.ToTensor(),
            ])
            tensor = transform(image).unsqueeze(0).to(self._device)
        else:
            tensor = image
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            tensor = tensor.to(self._device)

        depth = self.forward(tensor)

        if output_size is not None:
            depth = F.interpolate(
                depth, size=output_size, mode="bilinear", align_corners=False,
            )

        return depth

    def predict_and_unproject(
        self,
        image: Union[str, Path, torch.Tensor],
        intrinsics: torch.Tensor,
        extrinsics: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Predict depth and unproject to 3D point cloud.

        Args:
            image: Input image.
            intrinsics: (3, 3) camera intrinsic matrix.
            extrinsics: (4, 4) optional camera-to-world matrix.

        Returns:
            (N, 3) point cloud and None for colors.
        """
        depth = self.predict(image)
        depth_2d = depth.squeeze(0).squeeze(0)
        intrinsics = intrinsics.to(depth_2d.device)
        ext = extrinsics.to(depth_2d.device) if extrinsics is not None else None
        return depth_to_point_cloud(depth_2d, intrinsics, ext)

    def batch_predict(
        self, images: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Predict depth for multiple images."""
        return [self.predict(img) for img in images]

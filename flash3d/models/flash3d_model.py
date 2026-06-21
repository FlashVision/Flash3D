"""Flash3D – Unified 3D Vision model interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from flash3d.cfg.config import Flash3DConfig
from flash3d.registry import MODELS


class Flash3D(nn.Module):
    """Unified Flash3D model that wraps architecture-specific backends.

    Supports:
        - gaussian_splatting: 3D Gaussian Splatting (per-scene optimization)
        - nerf: Neural Radiance Fields (MLP-based volume rendering)
        - feed_forward_3dgs: Feed-forward 3DGS prediction (single-pass inference)
    """

    def __init__(
        self,
        config: Flash3DConfig | None = None,
        model_name: str = "gaussian_splatting",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.config = config or Flash3DConfig()
        self.model_name = model_name or self.config.model.name

        self.backbone = self._build_backbone(**kwargs)

    def _build_backbone(self, **kwargs: Any) -> nn.Module:
        if self.model_name not in MODELS:
            from flash3d.models.architectures.feed_forward_3dgs import FeedForward3DGS  # noqa: F401

        return MODELS.build(self.model_name, config=self.config, **kwargs)

    def forward(
        self,
        cameras: dict[str, torch.Tensor] | None = None,
        images: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Forward pass dispatched to the underlying architecture.

        Args:
            cameras: Dict with 'intrinsics', 'extrinsics', 'width', 'height'.
            images: Input images for feed-forward models [B, C, H, W].

        Returns:
            Dict with rendered outputs (rgb, depth, alpha, etc.).
        """
        return self.backbone(cameras=cameras, images=images, **kwargs)

    def render(
        self,
        camera: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Render a single view from the current model state."""
        if hasattr(self.backbone, "render"):
            return self.backbone.render(camera, **kwargs)
        return self.forward(cameras=camera, **kwargs)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str | Path, **kwargs: Any) -> Flash3D:
        """Load a pretrained Flash3D model from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        config_data = checkpoint.get("config", None)
        if config_data is None:
            config = Flash3DConfig()
        elif isinstance(config_data, Flash3DConfig):
            config = config_data
        elif isinstance(config_data, dict):
            config = Flash3DConfig()
            for k, v in config_data.items():
                if k == "model" and isinstance(v, dict):
                    for mk, mv in v.items():
                        if hasattr(config.model, mk):
                            setattr(config.model, mk, mv)
                elif k == "data" and isinstance(v, dict):
                    for dk, dv in v.items():
                        if hasattr(config.data, dk):
                            setattr(config.data, dk, dv)
                elif k == "train" and isinstance(v, dict):
                    for tk, tv in v.items():
                        if hasattr(config.train, tk):
                            setattr(config.train, tk, tv)
                elif k == "render" and isinstance(v, dict):
                    for rk, rv in v.items():
                        if hasattr(config.render, rk):
                            setattr(config.render, rk, rv)
                elif hasattr(config, k):
                    setattr(config, k, v)
        else:
            config = Flash3DConfig()

        model_name = checkpoint.get("model_name", config.model.name)
        model = cls(config=config, model_name=model_name, **kwargs)

        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", {}))
        if state_dict:
            model.load_state_dict(state_dict, strict=False)

        return model

    def save_checkpoint(self, path: str | Path, **extra: Any) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config.to_dict(),
            "model_name": self.model_name,
            **extra,
        }
        torch.save(checkpoint, path)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

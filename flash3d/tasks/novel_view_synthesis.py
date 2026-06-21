"""Novel View Synthesis task definition."""

from __future__ import annotations

from typing import Any

import torch

from flash3d.cfg.config import Flash3DConfig
from flash3d.registry import TASKS


@TASKS.register("novel_view_synthesis")
class NovelViewSynthesisTask:
    """Novel view synthesis: render unseen viewpoints from a trained 3D representation.

    Supports both per-scene optimization (3DGS, NeRF) and feed-forward prediction.
    Evaluation metrics: PSNR, SSIM, LPIPS.
    """

    def __init__(self, config: Flash3DConfig | None = None) -> None:
        self.config = config or Flash3DConfig()
        self.metrics = ["psnr", "ssim", "lpips"]

    def setup(self, model_name: str = "gaussian_splatting") -> None:
        """Set up model and training pipeline for NVS."""
        from flash3d.engine.trainer import Trainer
        from flash3d.models.flash3d_model import Flash3D

        self.config.model.name = model_name
        self.model = Flash3D(config=self.config)
        self.trainer = Trainer(config=self.config, model=self.model)

    def train(self, data_path: str, **kwargs: Any) -> dict[str, float]:
        """Train for novel view synthesis."""
        return self.trainer.train(**kwargs)

    def evaluate(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, float]:
        """Compute NVS metrics between predicted and ground truth views."""
        from flash3d.analytics.metrics import compute_psnr, compute_ssim

        results = {
            "psnr": compute_psnr(pred, target),
            "ssim": compute_ssim(pred, target),
        }
        return results

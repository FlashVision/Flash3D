"""Flash3D Validator – Evaluation on held-out views."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from flash3d.analytics.metrics import compute_psnr, compute_ssim
from flash3d.cfg.config import Flash3DConfig
from flash3d.models.flash3d_model import Flash3D


class Validator:
    """Evaluation engine for computing metrics on validation views."""

    def __init__(
        self,
        model: Flash3D,
        config: Optional[Flash3DConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or Flash3DConfig()
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def validate(
        self,
        dataloader: DataLoader,
        num_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """Run validation and compute metrics.

        Returns:
            Dict with average PSNR, SSIM, and L1 error.
        """
        self.model.eval()

        metrics_sum: Dict[str, float] = {"psnr": 0.0, "ssim": 0.0, "l1": 0.0}
        count = 0

        for batch in tqdm(dataloader, desc="Validating"):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            if "viewmatrix" in batch:
                output = self.model(cameras=batch)
            else:
                output = self.model(images=batch.get("image"))

            if "rgb" in output and "image" in batch:
                pred = output["rgb"]
                target = batch["image"]

                if pred.dim() == 3:
                    pred = pred.unsqueeze(0)
                if target.dim() == 3:
                    target = target.unsqueeze(0)

                for i in range(pred.shape[0]):
                    metrics_sum["psnr"] += compute_psnr(pred[i], target[i])
                    metrics_sum["ssim"] += compute_ssim(pred[i], target[i])
                    metrics_sum["l1"] += torch.nn.functional.l1_loss(pred[i], target[i]).item()
                    count += 1

            if num_samples and count >= num_samples:
                break

        if count == 0:
            return metrics_sum

        return {k: v / count for k, v in metrics_sum.items()}

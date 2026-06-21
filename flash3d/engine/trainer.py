"""Flash3D Trainer – Handles training loops for 3DGS, NeRF, and feed-forward models."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from flash3d.cfg.config import Flash3DConfig
from flash3d.models.flash3d_model import Flash3D


class Trainer:
    """Training engine for Flash3D models.

    Supports:
        - Per-scene optimization (3DGS, NeRF)
        - Feed-forward model training with supervision
        - Adaptive density control for Gaussian Splatting
        - Learning rate scheduling with warmup
    """

    def __init__(
        self,
        config: Optional[Flash3DConfig] = None,
        model: Optional[Flash3D] = None,
        **kwargs: Any,
    ) -> None:
        self.config = config or Flash3DConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")

        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = Flash3D(config=self.config).to(self.device)

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.iteration = 0
        self.best_loss = float("inf")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer with per-parameter learning rates for 3DGS."""
        cfg = self.config.train

        if self.config.model.name == "gaussian_splatting":
            param_groups = [
                {"params": [self.model.backbone.means], "lr": cfg.lr_position, "name": "means"},
                {"params": [self.model.backbone.scales], "lr": cfg.lr_scaling, "name": "scales"},
                {"params": [self.model.backbone.rotations], "lr": cfg.lr_rotation, "name": "rotations"},
                {"params": [self.model.backbone.opacities], "lr": cfg.lr_opacity, "name": "opacities"},
                {"params": [self.model.backbone.sh_coeffs], "lr": cfg.lr_sh, "name": "sh_coeffs"},
            ]
            return torch.optim.Adam(param_groups, eps=1e-15)
        else:
            return torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)

    def _build_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Build learning rate scheduler with exponential decay."""
        return torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.9999
        )

    def train(
        self,
        dataloader: Optional[DataLoader] = None,
        num_iterations: Optional[int] = None,
    ) -> Dict[str, float]:
        """Run the training loop.

        For 3DGS per-scene optimization, iterates over views.
        For feed-forward models, trains on batches.
        """
        max_iter = num_iterations or self.config.train.max_iterations
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.model.train()
        losses_log: Dict[str, list] = {"total": [], "l1": [], "ssim": []}

        pbar = tqdm(range(self.iteration, max_iter), desc="Training")
        for iteration in pbar:
            self.iteration = iteration

            if dataloader is not None:
                batch = next(iter(dataloader))
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
            else:
                batch = self._generate_synthetic_batch()

            loss, loss_dict = self._train_step(batch)

            for key, val in loss_dict.items():
                if key not in losses_log:
                    losses_log[key] = []
                losses_log[key].append(val)

            pbar.set_postfix(loss=f"{loss:.6f}")

            # Densification for 3DGS
            if (self.config.model.name == "gaussian_splatting"
                and self.config.train.densify_from <= iteration <= self.config.train.densify_until
                and iteration % self.config.train.densify_interval == 0):
                self._densify()

            # Save checkpoint
            if iteration > 0 and iteration % self.config.train.save_interval == 0:
                self._save_checkpoint(output_dir / f"checkpoint_{iteration:06d}.pth")

            if self.scheduler is not None:
                self.scheduler.step()

        self._save_checkpoint(output_dir / "checkpoint_final.pth")

        return {k: sum(v[-100:]) / max(len(v[-100:]), 1) for k, v in losses_log.items()}

    def _train_step(self, batch: Dict[str, Any]) -> tuple[float, Dict[str, float]]:
        """Single training iteration."""
        self.optimizer.zero_grad()

        if "viewmatrix" in batch:
            output = self.model(cameras=batch)
        else:
            output = self.model(images=batch.get("image"))

        loss, loss_dict = self._compute_loss(output, batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item(), loss_dict

    def _compute_loss(
        self,
        output: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined training loss."""
        loss = torch.tensor(0.0, device=self.device)
        loss_dict: Dict[str, float] = {}

        if "rgb" in output and "image" in batch:
            target = batch["image"]
            pred = output["rgb"]

            if pred.shape != target.shape:
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(0) if pred.dim() == 3 else pred,
                    size=target.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                if pred.dim() == 4 and target.dim() == 3:
                    pred = pred.squeeze(0)

            l1_loss = torch.nn.functional.l1_loss(pred, target)
            ssim_loss = 1.0 - self._ssim(pred, target)

            loss = 0.8 * l1_loss + 0.2 * ssim_loss
            loss_dict["l1"] = l1_loss.item()
            loss_dict["ssim"] = ssim_loss.item()

        loss_dict["total"] = loss.item()
        return loss, loss_dict

    @staticmethod
    def _ssim(
        pred: torch.Tensor,
        target: torch.Tensor,
        window_size: int = 11,
    ) -> torch.Tensor:
        """Compute structural similarity index."""
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        channels = pred.shape[1]

        kernel = torch.ones(channels, 1, window_size, window_size, device=pred.device)
        kernel = kernel / (window_size * window_size)

        mu_pred = torch.nn.functional.conv2d(pred, kernel, groups=channels, padding=window_size // 2)
        mu_target = torch.nn.functional.conv2d(target, kernel, groups=channels, padding=window_size // 2)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_cross = mu_pred * mu_target

        sigma_pred = torch.nn.functional.conv2d(pred * pred, kernel, groups=channels, padding=window_size // 2) - mu_pred_sq
        sigma_target = torch.nn.functional.conv2d(target * target, kernel, groups=channels, padding=window_size // 2) - mu_target_sq
        sigma_cross = torch.nn.functional.conv2d(pred * target, kernel, groups=channels, padding=window_size // 2) - mu_cross

        ssim_map = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / (
            (mu_pred_sq + mu_target_sq + C1) * (sigma_pred + sigma_target + C2)
        )

        return ssim_map.mean()

    def _densify(self) -> None:
        """Run adaptive density control for Gaussian Splatting."""
        if hasattr(self.model.backbone, "densify_and_prune"):
            self.model.backbone.densify_and_prune(
                grad_threshold=self.config.train.densify_grad_threshold,
            )
            self.optimizer = self._build_optimizer()

    def _generate_synthetic_batch(self) -> Dict[str, torch.Tensor]:
        """Generate a synthetic training batch for testing."""
        W = self.config.render.image_width
        H = self.config.render.image_height
        return {
            "image": torch.rand(3, H, W, device=self.device),
            "viewmatrix": torch.eye(4, device=self.device),
            "projmatrix": torch.eye(4, device=self.device),
            "camera_center": torch.zeros(3, device=self.device),
            "image_width": torch.tensor(W, device=self.device),
            "image_height": torch.tensor(H, device=self.device),
        }

    def _save_checkpoint(self, path: Path) -> None:
        """Save training checkpoint."""
        self.model.save_checkpoint(
            path,
            optimizer_state_dict=self.optimizer.state_dict(),
            iteration=self.iteration,
        )

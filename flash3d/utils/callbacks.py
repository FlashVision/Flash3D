"""Training callbacks for Flash3D."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


class TrainingCallback:
    """Base class for training callbacks."""

    def on_train_start(self, trainer: Any) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        pass

    def on_iteration_start(self, trainer: Any, iteration: int) -> None:
        pass

    def on_iteration_end(self, trainer: Any, iteration: int, loss: float) -> None:
        pass

    def on_validation_end(self, trainer: Any, metrics: Dict[str, float]) -> None:
        pass


class CheckpointCallback(TrainingCallback):
    """Save model checkpoints at regular intervals."""

    def __init__(self, save_dir: str | Path, save_every: int = 5000) -> None:
        self.save_dir = Path(save_dir)
        self.save_every = save_every
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_iteration_end(self, trainer: Any, iteration: int, loss: float) -> None:
        if iteration > 0 and iteration % self.save_every == 0:
            path = self.save_dir / f"checkpoint_{iteration:06d}.pth"
            if hasattr(trainer, "model"):
                trainer.model.save_checkpoint(path, iteration=iteration)


class LoggingCallback(TrainingCallback):
    """Log training metrics to console and optional backends."""

    def __init__(
        self,
        log_every: int = 100,
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        project_name: str = "flash3d",
    ) -> None:
        self.log_every = log_every
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        self.writer = None
        self.project_name = project_name

    def on_train_start(self, trainer: Any) -> None:
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter()
            except ImportError:
                pass

        if self.use_wandb:
            try:
                import wandb
                wandb.init(project=self.project_name)
            except ImportError:
                pass

    def on_iteration_end(self, trainer: Any, iteration: int, loss: float) -> None:
        if iteration % self.log_every != 0:
            return

        if self.writer is not None:
            self.writer.add_scalar("train/loss", loss, iteration)

        if self.use_wandb:
            try:
                import wandb
                wandb.log({"train/loss": loss, "iteration": iteration})
            except (ImportError, Exception):
                pass

    def on_train_end(self, trainer: Any) -> None:
        if self.writer is not None:
            self.writer.close()
        if self.use_wandb:
            try:
                import wandb
                wandb.finish()
            except (ImportError, Exception):
                pass


class EarlyStoppingCallback(TrainingCallback):
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.wait = 0

    def on_validation_end(self, trainer: Any, metrics: Dict[str, float]) -> None:
        loss = metrics.get("total", metrics.get("l1", float("inf")))
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if hasattr(trainer, "should_stop"):
                    trainer.should_stop = True

"""Flash3D Configuration Management."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name: str = "gaussian_splatting"
    num_gaussians: int = 100_000
    sh_degree: int = 3
    use_lora: bool = False
    lora_rank: int = 16
    pretrained: str | None = None


@dataclass
class DataConfig:
    dataset: str = "colmap"
    root_dir: str = "data/"
    image_size: tuple[int, int] = (800, 800)
    num_workers: int = 4
    train_split: float = 0.9
    white_background: bool = False


@dataclass
class TrainConfig:
    max_iterations: int = 30_000
    learning_rate: float = 1.6e-4
    lr_position: float = 1.6e-4
    lr_opacity: float = 0.05
    lr_scaling: float = 5e-3
    lr_rotation: float = 1e-3
    lr_sh: float = 2.5e-3
    densify_from: int = 500
    densify_until: int = 15_000
    densify_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_grad_threshold: float = 0.0002
    batch_size: int = 1
    save_interval: int = 5000
    eval_interval: int = 1000


@dataclass
class RenderConfig:
    image_width: int = 800
    image_height: int = 800
    near: float = 0.01
    far: float = 100.0
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class Flash3DConfig:
    """Top-level configuration for Flash3D training and inference."""

    task: str = "novel_view_synthesis"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    output_dir: str = "outputs/"
    device: str = "cuda"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> Flash3DConfig:
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f)

        config = cls()
        if "task" in raw:
            config.task = raw["task"]
        if "output_dir" in raw:
            config.output_dir = raw["output_dir"]
        if "device" in raw:
            config.device = raw["device"]
        if "seed" in raw:
            config.seed = raw["seed"]

        if "model" in raw:
            for k, v in raw["model"].items():
                if hasattr(config.model, k):
                    setattr(config.model, k, v)

        if "data" in raw:
            for k, v in raw["data"].items():
                if hasattr(config.data, k):
                    setattr(config.data, k, v)

        if "train" in raw:
            for k, v in raw["train"].items():
                if hasattr(config.train, k):
                    setattr(config.train, k, v)

        if "render" in raw:
            for k, v in raw["render"].items():
                if hasattr(config.render, k):
                    setattr(config.render, k, v)

        return config

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        import dataclasses

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _to_dict(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj):
                return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            return obj

        with open(path, "w") as f:
            yaml.dump(_to_dict(self), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict[str, Any]:
        import dataclasses

        return dataclasses.asdict(self)

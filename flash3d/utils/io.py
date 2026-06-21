"""I/O utilities for Flash3D."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_image(
    path: str | Path,
    size: tuple[int, int] | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Load an image as a PyTorch tensor.

    Args:
        path: Path to the image file.
        size: Optional (width, height) to resize.
        normalize: Whether to normalize to [0, 1].

    Returns:
        (3, H, W) float tensor.
    """
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.LANCZOS)

    img_np = np.array(img, dtype=np.float32)
    if normalize:
        img_np = img_np / 255.0

    return torch.from_numpy(img_np).permute(2, 0, 1)


def save_image(
    tensor: torch.Tensor,
    path: str | Path,
    normalize: bool = True,
) -> None:
    """Save a tensor as an image.

    Args:
        tensor: (3, H, W) or (H, W, 3) image tensor.
        path: Output file path.
        normalize: Whether input is in [0, 1] (True) or [0, 255] (False).
    """
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if tensor.dim() == 3 and tensor.shape[0] in (1, 3, 4):
        tensor = tensor.permute(1, 2, 0)

    arr = tensor.detach().cpu().numpy()
    if normalize:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)

    if arr.shape[-1] == 1:
        arr = arr.squeeze(-1)

    Image.fromarray(arr).save(path)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Save a configuration dict to YAML."""
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a PyTorch checkpoint."""
    return torch.load(path, map_location="cpu", weights_only=False)


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    """Save a PyTorch checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

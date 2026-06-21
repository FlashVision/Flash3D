"""Visualization utilities for 3D Vision outputs."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch


def visualize_depth(
    depth: torch.Tensor | np.ndarray,
    colormap: str = "turbo",
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> np.ndarray:
    """Convert depth map to colorized RGB visualization.

    Args:
        depth: (H, W) depth map.
        colormap: Colormap name ('turbo', 'viridis', 'plasma').
        min_val: Minimum depth for normalization.
        max_val: Maximum depth for normalization.

    Returns:
        (H, W, 3) uint8 RGB image.
    """
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()

    if depth.ndim == 3:
        depth = depth.squeeze()

    if min_val is None:
        min_val = depth[depth > 0].min() if (depth > 0).any() else 0.0
    if max_val is None:
        max_val = depth.max()

    normalized = np.clip((depth - min_val) / (max_val - min_val + 1e-8), 0, 1)

    if colormap == "turbo":
        colored = _turbo_colormap(normalized)
    elif colormap == "viridis":
        colored = _viridis_colormap(normalized)
    else:
        colored = _turbo_colormap(normalized)

    return (colored * 255).astype(np.uint8)


def visualize_point_cloud(
    points: torch.Tensor | np.ndarray,
    colors: Optional[torch.Tensor | np.ndarray] = None,
    output_path: Optional[str | Path] = None,
    point_size: float = 1.0,
) -> Optional[np.ndarray]:
    """Visualize a 3D point cloud (saves to file or returns projection).

    Args:
        points: (N, 3) point coordinates.
        colors: (N, 3) optional RGB colors in [0, 1].
        output_path: If provided, saves visualization.
        point_size: Rendering point size.

    Returns:
        (H, W, 3) rendered image if output_path is None, else None.
    """
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(colors, torch.Tensor):
        colors = colors.detach().cpu().numpy()

    # Simple orthographic projection for visualization
    H, W = 800, 800
    image = np.zeros((H, W, 3), dtype=np.float32)

    pts_centered = points - points.mean(axis=0)
    scale = max(pts_centered.max() - pts_centered.min(), 1e-6)
    pts_norm = pts_centered / scale

    px = ((pts_norm[:, 0] + 0.5) * (W - 1)).astype(np.int32)
    py = ((pts_norm[:, 1] + 0.5) * (H - 1)).astype(np.int32)

    valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    px, py = px[valid], py[valid]

    if colors is not None:
        valid_colors = colors[valid]
    else:
        depth_vals = pts_norm[valid, 2]
        d_norm = (depth_vals - depth_vals.min()) / (depth_vals.max() - depth_vals.min() + 1e-8)
        valid_colors = _turbo_colormap(d_norm)

    image[py, px] = valid_colors

    result = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    if output_path is not None:
        from PIL import Image
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result).save(output_path)
        return None

    return result


def create_video_from_frames(
    frames: List[np.ndarray] | List[Path],
    output_path: str | Path,
    fps: int = 30,
) -> Path:
    """Create an MP4 video from a list of frames.

    Args:
        frames: List of RGB numpy arrays or paths to frame images.
        output_path: Output video path.
        fps: Frames per second.

    Returns:
        Path to the saved video.
    """
    import cv2

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not frames:
        raise ValueError("No frames provided")

    if isinstance(frames[0], (str, Path)):
        from PIL import Image
        frames = [np.array(Image.open(f)) for f in frames]

    H, W = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()
    return output_path


def _turbo_colormap(values: np.ndarray) -> np.ndarray:
    """Approximate turbo colormap."""
    r = np.clip(np.where(values < 0.5, 2 * values, 2 - 2 * values), 0, 1)
    g = np.clip(np.where(values < 0.33, 3 * values,
                np.where(values < 0.66, 1.0, 3 - 3 * values)), 0, 1)
    b = np.clip(np.where(values < 0.5, 1 - 2 * values, 0), 0, 1)

    if values.ndim == 1:
        return np.stack([r, g, b], axis=-1)
    return np.stack([r, g, b], axis=-1)


def _viridis_colormap(values: np.ndarray) -> np.ndarray:
    """Approximate viridis colormap."""
    r = np.clip(values * 0.5 + 0.2, 0, 1)
    g = np.clip(values * 0.8, 0, 1)
    b = np.clip(0.8 - values * 0.6, 0, 1)
    return np.stack([r, g, b], axis=-1)

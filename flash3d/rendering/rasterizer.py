"""Differentiable Gaussian Rasterization.

Pure PyTorch implementation of tile-based Gaussian splatting rasterization.
For production use with CUDA acceleration, integrate with gsplat or
diff-gaussian-rasterization packages.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from flash3d.rendering.sh_utils import eval_sh


def rasterize_gaussians(
    means3d: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    sh_coeffs: torch.Tensor,
    viewmatrix: torch.Tensor,
    projmatrix: torch.Tensor,
    camera_center: torch.Tensor,
    image_width: int = 800,
    image_height: int = 800,
    sh_degree: int = 3,
    near: float = 0.01,
    far: float = 100.0,
    antialias: bool = False,
    mip_filter_size: float = 0.3,
) -> Dict[str, torch.Tensor]:
    """Differentiable rasterization of 3D Gaussians.

    Implements alpha-compositing of projected 2D Gaussians sorted by depth.

    Args:
        means3d: (N, 3) Gaussian centers.
        scales: (N, 3) Gaussian scales (positive).
        rotations: (N, 4) Gaussian rotations as unit quaternions.
        opacities: (N, 1) Gaussian opacities in [0, 1].
        sh_coeffs: (N, K, 3) Spherical harmonics coefficients.
        viewmatrix: (4, 4) world-to-camera transform.
        projmatrix: (4, 4) full projection matrix.
        camera_center: (3,) camera position in world coordinates.
        image_width: Output image width.
        image_height: Output image height.
        sh_degree: Maximum SH degree.
        near: Near clipping plane.
        far: Far clipping plane.

    Returns:
        Dict with 'rgb' (3, H, W), 'depth' (1, H, W), 'alpha' (1, H, W).
    """
    device = means3d.device
    N = means3d.shape[0]

    # Transform Gaussians to camera space
    means_hom = torch.cat([means3d, torch.ones(N, 1, device=device)], dim=-1)
    means_cam = (viewmatrix @ means_hom.T).T[:, :3]

    # Cull behind near plane
    visible_mask = means_cam[:, 2] > near
    if not visible_mask.any():
        return {
            "rgb": torch.zeros(3, image_height, image_width, device=device),
            "depth": torch.zeros(1, image_height, image_width, device=device),
            "alpha": torch.zeros(1, image_height, image_width, device=device),
        }

    # Project to screen space
    means_proj_hom = (projmatrix @ means_hom.T).T
    means_ndc = means_proj_hom[:, :3] / (means_proj_hom[:, 3:4] + 1e-7)

    means_screen = torch.zeros(N, 2, device=device)
    means_screen[:, 0] = (means_ndc[:, 0] + 1.0) * 0.5 * image_width
    means_screen[:, 1] = (means_ndc[:, 1] + 1.0) * 0.5 * image_height

    # Compute 2D covariance from 3D covariance projected
    cov2d = _compute_cov2d(
        means3d, scales, rotations, viewmatrix, projmatrix,
        image_width, image_height,
        antialias=antialias, mip_filter_size=mip_filter_size,
    )

    # Evaluate SH for view-dependent color
    view_dirs = F.normalize(means3d - camera_center.unsqueeze(0), dim=-1)
    colors = eval_sh(sh_degree, sh_coeffs, view_dirs)
    colors = torch.clamp(colors + 0.5, 0.0, 1.0)

    # Sort by depth for back-to-front rendering
    depths = means_cam[:, 2]
    sort_indices = torch.argsort(depths)

    # Rasterize using alpha compositing
    rgb_image = torch.zeros(3, image_height, image_width, device=device)
    depth_image = torch.zeros(1, image_height, image_width, device=device)
    alpha_image = torch.zeros(1, image_height, image_width, device=device)

    pixel_coords_y, pixel_coords_x = torch.meshgrid(
        torch.arange(image_height, device=device, dtype=torch.float32),
        torch.arange(image_width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    pixel_coords = torch.stack([pixel_coords_x, pixel_coords_y], dim=-1)

    # Tile-based rasterization (simplified for differentiability)
    tile_size = 16
    n_tiles_x = (image_width + tile_size - 1) // tile_size
    n_tiles_y = (image_height + tile_size - 1) // tile_size

    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            y_start = ty * tile_size
            y_end = min(y_start + tile_size, image_height)
            x_start = tx * tile_size
            x_end = min(x_start + tile_size, image_width)

            tile_center = torch.tensor(
                [(x_start + x_end) / 2.0, (y_start + y_end) / 2.0], device=device
            )
            tile_radius = tile_size * 1.5

            dists = (means_screen - tile_center.unsqueeze(0)).norm(dim=-1)
            tile_mask = visible_mask & (dists < tile_radius + 3.0 * scales.max(dim=-1).values)

            if not tile_mask.any():
                continue

            tile_indices = sort_indices[tile_mask[sort_indices]]
            if tile_indices.numel() == 0:
                continue

            tile_pixels = pixel_coords[y_start:y_end, x_start:x_end]
            H_tile, W_tile = tile_pixels.shape[:2]
            tile_pixels_flat = tile_pixels.reshape(-1, 2)

            remaining_alpha = torch.ones(H_tile * W_tile, device=device)

            for idx in tile_indices[:64]:  # Limit per-tile Gaussians for memory
                mu = means_screen[idx]
                cov = cov2d[idx]
                opacity = opacities[idx, 0]
                color = colors[idx]
                depth_val = depths[idx]

                diff = tile_pixels_flat - mu.unsqueeze(0)

                det = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]
                if det < 1e-6:
                    continue

                inv_cov = torch.stack([
                    torch.stack([cov[1, 1], -cov[0, 1]]),
                    torch.stack([-cov[1, 0], cov[0, 0]]),
                ]) / det

                mahal = (diff @ inv_cov * diff).sum(dim=-1)
                gauss_weight = torch.exp(-0.5 * mahal)
                alpha = (opacity * gauss_weight).clamp(0, 0.99)

                weight = alpha * remaining_alpha

                rgb_contribution = weight.unsqueeze(-1) * color.unsqueeze(0)
                rgb_image[:, y_start:y_end, x_start:x_end] += (
                    rgb_contribution.reshape(H_tile, W_tile, 3).permute(2, 0, 1)
                )
                depth_image[0, y_start:y_end, x_start:x_end] += (
                    (weight * depth_val).reshape(H_tile, W_tile)
                )
                alpha_image[0, y_start:y_end, x_start:x_end] += weight.reshape(H_tile, W_tile)

                remaining_alpha = remaining_alpha * (1 - alpha)

    return {
        "rgb": rgb_image.clamp(0, 1),
        "depth": depth_image,
        "alpha": alpha_image.clamp(0, 1),
    }


def _compute_cov2d(
    means3d: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    viewmatrix: torch.Tensor,
    projmatrix: torch.Tensor,
    image_width: int,
    image_height: int,
    antialias: bool = False,
    mip_filter_size: float = 0.3,
) -> torch.Tensor:
    """Compute 2D covariance matrices from 3D Gaussians via EWA splatting.

    Supports mip-splatting style anti-aliasing by applying a 3D smoothing
    filter that grows with distance before projection, preventing aliasing
    from Gaussians smaller than a pixel.

    Args:
        antialias: Enable mip-splatting / EWA anti-aliasing.
        mip_filter_size: Base filter size for anti-aliasing (in pixels).

    Returns:
        (N, 2, 2) projected covariance matrices.
    """
    N = means3d.shape[0]
    device = means3d.device

    S = torch.diag_embed(scales)
    w, x, y, z = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
    R = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
        2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y),
    ], dim=-1).reshape(N, 3, 3)

    RS = R @ S
    cov3d = RS @ RS.transpose(-1, -2)

    W = viewmatrix[:3, :3]
    means_cam = (W @ means3d.T).T + viewmatrix[:3, 3].unsqueeze(0)
    tz = means_cam[:, 2].clamp(min=0.01)

    fx = projmatrix[0, 0] * image_width * 0.5
    fy = projmatrix[1, 1] * image_height * 0.5

    if antialias:
        # Mip-splatting: apply 3D low-pass filter before projection.
        # The filter size in world space is proportional to distance / focal length,
        # ensuring Gaussians are at least one pixel wide after projection.
        pixel_size_x = tz / fx
        pixel_size_y = tz / fy
        mip_scale = mip_filter_size * torch.stack([pixel_size_x, pixel_size_y, tz * 0.001], dim=-1)
        mip_cov = torch.diag_embed(mip_scale ** 2)
        cov3d_cam = W.unsqueeze(0) @ cov3d @ W.T.unsqueeze(0)
        cov3d_cam = cov3d_cam + mip_cov
    else:
        cov3d_cam = W.unsqueeze(0) @ cov3d @ W.T.unsqueeze(0)

    J = torch.zeros(N, 2, 3, device=device)
    J[:, 0, 0] = fx / tz
    J[:, 0, 2] = -fx * means_cam[:, 0] / (tz * tz)
    J[:, 1, 1] = fy / tz
    J[:, 1, 2] = -fy * means_cam[:, 1] / (tz * tz)

    cov2d = J @ cov3d_cam @ J.transpose(-1, -2)

    if not antialias:
        cov2d[:, 0, 0] += 0.3
        cov2d[:, 1, 1] += 0.3

    return cov2d

"""3D Vision quality metrics: PSNR, SSIM, LPIPS, Chamfer Distance."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Peak Signal-to-Noise Ratio.

    Args:
        pred: Predicted image (C, H, W) or (B, C, H, W) in [0, 1].
        target: Ground truth image, same shape as pred.

    Returns:
        PSNR value in dB.
    """
    mse = F.mse_loss(pred, target).item()
    if mse < 1e-10:
        return 100.0
    return -10.0 * math.log10(mse)


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
) -> float:
    """Compute Structural Similarity Index (SSIM).

    Args:
        pred: (C, H, W) or (B, C, H, W) predicted image.
        target: Ground truth image, same shape.
        window_size: Size of the Gaussian window.
        data_range: Value range of input images.

    Returns:
        SSIM value in [0, 1].
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    channels = pred.shape[1]

    # Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    window = gauss.unsqueeze(1) * gauss.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1)

    pad = window_size // 2

    mu_pred = F.conv2d(pred, window, padding=pad, groups=channels)
    mu_target = F.conv2d(target, window, padding=pad, groups=channels)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_cross = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu_pred_sq
    sigma_target_sq = F.conv2d(target * target, window, padding=pad, groups=channels) - mu_target_sq
    sigma_cross = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_cross

    ssim_map = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / (
        (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
    )

    return ssim_map.mean().item()


def compute_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    net: str = "vgg",
) -> float:
    """Compute Learned Perceptual Image Patch Similarity (LPIPS).

    Requires the `lpips` package to be installed.

    Args:
        pred: (C, H, W) or (B, C, H, W) predicted image in [0, 1].
        target: Ground truth image.
        net: Backbone network ('vgg' or 'alex').

    Returns:
        LPIPS distance (lower is better).
    """
    try:
        import lpips

        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        # LPIPS expects images in [-1, 1]
        pred_scaled = pred * 2.0 - 1.0
        target_scaled = target * 2.0 - 1.0

        loss_fn = lpips.LPIPS(net=net, verbose=False).to(pred.device)
        with torch.no_grad():
            distance = loss_fn(pred_scaled, target_scaled)
        return distance.item()
    except ImportError:
        return 0.0


def compute_chamfer_distance(
    pred_points: torch.Tensor,
    target_points: torch.Tensor,
) -> float:
    """Compute Chamfer Distance between two point clouds.

    Args:
        pred_points: (N, 3) predicted point cloud.
        target_points: (M, 3) target point cloud.

    Returns:
        Symmetric Chamfer distance.
    """
    dist_matrix = torch.cdist(pred_points, target_points)

    # pred -> target
    min_dist_pred, _ = dist_matrix.min(dim=1)
    # target -> pred
    min_dist_target, _ = dist_matrix.min(dim=0)

    chamfer = min_dist_pred.mean() + min_dist_target.mean()
    return chamfer.item()


def compute_f1_score(
    pred_points: torch.Tensor,
    target_points: torch.Tensor,
    threshold: float = 0.01,
) -> float:
    """Compute F1-score for point cloud reconstruction quality.

    Args:
        pred_points: (N, 3) predicted points.
        target_points: (M, 3) ground truth points.
        threshold: Distance threshold for considering a match.

    Returns:
        F1-score in [0, 1].
    """
    dist_matrix = torch.cdist(pred_points, target_points)

    precision = (dist_matrix.min(dim=1).values < threshold).float().mean().item()
    recall = (dist_matrix.min(dim=0).values < threshold).float().mean().item()

    if precision + recall < 1e-8:
        return 0.0
    return 2 * precision * recall / (precision + recall)

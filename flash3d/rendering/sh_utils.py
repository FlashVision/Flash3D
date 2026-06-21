"""Spherical Harmonics utilities for view-dependent color representation."""

from __future__ import annotations

import torch
import math


# SH basis constants
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]


def eval_sh(degree: int, sh_coeffs: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Evaluate spherical harmonics at given directions.

    Args:
        degree: Maximum SH degree (0-3).
        sh_coeffs: (N, K, 3) SH coefficients where K >= (degree+1)^2.
        dirs: (N, 3) normalized view directions.

    Returns:
        (N, 3) evaluated color values.
    """
    N = sh_coeffs.shape[0]
    result = torch.zeros(N, 3, device=sh_coeffs.device)

    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]

    # Degree 0
    result += SH_C0 * sh_coeffs[:, 0]

    if degree < 1:
        return result

    # Degree 1
    result += -SH_C1 * y.unsqueeze(-1) * sh_coeffs[:, 1]
    result += SH_C1 * z.unsqueeze(-1) * sh_coeffs[:, 2]
    result += -SH_C1 * x.unsqueeze(-1) * sh_coeffs[:, 3]

    if degree < 2:
        return result

    # Degree 2
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    result += SH_C2[0] * xy.unsqueeze(-1) * sh_coeffs[:, 4]
    result += SH_C2[1] * yz.unsqueeze(-1) * sh_coeffs[:, 5]
    result += SH_C2[2] * (2.0 * zz - xx - yy).unsqueeze(-1) * sh_coeffs[:, 6]
    result += SH_C2[3] * xz.unsqueeze(-1) * sh_coeffs[:, 7]
    result += SH_C2[4] * (xx - yy).unsqueeze(-1) * sh_coeffs[:, 8]

    if degree < 3:
        return result

    # Degree 3
    result += SH_C3[0] * (y * (3.0 * xx - yy)).unsqueeze(-1) * sh_coeffs[:, 9]
    result += SH_C3[1] * (xy * z).unsqueeze(-1) * sh_coeffs[:, 10]
    result += SH_C3[2] * (y * (4.0 * zz - xx - yy)).unsqueeze(-1) * sh_coeffs[:, 11]
    result += SH_C3[3] * (z * (2.0 * zz - 3.0 * xx - 3.0 * yy)).unsqueeze(-1) * sh_coeffs[:, 12]
    result += SH_C3[4] * (x * (4.0 * zz - xx - yy)).unsqueeze(-1) * sh_coeffs[:, 13]
    result += SH_C3[5] * (z * (xx - yy)).unsqueeze(-1) * sh_coeffs[:, 14]
    result += SH_C3[6] * (x * (xx - 3.0 * yy)).unsqueeze(-1) * sh_coeffs[:, 15]

    return result


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB color to 0th-order SH coefficient."""
    return (rgb - 0.5) / SH_C0


def sh_to_rgb(sh0: torch.Tensor) -> torch.Tensor:
    """Convert 0th-order SH coefficient to RGB."""
    return sh0 * SH_C0 + 0.5


def get_num_sh_coeffs(degree: int) -> int:
    """Get number of SH coefficients for a given degree."""
    return (degree + 1) ** 2


def random_sh_coeffs(
    num_points: int,
    degree: int = 3,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generate random SH coefficients for testing.

    Returns:
        (N, (degree+1)^2, 3) tensor of SH coefficients.
    """
    num_coeffs = get_num_sh_coeffs(degree)
    sh = torch.randn(num_points, num_coeffs, 3, device=device) * 0.1
    sh[:, 0] = rgb_to_sh(torch.rand(num_points, 3, device=device))
    return sh

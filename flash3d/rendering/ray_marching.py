"""NeRF Volume Rendering – Ray marching with alpha compositing."""

from __future__ import annotations

import torch


def volume_render_rays(
    density: torch.Tensor,
    rgb: torch.Tensor,
    t_vals: torch.Tensor,
    white_background: bool = False,
) -> dict[str, torch.Tensor]:
    """Classic NeRF volume rendering via numerical quadrature.

    Implements the integral: C(r) = sum_i T_i (1 - exp(-sigma_i * delta_i)) * c_i
    where T_i = exp(-sum_{j<i} sigma_j * delta_j).

    Args:
        density: (..., N_samples) volume densities along each ray.
        rgb: (..., N_samples, 3) color values along each ray.
        t_vals: (N_samples,) distance values along rays.
        white_background: Whether to composite over white background.

    Returns:
        Dict with:
            'rgb': (..., 3) rendered color.
            'depth': (...,) expected depth.
            'alpha': (...,) accumulated opacity.
            'weights': (..., N_samples) sample weights.
    """
    # Compute distances between samples
    dists = t_vals[..., 1:] - t_vals[..., :-1]
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)

    if density.dim() > 1 and dists.dim() == 1:
        dists = dists.expand_as(density)

    # Alpha from density: alpha_i = 1 - exp(-sigma_i * delta_i)
    alpha = 1.0 - torch.exp(-density * dists)

    # Transmittance: T_i = prod_{j<i} (1 - alpha_j)
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1,
    )[..., :-1]

    # Sample weights
    weights = alpha * transmittance

    # Composite color
    rgb_rendered = (weights.unsqueeze(-1) * rgb).sum(dim=-2)

    # Expected depth
    depth = (weights * t_vals.expand_as(weights)).sum(dim=-1)

    # Accumulated opacity
    acc = weights.sum(dim=-1)

    if white_background:
        rgb_rendered = rgb_rendered + (1.0 - acc.unsqueeze(-1))

    return {
        "rgb": rgb_rendered,
        "depth": depth,
        "alpha": acc,
        "weights": weights,
    }


def hierarchical_sampling(
    t_vals: torch.Tensor,
    weights: torch.Tensor,
    num_importance_samples: int = 64,
    deterministic: bool = False,
) -> torch.Tensor:
    """Hierarchical importance sampling for NeRF.

    Samples more densely in regions with high weight (likely surfaces).

    Args:
        t_vals: (..., N_coarse) coarse sample distances.
        weights: (..., N_coarse) coarse sample weights.
        num_importance_samples: Number of fine samples to add.
        deterministic: Whether to use deterministic sampling.

    Returns:
        (..., N_coarse + num_importance_samples) combined sorted t values.
    """
    weights = weights + 1e-5
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    if deterministic:
        u = torch.linspace(0.0, 1.0, num_importance_samples, device=t_vals.device)
        u = u.expand(*cdf.shape[:-1], num_importance_samples)
    else:
        u = torch.rand(*cdf.shape[:-1], num_importance_samples, device=t_vals.device)

    # Inverse CDF sampling
    indices = torch.searchsorted(cdf, u, right=True)
    below = (indices - 1).clamp(min=0)
    above = indices.clamp(max=cdf.shape[-1] - 1)

    cdf_below = torch.gather(cdf, -1, below)
    cdf_above = torch.gather(cdf, -1, above)
    t_below = torch.gather(t_vals, -1, below.clamp(max=t_vals.shape[-1] - 1))
    t_above = torch.gather(t_vals, -1, above.clamp(max=t_vals.shape[-1] - 1))

    denom = cdf_above - cdf_below
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t_fine = t_below + (u - cdf_below) / denom * (t_above - t_below)

    # Combine coarse and fine samples
    t_combined, _ = torch.sort(torch.cat([t_vals, t_fine], dim=-1), dim=-1)
    return t_combined


def sample_along_rays(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float = 0.01,
    far: float = 100.0,
    num_samples: int = 64,
    perturb: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample 3D points along camera rays.

    Args:
        rays_o: (..., 3) ray origins.
        rays_d: (..., 3) ray directions.
        near: Near bound.
        far: Far bound.
        num_samples: Number of samples per ray.
        perturb: Whether to add noise to samples.

    Returns:
        points: (..., N_samples, 3) 3D sample positions.
        t_vals: (..., N_samples) distance values.
    """
    t_vals = torch.linspace(near, far, num_samples, device=rays_o.device)
    t_vals = t_vals.expand(*rays_o.shape[:-1], num_samples)

    if perturb:
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], dim=-1)
        lower = torch.cat([t_vals[..., :1], mids], dim=-1)
        t_rand = torch.rand_like(t_vals)
        t_vals = lower + (upper - lower) * t_rand

    points = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * t_vals.unsqueeze(-1)
    return points, t_vals

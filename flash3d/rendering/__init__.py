"""Flash3D rendering module – Differentiable rasterization and ray marching."""

from flash3d.rendering.cameras import Camera, generate_rays, perspective_projection
from flash3d.rendering.rasterizer import rasterize_gaussians
from flash3d.rendering.ray_marching import volume_render_rays
from flash3d.rendering.sh_utils import SH_C0, SH_C1, eval_sh

__all__ = [
    "rasterize_gaussians",
    "volume_render_rays",
    "Camera",
    "generate_rays",
    "perspective_projection",
    "eval_sh",
    "SH_C0",
    "SH_C1",
]

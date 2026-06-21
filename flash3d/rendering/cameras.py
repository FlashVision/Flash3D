"""Camera models and ray generation for 3D rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np


@dataclass
class Camera:
    """Pinhole camera model with intrinsics and extrinsics.

    Attributes:
        fx, fy: Focal lengths in pixels.
        cx, cy: Principal point coordinates.
        width, height: Image dimensions.
        R: (3, 3) rotation matrix (world to camera).
        t: (3,) translation vector.
        near, far: Clipping planes.
    """
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    R: Optional[torch.Tensor] = None
    t: Optional[torch.Tensor] = None
    near: float = 0.01
    far: float = 100.0

    @property
    def intrinsics(self) -> torch.Tensor:
        """Get 3x3 intrinsic matrix."""
        K = torch.eye(3)
        K[0, 0] = self.fx
        K[1, 1] = self.fy
        K[0, 2] = self.cx
        K[1, 2] = self.cy
        return K

    @property
    def extrinsics(self) -> torch.Tensor:
        """Get 4x4 world-to-camera extrinsic matrix."""
        E = torch.eye(4)
        if self.R is not None:
            E[:3, :3] = self.R
        if self.t is not None:
            E[:3, 3] = self.t
        return E

    @property
    def viewmatrix(self) -> torch.Tensor:
        """Get view matrix (world-to-camera)."""
        return self.extrinsics

    @property
    def projmatrix(self) -> torch.Tensor:
        """Get full projection matrix."""
        return self._perspective_matrix() @ self.extrinsics

    @property
    def camera_center(self) -> torch.Tensor:
        """Get camera center in world coordinates."""
        if self.R is not None and self.t is not None:
            return -self.R.T @ self.t
        return torch.zeros(3)

    def _perspective_matrix(self) -> torch.Tensor:
        """Build OpenGL-style perspective projection matrix."""
        n, f = self.near, self.far
        P = torch.zeros(4, 4)
        P[0, 0] = 2.0 * self.fx / self.width
        P[1, 1] = 2.0 * self.fy / self.height
        P[0, 2] = 1.0 - 2.0 * self.cx / self.width
        P[1, 2] = 1.0 - 2.0 * self.cy / self.height
        P[2, 2] = -(f + n) / (f - n)
        P[2, 3] = -2.0 * f * n / (f - n)
        P[3, 2] = -1.0
        return P

    def to_dict(self) -> dict:
        return {
            "viewmatrix": self.viewmatrix,
            "projmatrix": self.projmatrix,
            "camera_center": self.camera_center,
            "image_width": self.width,
            "image_height": self.height,
            "intrinsics": self.intrinsics,
            "extrinsics": self.extrinsics,
        }


def generate_rays(
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    width: int,
    height: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate camera rays for all pixels.

    Args:
        intrinsics: (3, 3) camera intrinsic matrix.
        extrinsics: (4, 4) camera-to-world (or world-to-camera inverse) matrix.
        width: Image width.
        height: Image height.

    Returns:
        rays_o: (H*W, 3) ray origins in world space.
        rays_d: (H*W, 3) ray directions in world space (normalized).
    """
    device = intrinsics.device

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u = torch.arange(width, device=device, dtype=torch.float32)
    v = torch.arange(height, device=device, dtype=torch.float32)
    v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")

    dirs_cam = torch.stack([
        (u_grid - cx) / fx,
        (v_grid - cy) / fy,
        torch.ones_like(u_grid),
    ], dim=-1)

    dirs_cam = dirs_cam.reshape(-1, 3)

    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]

    # If extrinsics is world-to-camera, invert for ray generation
    R_inv = R.T
    t_inv = -R_inv @ t

    rays_d = (R_inv @ dirs_cam.T).T
    rays_d = F.normalize(rays_d, dim=-1)
    rays_o = t_inv.unsqueeze(0).expand_as(rays_d)

    return rays_o, rays_d


def perspective_projection(
    points3d: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project 3D points to 2D pixel coordinates.

    Args:
        points3d: (..., 3) points in world coordinates.
        intrinsics: (3, 3) camera intrinsic matrix.
        extrinsics: (4, 4) world-to-camera matrix.

    Returns:
        pixels: (..., 2) pixel coordinates (u, v).
        depths: (...,) depth values.
    """
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]

    points_cam = (R @ points3d.reshape(-1, 3).T).T + t.unsqueeze(0)
    points_cam = points_cam.reshape(*points3d.shape)

    depths = points_cam[..., 2]
    z = depths.clamp(min=1e-5)

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u = fx * points_cam[..., 0] / z + cx
    v = fy * points_cam[..., 1] / z + cy

    pixels = torch.stack([u, v], dim=-1)
    return pixels, depths


def interpolate_cameras(
    cam_start: Camera,
    cam_end: Camera,
    num_frames: int,
) -> list[Camera]:
    """Smoothly interpolate between two camera poses for trajectory generation.

    Uses spherical linear interpolation (slerp) for rotations and
    linear interpolation for translations.
    """
    cameras = []

    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)

        fx = cam_start.fx * (1 - t) + cam_end.fx * t
        fy = cam_start.fy * (1 - t) + cam_end.fy * t
        cx = cam_start.cx * (1 - t) + cam_end.cx * t
        cy = cam_start.cy * (1 - t) + cam_end.cy * t

        if cam_start.R is not None and cam_end.R is not None:
            R = _slerp_rotation(cam_start.R, cam_end.R, t)
        else:
            R = cam_start.R

        if cam_start.t is not None and cam_end.t is not None:
            trans = cam_start.t * (1 - t) + cam_end.t * t
        else:
            trans = cam_start.t

        cameras.append(Camera(
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=cam_start.width, height=cam_start.height,
            R=R, t=trans,
            near=cam_start.near, far=cam_start.far,
        ))

    return cameras


def _slerp_rotation(R1: torch.Tensor, R2: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between rotation matrices."""
    R_rel = R2 @ R1.T

    trace = R_rel[0, 0] + R_rel[1, 1] + R_rel[2, 2]
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = cos_angle.clamp(-1.0, 1.0)
    angle = torch.acos(cos_angle)

    if angle.abs() < 1e-6:
        return R1

    # Log map: matrix -> axis-angle, scale, then exp map back
    axis_angle = angle * t
    skew = (R_rel - R_rel.T) / (2.0 * torch.sin(angle) + 1e-8)
    axis = torch.stack([skew[2, 1], skew[0, 2], skew[1, 0]])

    K = torch.zeros(3, 3, device=R1.device)
    K[0, 1] = -axis[2]
    K[0, 2] = axis[1]
    K[1, 0] = axis[2]
    K[1, 2] = -axis[0]
    K[2, 0] = -axis[1]
    K[2, 1] = axis[0]

    R_interp = torch.eye(3, device=R1.device) + torch.sin(axis_angle) * K + (1 - torch.cos(axis_angle)) * (K @ K)
    return R_interp @ R1

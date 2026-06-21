"""3D spatial transformations: SE(3), rotations, and coordinate conversions."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class SE3:
    """SE(3) rigid body transformation (rotation + translation).

    Internally stores a 4x4 homogeneous transformation matrix.
    """

    def __init__(self, matrix: torch.Tensor) -> None:
        """
        Args:
            matrix: (4, 4) or (B, 4, 4) homogeneous transform.
        """
        assert matrix.shape[-2:] == (4, 4)
        self._matrix = matrix

    @property
    def matrix(self) -> torch.Tensor:
        return self._matrix

    @property
    def rotation(self) -> torch.Tensor:
        return self._matrix[..., :3, :3]

    @property
    def translation(self) -> torch.Tensor:
        return self._matrix[..., :3, 3]

    @classmethod
    def identity(cls, batch_size: int = 0, device: torch.device = torch.device("cpu")) -> SE3:
        if batch_size > 0:
            return cls(torch.eye(4, device=device).unsqueeze(0).expand(batch_size, -1, -1).clone())
        return cls(torch.eye(4, device=device))

    @classmethod
    def from_rotation_translation(
        cls,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> SE3:
        """Construct from rotation matrix and translation vector."""
        if R.dim() == 2:
            mat = torch.eye(4, device=R.device)
            mat[:3, :3] = R
            mat[:3, 3] = t
        else:
            B = R.shape[0]
            mat = torch.eye(4, device=R.device).unsqueeze(0).expand(B, -1, -1).clone()
            mat[:, :3, :3] = R
            mat[:, :3, 3] = t
        return cls(mat)

    @classmethod
    def from_quaternion_translation(
        cls,
        quat: torch.Tensor,
        trans: torch.Tensor,
    ) -> SE3:
        """Construct from unit quaternion (wxyz) and translation."""
        R = quaternion_to_rotation_matrix(quat)
        return cls.from_rotation_translation(R, trans)

    def inverse(self) -> SE3:
        """Compute the inverse transformation."""
        R_inv = self.rotation.transpose(-1, -2)
        t_inv = -(R_inv @ self.translation.unsqueeze(-1)).squeeze(-1)
        return SE3.from_rotation_translation(R_inv, t_inv)

    def compose(self, other: SE3) -> SE3:
        """Compose two SE(3) transformations: self @ other."""
        return SE3(self._matrix @ other._matrix)

    def transform_points(self, points: torch.Tensor) -> torch.Tensor:
        """Apply transformation to 3D points.

        Args:
            points: (..., 3) point coordinates.

        Returns:
            Transformed points (..., 3).
        """
        R = self.rotation
        t = self.translation
        if R.dim() == 2:
            return (R @ points.T).T + t
        return torch.einsum("...ij,...j->...i", R, points) + t.unsqueeze(-2)

    def __matmul__(self, other: SE3) -> SE3:
        return self.compose(other)


def rotation_matrix_from_euler(
    roll: float,
    pitch: float,
    yaw: float,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create rotation matrix from Euler angles (ZYX convention).

    Args:
        roll: Rotation around X axis (radians).
        pitch: Rotation around Y axis (radians).
        yaw: Rotation around Z axis (radians).

    Returns:
        (3, 3) rotation matrix.
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    R = torch.tensor(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        device=device,
        dtype=torch.float32,
    )

    return R


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (wxyz) to rotation matrix.

    Args:
        q: (..., 4) unit quaternions.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    q = F.normalize(q, dim=-1)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    R = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)

    return R


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to quaternion (wxyz).

    Args:
        R: (..., 3, 3) rotation matrices.

    Returns:
        (..., 4) unit quaternions.
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

    w = 0.5 * torch.sqrt((trace + 1.0).clamp(min=1e-8))
    x = (R[..., 2, 1] - R[..., 1, 2]) / (4.0 * w + 1e-8)
    y = (R[..., 0, 2] - R[..., 2, 0]) / (4.0 * w + 1e-8)
    z = (R[..., 1, 0] - R[..., 0, 1]) / (4.0 * w + 1e-8)

    return F.normalize(torch.stack([w, x, y, z], dim=-1), dim=-1)


def look_at(
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor = None,
) -> torch.Tensor:
    """Construct a look-at view matrix.

    Args:
        eye: (3,) camera position.
        center: (3,) target position.
        up: (3,) up direction (default: Y-up).

    Returns:
        (4, 4) world-to-camera transformation.
    """
    if up is None:
        up = torch.tensor([0.0, 1.0, 0.0], device=eye.device)

    forward = F.normalize(center - eye, dim=-1)
    right = F.normalize(torch.linalg.cross(forward, up), dim=-1)
    new_up = torch.linalg.cross(right, forward)

    mat = torch.eye(4, device=eye.device)
    mat[0, :3] = right
    mat[1, :3] = new_up
    mat[2, :3] = -forward
    mat[:3, 3] = -mat[:3, :3] @ eye

    return mat


def generate_orbit_cameras(
    center: torch.Tensor,
    radius: float = 3.0,
    num_frames: int = 120,
    elevation: float = 0.3,
    device: torch.device = torch.device("cpu"),
) -> list[torch.Tensor]:
    """Generate orbital camera trajectory around a point.

    Returns list of (4, 4) view matrices.
    """
    cameras = []
    for i in range(num_frames):
        angle = 2 * math.pi * i / num_frames
        x = radius * math.cos(angle)
        z = radius * math.sin(angle)
        y = radius * elevation

        eye = torch.tensor([x, y, z], device=device) + center
        mat = look_at(eye, center)
        cameras.append(mat)

    return cameras

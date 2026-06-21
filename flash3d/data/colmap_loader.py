"""Full COLMAP SfM output loader for Flash3D datasets.

Parses cameras, images, and points3D from COLMAP sparse reconstruction
output (both binary and text formats) and provides a unified interface
for dataset creation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from flash3d.data.colmap_utils import (
    read_cameras_binary,
    read_cameras_text,
    read_images_binary,
    read_points3d_binary,
)


def qvec_to_rotation_matrix(qvec: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = qvec
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])
    return R


def camera_params_to_intrinsics(
    model_name: str, params: np.ndarray, width: int, height: int,
) -> np.ndarray:
    """Convert COLMAP camera parameters to 3x3 intrinsic matrix."""
    K = np.eye(3)
    model_name = model_name.upper() if isinstance(model_name, str) else str(model_name)

    if model_name in ("SIMPLE_PINHOLE", "0"):
        f, cx, cy = params[0], params[1], params[2]
        K[0, 0] = K[1, 1] = f
        K[0, 2] = cx
        K[1, 2] = cy
    elif model_name in ("PINHOLE", "1"):
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
    elif model_name in ("SIMPLE_RADIAL", "2"):
        f, cx, cy = params[0], params[1], params[2]
        K[0, 0] = K[1, 1] = f
        K[0, 2] = cx
        K[1, 2] = cy
    elif model_name in ("RADIAL", "3"):
        f, cx, cy = params[0], params[1], params[2]
        K[0, 0] = K[1, 1] = f
        K[0, 2] = cx
        K[1, 2] = cy
    elif model_name in ("OPENCV", "4"):
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
    else:
        if len(params) >= 4:
            K[0, 0] = params[0]
            K[1, 1] = params[1]
            K[0, 2] = params[2]
            K[1, 2] = params[3]
        elif len(params) >= 3:
            K[0, 0] = K[1, 1] = params[0]
            K[0, 2] = params[1]
            K[1, 2] = params[2]

    return K


def get_distortion_params(model_name: str, params: np.ndarray) -> np.ndarray:
    """Extract distortion coefficients from COLMAP camera params."""
    model_name = model_name.upper() if isinstance(model_name, str) else str(model_name)

    if model_name in ("SIMPLE_RADIAL", "2"):
        return np.array([params[3], 0, 0, 0]) if len(params) > 3 else np.zeros(4)
    if model_name in ("RADIAL", "3"):
        k1 = params[3] if len(params) > 3 else 0
        k2 = params[4] if len(params) > 4 else 0
        return np.array([k1, k2, 0, 0])
    if model_name in ("OPENCV", "4"):
        k1 = params[4] if len(params) > 4 else 0
        k2 = params[5] if len(params) > 5 else 0
        p1 = params[6] if len(params) > 6 else 0
        p2 = params[7] if len(params) > 7 else 0
        return np.array([k1, k2, p1, p2])
    return np.zeros(4)


class COLMAPScene:
    """Complete COLMAP scene representation with cameras, images, and 3D points.

    Provides methods to load, access, and convert COLMAP reconstruction
    data into formats suitable for 3DGS and NeRF training.
    """

    def __init__(
        self,
        root_dir: str,
        images_dir: str | None = None,
        sparse_dir: str = "sparse/0",
        load_points: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.images_dir = Path(images_dir) if images_dir else self.root_dir / "images"
        self.sparse_dir = self.root_dir / sparse_dir

        self.cameras: dict[int, dict[str, Any]] = {}
        self.images: dict[int, dict[str, Any]] = {}
        self.points3d: dict[int, dict[str, Any]] = {}

        self._load(load_points)

    def _load(self, load_points: bool) -> None:
        """Load all COLMAP data from binary or text files."""
        cameras_bin = self.sparse_dir / "cameras.bin"
        cameras_txt = self.sparse_dir / "cameras.txt"
        images_bin = self.sparse_dir / "images.bin"
        self.sparse_dir / "images.txt"
        points_bin = self.sparse_dir / "points3D.bin"

        if cameras_bin.exists():
            self.cameras = read_cameras_binary(cameras_bin)
        elif cameras_txt.exists():
            self.cameras = read_cameras_text(cameras_txt)

        if images_bin.exists():
            self.images = read_images_binary(images_bin)

        if load_points and points_bin.exists():
            self.points3d = read_points3d_binary(points_bin)

    @property
    def num_images(self) -> int:
        return len(self.images)

    @property
    def num_cameras(self) -> int:
        return len(self.cameras)

    @property
    def num_points(self) -> int:
        return len(self.points3d)

    def get_image_names(self) -> list[str]:
        """Get sorted list of image filenames."""
        return sorted(img["name"] for img in self.images.values())

    def get_intrinsics(self, image_id: int) -> np.ndarray:
        """Get 3x3 intrinsic matrix for a given image."""
        img_data = self.images[image_id]
        cam_data = self.cameras[img_data["camera_id"]]
        model_name = cam_data.get("model_name", str(cam_data.get("model_id", "")))
        return camera_params_to_intrinsics(
            model_name, cam_data["params"], cam_data["width"], cam_data["height"],
        )

    def get_extrinsics(self, image_id: int) -> np.ndarray:
        """Get 4x4 world-to-camera extrinsic matrix for a given image."""
        img_data = self.images[image_id]
        R = qvec_to_rotation_matrix(img_data["qvec"])
        t = img_data["tvec"]
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = R
        extrinsics[:3, 3] = t
        return extrinsics

    def get_camera_to_world(self, image_id: int) -> np.ndarray:
        """Get 4x4 camera-to-world transform (inverse of extrinsics)."""
        ext = self.get_extrinsics(image_id)
        c2w = np.eye(4)
        c2w[:3, :3] = ext[:3, :3].T
        c2w[:3, 3] = -ext[:3, :3].T @ ext[:3, 3]
        return c2w

    def get_point_cloud(self) -> tuple[np.ndarray, np.ndarray]:
        """Get point cloud as (N, 3) positions and (N, 3) RGB colors in [0, 1]."""
        if not self.points3d:
            return np.zeros((0, 3)), np.zeros((0, 3))

        points = np.array([p["xyz"] for p in self.points3d.values()])
        colors = np.array([p["rgb"] for p in self.points3d.values()]) / 255.0
        return points, colors

    def get_point_cloud_torch(
        self, device: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get point cloud as PyTorch tensors."""
        pts, cols = self.get_point_cloud()
        return (
            torch.from_numpy(pts).float().to(device),
            torch.from_numpy(cols).float().to(device),
        )

    def get_all_cameras_torch(
        self, device: str = "cpu",
    ) -> list[dict[str, torch.Tensor]]:
        """Get all camera parameters as PyTorch tensors.

        Returns list of dicts suitable for Gaussian splatting or NeRF rendering.
        """
        camera_list = []
        for image_id, img_data in sorted(self.images.items()):
            cam_data = self.cameras[img_data["camera_id"]]
            K = self.get_intrinsics(image_id)
            w2c = self.get_extrinsics(image_id)
            c2w = self.get_camera_to_world(image_id)

            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            W, H = cam_data["width"], cam_data["height"]

            proj = _make_projection_matrix(fx, fy, cx, cy, W, H, 0.01, 100.0)

            camera_list.append({
                "image_id": image_id,
                "image_name": img_data["name"],
                "image_width": W,
                "image_height": H,
                "intrinsics": torch.from_numpy(K).float().to(device),
                "extrinsics": torch.from_numpy(w2c).float().to(device),
                "camera_to_world": torch.from_numpy(c2w).float().to(device),
                "viewmatrix": torch.from_numpy(w2c).float().to(device),
                "projmatrix": torch.from_numpy(proj).float().to(device),
                "camera_center": torch.from_numpy(c2w[:3, 3]).float().to(device),
            })

        return camera_list

    def split_train_test(
        self, test_ratio: float = 0.1,
    ) -> tuple[list[int], list[int]]:
        """Split image IDs into train and test sets."""
        image_ids = sorted(self.images.keys())
        n_test = max(1, int(len(image_ids) * test_ratio))
        test_ids = image_ids[::len(image_ids) // n_test][:n_test]
        train_ids = [i for i in image_ids if i not in test_ids]
        return train_ids, test_ids

    def compute_scene_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute axis-aligned bounding box of the scene from 3D points."""
        pts, _ = self.get_point_cloud()
        if pts.shape[0] == 0:
            return np.array([-1, -1, -1.0]), np.array([1, 1, 1.0])
        return pts.min(axis=0), pts.max(axis=0)

    def compute_scene_center_and_radius(self) -> tuple[np.ndarray, float]:
        """Compute scene center and radius for normalization."""
        pts, _ = self.get_point_cloud()
        if pts.shape[0] == 0:
            camera_positions = []
            for img_id in self.images:
                c2w = self.get_camera_to_world(img_id)
                camera_positions.append(c2w[:3, 3])
            if camera_positions:
                pts = np.array(camera_positions)
            else:
                return np.zeros(3), 1.0

        center = pts.mean(axis=0)
        radius = float(np.linalg.norm(pts - center, axis=1).max())
        return center, max(radius, 1e-3)


def _make_projection_matrix(
    fx: float, fy: float, cx: float, cy: float,
    width: int, height: int, near: float, far: float,
) -> np.ndarray:
    """Build a 4x4 OpenGL-style projection matrix."""
    proj = np.zeros((4, 4))
    proj[0, 0] = 2 * fx / width
    proj[1, 1] = 2 * fy / height
    proj[0, 2] = 1 - 2 * cx / width
    proj[1, 2] = 2 * cy / height - 1
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -2 * far * near / (far - near)
    proj[3, 2] = -1.0
    return proj

"""3D Vision datasets: COLMAP, ScanNet, RealEstate10K, DL3DV."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from flash3d.data.colmap_utils import read_cameras_binary, read_images_binary, read_points3d_binary
from flash3d.data.transforms import Compose
from flash3d.registry import DATASETS


@DATASETS.register("colmap")
class COLMAPDataset(Dataset):
    """Dataset loader for COLMAP-format scene data.

    Expects structure:
        root/
            images/
            sparse/0/
                cameras.bin
                images.bin
                points3D.bin
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        image_size: tuple[int, int] = (800, 800),
        transforms: Compose | None = None,
        white_background: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.transforms = transforms
        self.white_background = white_background

        self._load_colmap_data()

    def _load_colmap_data(self) -> None:
        sparse_dir = self.root_dir / "sparse" / "0"
        images_dir = self.root_dir / "images"

        if sparse_dir.exists():
            self.cameras = read_cameras_binary(sparse_dir / "cameras.bin")
            self.images_meta = read_images_binary(sparse_dir / "images.bin")
            self.points3d = read_points3d_binary(sparse_dir / "points3D.bin")
        else:
            self.cameras = {}
            self.images_meta = {}
            self.points3d = {}

        self.image_paths = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))

        n = len(self.image_paths)
        split_idx = int(n * 0.9)
        if self.split == "train":
            self.image_paths = self.image_paths[:split_idx]
        else:
            self.image_paths = self.image_paths[split_idx:]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from PIL import Image

        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.image_size, Image.LANCZOS)
        image_np = np.array(image, dtype=np.float32) / 255.0

        if self.white_background:
            image_np = image_np * 1.0 + (1.0 - 1.0) * 1.0

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

        sample = {
            "image": image_tensor,
            "image_path": str(img_path),
            "idx": idx,
        }

        img_name = img_path.name
        for img_id, img_meta in self.images_meta.items():
            if img_meta.get("name") == img_name:
                qvec = img_meta["qvec"]
                tvec = img_meta["tvec"]
                R = self._qvec2rotmat(qvec)
                sample["rotation"] = torch.from_numpy(R).float()
                sample["translation"] = torch.from_numpy(tvec).float()
                sample["camera_id"] = img_meta.get("camera_id", 0)
                break

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    @staticmethod
    def _qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        q = np.array(qvec, dtype=np.float64)
        n = np.dot(q, q)
        if n < 1e-10:
            return np.eye(3)
        q *= np.sqrt(2.0 / n)
        q_outer = np.outer(q, q)
        return np.array([
            [1.0 - q_outer[2, 2] - q_outer[3, 3], q_outer[1, 2] - q_outer[3, 0], q_outer[1, 3] + q_outer[2, 0]],
            [q_outer[1, 2] + q_outer[3, 0], 1.0 - q_outer[1, 1] - q_outer[3, 3], q_outer[2, 3] - q_outer[1, 0]],
            [q_outer[1, 3] - q_outer[2, 0], q_outer[2, 3] + q_outer[1, 0], 1.0 - q_outer[1, 1] - q_outer[2, 2]],
        ])


@DATASETS.register("scannet")
class ScanNetDataset(Dataset):
    """ScanNet indoor scene dataset loader.

    Expects structure:
        root/
            scene0000_00/
                color/
                depth/
                pose/
                intrinsic/
    """

    def __init__(
        self,
        root_dir: str | Path,
        scene_id: str = "scene0000_00",
        split: str = "train",
        image_size: tuple[int, int] = (640, 480),
        frame_skip: int = 10,
        transforms: Compose | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.scene_dir = self.root_dir / scene_id
        self.split = split
        self.image_size = image_size
        self.frame_skip = frame_skip
        self.transforms = transforms

        self._load_frames()

    def _load_frames(self) -> None:
        color_dir = self.scene_dir / "color"
        if not color_dir.exists():
            self.frames: list[dict[str, Any]] = []
            return

        frame_files = sorted(color_dir.glob("*.jpg"))
        frame_files = frame_files[:: self.frame_skip]

        n = len(frame_files)
        split_idx = int(n * 0.8)
        if self.split == "train":
            frame_files = frame_files[:split_idx]
        else:
            frame_files = frame_files[split_idx:]

        self.frames = []
        for f in frame_files:
            frame_id = f.stem
            self.frames.append({
                "color": f,
                "depth": self.scene_dir / "depth" / f"{frame_id}.png",
                "pose": self.scene_dir / "pose" / f"{frame_id}.txt",
            })

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from PIL import Image

        frame = self.frames[idx]

        color = Image.open(frame["color"]).convert("RGB")
        color = color.resize(self.image_size, Image.LANCZOS)
        color_tensor = torch.from_numpy(
            np.array(color, dtype=np.float32) / 255.0
        ).permute(2, 0, 1)

        sample: dict[str, Any] = {"image": color_tensor, "idx": idx}

        if frame["depth"].exists():
            depth = Image.open(frame["depth"])
            depth = depth.resize(self.image_size, Image.NEAREST)
            depth_np = np.array(depth, dtype=np.float32) / 1000.0
            sample["depth"] = torch.from_numpy(depth_np).unsqueeze(0)

        if frame["pose"].exists():
            pose = np.loadtxt(frame["pose"]).reshape(4, 4).astype(np.float32)
            sample["pose"] = torch.from_numpy(pose)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


@DATASETS.register("realestate10k")
class RealEstate10KDataset(Dataset):
    """RealEstate10K dataset for wide-baseline view synthesis.

    Expects pre-extracted frames with camera parameters.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        image_size: tuple[int, int] = (256, 256),
        num_context_views: int = 2,
        transforms: Compose | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.num_context_views = num_context_views
        self.transforms = transforms

        self._load_sequences()

    def _load_sequences(self) -> None:
        split_file = self.root_dir / f"{self.split}.txt"
        self.sequences: list[dict[str, Any]] = []

        if split_file.exists():
            with open(split_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        seq_dir = self.root_dir / parts[0]
                        if seq_dir.exists():
                            frames = sorted(seq_dir.glob("*.png"))
                            if len(frames) > self.num_context_views:
                                self.sequences.append({
                                    "dir": seq_dir,
                                    "frames": frames,
                                })

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from PIL import Image

        seq = self.sequences[idx]
        frames = seq["frames"]

        indices = np.random.choice(len(frames), self.num_context_views + 1, replace=False)
        indices = sorted(indices)

        images = []
        for i in indices:
            img = Image.open(frames[i]).convert("RGB")
            img = img.resize(self.image_size, Image.LANCZOS)
            img_t = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
            images.append(img_t)

        sample = {
            "context_images": torch.stack(images[:-1]),
            "target_image": images[-1],
            "idx": idx,
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


@DATASETS.register("dl3dv")
class DL3DVDataset(Dataset):
    """DL3DV-10K large-scale multi-view dataset.

    Supports diverse real-world scenes with varying complexity.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        image_size: tuple[int, int] = (512, 512),
        max_views_per_scene: int = 50,
        transforms: Compose | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.max_views = max_views_per_scene
        self.transforms = transforms

        self._load_scenes()

    def _load_scenes(self) -> None:
        split_dir = self.root_dir / self.split
        self.scenes: list[Path] = []

        if split_dir.exists():
            self.scenes = sorted([d for d in split_dir.iterdir() if d.is_dir()])

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from PIL import Image

        scene_dir = self.scenes[idx] if self.scenes else self.root_dir
        images_dir = scene_dir / "images"

        image_files = sorted(images_dir.glob("*.jpg"))[:self.max_views] if images_dir.exists() else []

        images = []
        for img_path in image_files[:4]:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(self.image_size, Image.LANCZOS)
            images.append(
                torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
            )

        sample: dict[str, Any] = {"idx": idx, "scene_path": str(scene_dir)}
        if images:
            sample["images"] = torch.stack(images)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

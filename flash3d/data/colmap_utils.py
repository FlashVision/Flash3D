"""COLMAP file parsing utilities for binary and text formats."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np


def read_cameras_binary(path: str | Path) -> dict[int, dict[str, Any]]:
    """Read cameras.bin from COLMAP sparse reconstruction.

    Returns dict mapping camera_id -> {model, width, height, params}.
    """
    cameras = {}
    path = Path(path)
    if not path.exists():
        return cameras

    with open(path, "rb") as f:
        num_cameras = _read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id = _read_next_bytes(f, 4, "i")[0]
            model_id = _read_next_bytes(f, 4, "i")[0]
            width = _read_next_bytes(f, 8, "Q")[0]
            height = _read_next_bytes(f, 8, "Q")[0]

            num_params = _CAMERA_MODEL_NUM_PARAMS.get(model_id, 4)
            params = _read_next_bytes(f, 8 * num_params, "d" * num_params)

            cameras[camera_id] = {
                "model_id": model_id,
                "model_name": _CAMERA_MODEL_NAMES.get(model_id, "UNKNOWN"),
                "width": width,
                "height": height,
                "params": np.array(params),
            }
    return cameras


def read_images_binary(path: str | Path) -> dict[int, dict[str, Any]]:
    """Read images.bin from COLMAP sparse reconstruction.

    Returns dict mapping image_id -> {qvec, tvec, camera_id, name, xys, point3d_ids}.
    """
    images = {}
    path = Path(path)
    if not path.exists():
        return images

    with open(path, "rb") as f:
        num_images = _read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_images):
            image_id = _read_next_bytes(f, 4, "i")[0]
            qvec = np.array(_read_next_bytes(f, 32, "dddd"))
            tvec = np.array(_read_next_bytes(f, 24, "ddd"))
            camera_id = _read_next_bytes(f, 4, "i")[0]

            name = b""
            while True:
                char = f.read(1)
                if char == b"\x00":
                    break
                name += char
            name_str = name.decode("utf-8")

            num_points2d = _read_next_bytes(f, 8, "Q")[0]
            xys = np.zeros((num_points2d, 2))
            point3d_ids = np.zeros(num_points2d, dtype=np.int64)

            for j in range(num_points2d):
                xy = _read_next_bytes(f, 16, "dd")
                xys[j] = xy
                p3d_id = _read_next_bytes(f, 8, "q")[0]
                point3d_ids[j] = p3d_id

            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name_str,
                "xys": xys,
                "point3d_ids": point3d_ids,
            }
    return images


def read_points3d_binary(path: str | Path) -> dict[int, dict[str, Any]]:
    """Read points3D.bin from COLMAP sparse reconstruction.

    Returns dict mapping point3d_id -> {xyz, rgb, error, track}.
    """
    points = {}
    path = Path(path)
    if not path.exists():
        return points

    with open(path, "rb") as f:
        num_points = _read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_points):
            point3d_id = _read_next_bytes(f, 8, "Q")[0]
            xyz = np.array(_read_next_bytes(f, 24, "ddd"))
            rgb = np.array(_read_next_bytes(f, 3, "BBB"))
            error = _read_next_bytes(f, 8, "d")[0]
            track_length = _read_next_bytes(f, 8, "Q")[0]

            track = []
            for _ in range(track_length):
                img_id = _read_next_bytes(f, 4, "i")[0]
                pt2d_idx = _read_next_bytes(f, 4, "i")[0]
                track.append((img_id, pt2d_idx))

            points[point3d_id] = {
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "track": track,
            }
    return points


def read_cameras_text(path: str | Path) -> dict[int, dict[str, Any]]:
    """Read cameras.txt from COLMAP sparse reconstruction."""
    cameras = {}
    path = Path(path)
    if not path.exists():
        return cameras

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model_name = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = np.array([float(p) for p in parts[4:]])
            cameras[camera_id] = {
                "model_name": model_name,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def _read_next_bytes(f, num_bytes: int, format_char_sequence: str) -> tuple:
    data = f.read(num_bytes)
    return struct.unpack("<" + format_char_sequence, data)


_CAMERA_MODEL_NUM_PARAMS = {
    0: 3,  # SIMPLE_PINHOLE
    1: 4,  # PINHOLE
    2: 4,  # SIMPLE_RADIAL
    3: 5,  # RADIAL
    4: 8,  # OPENCV
    5: 12,  # OPENCV_FISHEYE
    6: 5,  # FULL_OPENCV
}

_CAMERA_MODEL_NAMES = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
}

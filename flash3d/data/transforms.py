"""Data transforms for 3D vision datasets."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch


class Compose:
    """Compose multiple transforms sequentially."""

    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for t in self.transforms:
            sample = t(sample)
        return sample


class Resize:
    """Resize image tensors to target size."""

    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if "image" in sample and sample["image"].dim() == 3:
            sample["image"] = torch.nn.functional.interpolate(
                sample["image"].unsqueeze(0),
                size=self.size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        if "depth" in sample and sample["depth"].dim() == 3:
            sample["depth"] = torch.nn.functional.interpolate(
                sample["depth"].unsqueeze(0),
                size=self.size,
                mode="nearest",
            ).squeeze(0)
        return sample


class Normalize:
    """Normalize image tensor with mean and std."""

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if "image" in sample:
            sample["image"] = (sample["image"] - self.mean) / self.std
        return sample


class ToTensor:
    """Convert numpy arrays in sample to tensors."""

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                if value.ndim == 3 and value.shape[2] in (1, 3, 4):
                    sample[key] = torch.from_numpy(value.transpose(2, 0, 1)).float()
                else:
                    sample[key] = torch.from_numpy(value).float()
        return sample


class RandomCrop:
    """Random crop for image tensors."""

    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if "image" not in sample:
            return sample

        _, h, w = sample["image"].shape
        th, tw = self.size

        if h < th or w < tw:
            return sample

        top = np.random.randint(0, h - th + 1)
        left = np.random.randint(0, w - tw + 1)

        sample["image"] = sample["image"][:, top : top + th, left : left + tw]
        if "depth" in sample:
            sample["depth"] = sample["depth"][:, top : top + th, left : left + tw]

        return sample


class RandomHorizontalFlip:
    """Random horizontal flip with associated camera adjustments."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.random() < self.p:
            if "image" in sample:
                sample["image"] = sample["image"].flip(-1)
            if "depth" in sample:
                sample["depth"] = sample["depth"].flip(-1)
        return sample


class ColorJitter:
    """Random color augmentation for robustness."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
    ) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if "image" not in sample:
            return sample

        img = sample["image"]

        b = 1.0 + (torch.rand(1).item() * 2 - 1) * self.brightness
        img = img * b

        c = 1.0 + (torch.rand(1).item() * 2 - 1) * self.contrast
        mean = img.mean()
        img = (img - mean) * c + mean

        sample["image"] = img.clamp(0, 1)
        return sample

"""LoRA (Low-Rank Adaptation) for efficient fine-tuning of 3D models."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Linear layer with Low-Rank Adaptation.

    Implements W' = W + BA where B is (out, rank) and A is (rank, in).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 1.0,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_lora_weights()

    def _init_lora_weights(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.linear(x)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return base_out + lora_out

    def merge_weights(self) -> None:
        """Merge LoRA weights into the base linear layer (for inference)."""
        with torch.no_grad():
            self.linear.weight += (self.lora_B @ self.lora_A) * self.scaling


class LoRAConv2d(nn.Module):
    """Conv2d layer with Low-Rank Adaptation for spatial feature extractors."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        rank: int = 8,
        alpha: float = 1.0,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False,
        )
        self.conv.weight.requires_grad = False

        self.lora_down = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)
        self.lora_up = nn.Conv2d(rank, out_channels, kernel_size=1, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.conv(x)
        lora_out = self.lora_up(self.lora_down(x)) * self.scaling
        return base_out + lora_out


def apply_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 1.0,
    target_modules: Optional[list[str]] = None,
) -> nn.Module:
    """Apply LoRA adapters to linear layers in a model.

    Args:
        model: Model to adapt.
        rank: LoRA rank.
        alpha: Scaling factor.
        target_modules: List of module name patterns to apply LoRA to.
                       If None, applies to all nn.Linear layers.

    Returns:
        Modified model with LoRA layers.
    """
    if target_modules is None:
        target_modules = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if target_modules and not any(t in name for t in target_modules):
                continue

            lora_layer = LoRALinear(
                in_features=module.in_features,
                out_features=module.out_features,
                rank=rank,
                alpha=alpha,
                bias=module.bias is not None,
            )
            lora_layer.linear.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                lora_layer.linear.bias.data.copy_(module.bias.data)

            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = dict(model.named_modules())[parts[0]]
                setattr(parent, parts[1], lora_layer)
            else:
                setattr(model, name, lora_layer)

    return model

from __future__ import annotations

import torch

from .native_support import BOUND_SUPPORTED_INPUT_RANKS


def supports_bound_input(x: torch.Tensor) -> bool:
    return x.ndim in BOUND_SUPPORTED_INPUT_RANKS


def runtime_shape_key(x: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(v) for v in x.shape)


def rows_for_bound_tensor(x: torch.Tensor, *, label: str) -> int:
    if x.ndim == 3:
        return int(x.shape[0] * x.shape[1])
    if x.ndim == 2:
        return int(x.shape[0])
    raise ValueError(f"Unsupported tensor rank for {label}: {x.ndim}")

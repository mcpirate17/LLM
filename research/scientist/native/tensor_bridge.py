from __future__ import annotations

from typing import Any

import numpy as np


def supports_host_array_bridge(*values: Any) -> bool:
    """Return True when all tensor-like inputs already live on CPU.

    The native dispatch helpers in this module convert tensors to NumPy arrays.
    Doing that for CUDA tensors forces synchronous D2H/H2D copies, which is
    counterproductive for GPU execution. Callers should fall back to the regular
    PyTorch path when this returns False.
    """
    try:
        import torch
    except Exception:  # pragma: no cover - torch is available in normal runs
        torch = None

    for value in values:
        if torch is not None and torch.is_tensor(value):
            device = getattr(value, "device", None)
            if device is not None and getattr(device, "type", None) != "cpu":
                return False
    return True


def to_native_array(value: Any) -> np.ndarray:
    """Return a contiguous float32 numpy array for native dispatch."""
    if hasattr(value, "detach"):
        tensor = value.detach()
        if getattr(tensor, "device", None) is not None and tensor.device.type != "cpu":
            tensor = tensor.cpu()
        tensor = tensor.contiguous()
        if str(tensor.dtype) != "torch.float32":
            tensor = tensor.float()
        return tensor.numpy()
    return np.ascontiguousarray(value, dtype=np.float32)


def to_native_flat_array(value: Any) -> np.ndarray:
    """Return a flattened contiguous float32 numpy view for flat native ops."""
    return to_native_array(value).reshape(-1)


def to_device_tensor(value: Any, *, reference: Any) -> Any:
    """Convert a numpy-like result back to torch when the reference is torch."""
    if not hasattr(reference, "detach"):
        return np.asarray(value, dtype=np.float32)

    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))
    device = getattr(reference, "device", None)
    if device is not None:
        tensor = tensor.to(device)
    return tensor

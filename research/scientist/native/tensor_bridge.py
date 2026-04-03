from __future__ import annotations

from typing import Any

import numpy as np


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

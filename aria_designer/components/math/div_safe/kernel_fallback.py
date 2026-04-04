"""Python fallback kernel for div_safe."""

import torch
from aria_designer.components.base import make_binary_handler

ComponentHandler = make_binary_handler(
    lambda a, b: a / (b + 1e-6 * torch.where(b >= 0, 1.0, -1.0)),
    native_op_name="div_safe",
)

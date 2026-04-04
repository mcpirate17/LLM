"""Python fallback kernel for minimum."""

import torch
from aria_designer.components.base import make_binary_handler

ComponentHandler = make_binary_handler(
    lambda a, b: torch.minimum(a, b), native_op_name="minimum"
)

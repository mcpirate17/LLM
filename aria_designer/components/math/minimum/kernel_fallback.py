"""Python fallback kernel for minimum."""

import torch
from aria_designer.runtime.fallback_templates import make_torch_binary_handler

ComponentHandler = make_torch_binary_handler(
    lambda a, b: torch.minimum(a, b), native_op_name="minimum"
)

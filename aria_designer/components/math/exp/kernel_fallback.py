"""Python fallback kernel for exp."""

import torch
from aria_designer.components.base import make_unary_handler

ComponentHandler = make_unary_handler(
    lambda x: torch.exp(torch.clamp(x, -20, 20)), native_op_name="exp"
)

"""Python fallback kernel for sqrt."""

import torch

from aria_designer.components.base import make_unary_handler

ComponentHandler = make_unary_handler(
    lambda x: torch.sqrt(torch.clamp(x, min=0.0)),
    native_op_name="sqrt",
)

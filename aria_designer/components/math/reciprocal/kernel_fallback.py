"""Python fallback kernel for reciprocal."""

import torch
from components.base import make_unary_handler

ComponentHandler = make_unary_handler(
    lambda x: torch.where(x >= 0, 1.0 / (x + 1e-8), 1.0 / (x - 1e-8)),
    native_op_name="reciprocal",
)

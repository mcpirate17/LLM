"""Python fallback kernel for cumsum (cumulative sum along sequence dim)."""

import torch
from components.base import make_unary_handler

ComponentHandler = make_unary_handler(
    lambda x: torch.cumsum(x, dim=-2), native_op_name="cumsum"
)

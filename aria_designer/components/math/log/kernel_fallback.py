"""Python fallback kernel for log."""

import torch
from components.base import make_unary_handler

ComponentHandler = make_unary_handler(
    lambda x: torch.log(torch.clamp(x.abs(), min=1e-8)), native_op_name="log"
)

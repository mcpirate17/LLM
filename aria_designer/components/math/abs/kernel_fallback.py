"""Python fallback kernel for abs."""

import torch
from components.base import make_unary_handler

ComponentHandler = make_unary_handler(lambda x: torch.abs(x), native_op_name="abs")

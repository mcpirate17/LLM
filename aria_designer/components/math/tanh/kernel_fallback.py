"""Python fallback kernel for tanh."""

import torch
from aria_designer.components.base import make_unary_handler

ComponentHandler = make_unary_handler(lambda x: torch.tanh(x), native_op_name="tanh")

"""Python fallback kernel for sin."""

import torch
from aria_designer.components.base import make_unary_handler

ComponentHandler = make_unary_handler(torch.sin, native_op_name="sin")

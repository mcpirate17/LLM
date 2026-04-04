"""Python fallback kernel for cos."""

import torch
from aria_designer.components.base import make_unary_handler

ComponentHandler = make_unary_handler(torch.cos, native_op_name="cos")

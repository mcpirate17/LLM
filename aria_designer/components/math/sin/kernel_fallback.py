"""Python fallback kernel for sin."""

import torch
from aria_designer.runtime.fallback_templates import make_torch_unary_handler

ComponentHandler = make_torch_unary_handler(torch.sin, native_op_name="sin")

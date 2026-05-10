"""Python fallback kernel for cos."""

import torch
from aria_designer.runtime.fallback_templates import make_torch_unary_handler

ComponentHandler = make_torch_unary_handler(torch.cos, native_op_name="cos")

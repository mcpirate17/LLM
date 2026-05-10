"""Python fallback kernel for abs."""

import torch
from aria_designer.runtime.fallback_templates import make_torch_unary_handler

ComponentHandler = make_torch_unary_handler(torch.abs, native_op_name="abs")

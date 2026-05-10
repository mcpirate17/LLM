"""Python fallback kernel for tanh."""

import torch
from aria_designer.runtime.fallback_templates import make_torch_unary_handler

ComponentHandler = make_torch_unary_handler(torch.tanh, native_op_name="tanh")

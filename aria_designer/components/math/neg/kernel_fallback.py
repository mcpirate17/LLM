"""Python fallback kernel for neg."""

import torch
from aria_designer.runtime.fallback_templates import make_torch_unary_handler

ComponentHandler = make_torch_unary_handler(torch.neg, native_op_name="neg")

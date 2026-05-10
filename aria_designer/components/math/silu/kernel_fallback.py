"""Python fallback kernel for silu."""

import torch.nn.functional as F
from aria_designer.runtime.fallback_templates import make_torch_unary_handler

ComponentHandler = make_torch_unary_handler(F.silu, native_op_name="silu")

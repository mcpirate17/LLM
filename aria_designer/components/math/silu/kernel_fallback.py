"""Python fallback kernel for silu."""

import torch.nn.functional as F
from aria_designer.components.base import make_unary_handler

ComponentHandler = make_unary_handler(lambda x: F.silu(x), native_op_name="silu")

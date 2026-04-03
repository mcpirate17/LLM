"""Python fallback kernel for gelu."""

import torch.nn.functional as F
from components.base import make_unary_handler

ComponentHandler = make_unary_handler(lambda x: F.gelu(x), native_op_name="gelu")

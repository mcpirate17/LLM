"""Python fallback kernel for relu."""

import torch.nn.functional as F
from aria_designer.runtime.fallback_templates import make_torch_unary_handler

ComponentHandler = make_torch_unary_handler(F.relu, native_op_name="relu")

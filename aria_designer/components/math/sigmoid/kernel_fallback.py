"""Python fallback kernel for sigmoid."""

import torch
from components.base import make_unary_handler

ComponentHandler = make_unary_handler(
    lambda x: torch.sigmoid(x), native_op_name="sigmoid"
)

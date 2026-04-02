"""Python fallback kernel for norm_last (L2 norm along last dimension)."""

import torch
from components.base import make_unary_handler

ComponentHandler = make_unary_handler(lambda x: torch.norm(x, dim=-1, keepdim=True))

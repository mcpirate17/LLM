"""Python fallback kernel for cumprod_safe (clamped cumulative product along seq dim)."""

import torch
from components.base import make_unary_handler

ComponentHandler = make_unary_handler(
    lambda x: torch.cumprod(torch.clamp(x, min=1e-8), dim=-2)
)

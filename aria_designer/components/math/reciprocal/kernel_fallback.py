"""Python fallback kernel for reciprocal."""

import torch
from components.base import make_unary_handler

ComponentHandler = make_unary_handler(
    lambda x: (
        1.0 / torch.clamp(x, min=1e-8)
        if x.mean() > 0
        else 1.0 / torch.clamp(x, max=-1e-8)
    )
)

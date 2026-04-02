"""Python fallback kernel for sign_ste (sign with straight-through estimator)."""

import torch
from components.base import make_unary_handler


def _sign_ste(x: torch.Tensor) -> torch.Tensor:
    """Forward: sign(x). Backward: straight-through (gradient passes as-is)."""
    return x + (torch.sign(x) - x).detach()


ComponentHandler = make_unary_handler(_sign_ste)

"""Python fallback kernel for exp."""

import torch
from components.base import make_unary_handler

ComponentHandler = make_unary_handler(lambda x: torch.exp(torch.clamp(x, -20, 20)))

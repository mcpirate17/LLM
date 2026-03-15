"""Python fallback kernel for neg."""
import torch
from components.base import make_unary_handler

ComponentHandler = make_unary_handler(lambda x: torch.neg(x))

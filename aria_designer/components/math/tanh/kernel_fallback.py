"""Python fallback kernel for tanh."""
import torch
from components.base import make_unary_handler

ComponentHandler = make_unary_handler(lambda x: torch.tanh(x))

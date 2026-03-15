"""Python fallback kernel for maximum."""
import torch
from components.base import make_binary_handler

ComponentHandler = make_binary_handler(lambda a, b: torch.maximum(a, b))

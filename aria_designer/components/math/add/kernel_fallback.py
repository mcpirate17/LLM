"""Python fallback kernel for add."""
import torch
from components.base import make_binary_handler

ComponentHandler = make_binary_handler(lambda a, b: a + b)

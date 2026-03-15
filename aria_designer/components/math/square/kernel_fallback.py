"""Python fallback kernel for square."""
import torch
from components.base import make_unary_handler

ComponentHandler = make_unary_handler(lambda x: x * x)

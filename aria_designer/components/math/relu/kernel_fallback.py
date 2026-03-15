"""Python fallback kernel for relu."""
import torch
import torch.nn.functional as F
from components.base import make_unary_handler

ComponentHandler = make_unary_handler(lambda x: F.relu(x))

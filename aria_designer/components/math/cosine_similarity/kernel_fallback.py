"""Python fallback kernel for cosine_similarity."""

import torch.nn.functional as F
from components.base import make_binary_handler

ComponentHandler = make_binary_handler(
    lambda a, b: F.cosine_similarity(a, b, dim=-1, eps=1e-8)
)

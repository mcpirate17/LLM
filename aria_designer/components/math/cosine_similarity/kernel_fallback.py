"""Python fallback kernel for cosine_similarity."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from components.base import SimpleBinaryOpHandler

class CosineSimilarityModule(nn.Module):
    def forward(self, a, b):
        return F.cosine_similarity(a, b, dim=-1, eps=1e-8)

class ComponentHandler(SimpleBinaryOpHandler):
    def __init__(self):
        super().__init__(
            CosineSimilarityModule,
            lambda a, b: F.cosine_similarity(a, b, dim=-1, eps=1e-8),
            native_op_name="cosine_similarity",
        )

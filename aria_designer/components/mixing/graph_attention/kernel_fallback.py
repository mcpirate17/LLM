"""Python fallback kernel for graph_attention."""

import math
import torch
import torch.nn.functional as F


class ComponentHandler:
    """Fallback handler for graph_attention: causal self-attention (no edge features in fallback)."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]  # (B, S, D)
        B, S, D = x.shape
        scale = math.sqrt(D)
        scores = torch.bmm(x, x.transpose(-1, -2)) / scale
        mask = torch.triu(
            torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores.masked_fill_(mask.unsqueeze(0), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        return {"y": torch.bmm(attn, x)}

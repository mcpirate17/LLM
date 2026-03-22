"""Python fallback kernel for local_window_attn."""

import math
import torch
import torch.nn.functional as F


class ComponentHandler:
    """Fallback handler for local_window_attn: attend only within a local window."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]  # (B, S, D)
        window = config.get("window_size", 64)
        B, S, D = x.shape
        # Simple self-attention with local window mask
        scale = math.sqrt(D)
        scores = torch.bmm(x, x.transpose(-1, -2)) / scale  # (B, S, S)
        # Build local causal mask
        rows = torch.arange(S, device=x.device).unsqueeze(1)
        cols = torch.arange(S, device=x.device).unsqueeze(0)
        mask = (cols > rows) | (rows - cols >= window)
        scores.masked_fill_(mask.unsqueeze(0), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        return {"y": torch.bmm(attn, x)}

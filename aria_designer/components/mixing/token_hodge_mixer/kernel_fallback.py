"""Python fallback kernel for token_hodge_mixer."""

import torch
import torch.nn.functional as F


class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        prev = F.pad(x[:, :-1], (0, 0, 1, 0))
        edge = x - prev
        prev_edge = F.pad(edge[:, :-1], (0, 0, 1, 0))
        boundary = edge - prev_edge
        scale = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype).view(
            1, -1, 1
        )
        return {"y": x + torch.tanh(boundary.cumsum(dim=1) / scale)}

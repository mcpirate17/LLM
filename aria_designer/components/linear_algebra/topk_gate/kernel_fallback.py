"""Python fallback kernel for topk_gate."""

import torch
import torch.nn.functional as F


class ComponentHandler:
    """Fallback handler for topk_gate: split into 2 channels, gate via softmax."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        D = x.shape[-1]
        half = D // 2
        # Simple 2-channel gating: softmax over channel scores
        scores = torch.stack(
            [x[..., :half].mean(dim=-1), x[..., half : 2 * half].mean(dim=-1)], dim=-1
        )
        gates = F.softmax(scores, dim=-1)
        g0 = gates[..., 0:1]
        g1 = gates[..., 1:2]
        y = torch.cat(
            [x[..., :half] * g0, x[..., half : 2 * half] * g1, x[..., 2 * half :]],
            dim=-1,
        )
        return {"y": y}

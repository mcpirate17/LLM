"""Python fallback kernel for sparsemax_attention."""

import torch


def _sparsemax(logits):
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    shifted = shifted.clamp(min=-20.0)
    zs = torch.sort(shifted, dim=-1, descending=True).values
    ks = torch.arange(1, logits.shape[-1] + 1, device=logits.device, dtype=logits.dtype)
    view = [1] * logits.ndim
    view[-1] = logits.shape[-1]
    ks = ks.reshape(view)
    support = 1 + ks * zs > zs.cumsum(dim=-1)
    k_z = support.sum(dim=-1, keepdim=True).clamp(min=1)
    tau = (zs.cumsum(dim=-1).gather(-1, k_z - 1) - 1) / k_z.to(logits.dtype)
    return torch.clamp(shifted - tau, min=0)


class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        _, S, D = x.shape
        scores = torch.matmul(x, x.transpose(-2, -1)) * (D**-0.5)
        mask = torch.triu(
            torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1
        )
        weights = _sparsemax(scores.masked_fill(mask, -1e9))
        return {"y": torch.matmul(weights, x)}

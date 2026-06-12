"""Python fallback kernel for sparsemax_attention."""

import torch

from aria_designer.components.base import make_causal_attention_handler


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


ComponentHandler = make_causal_attention_handler(
    lambda scores, config: _sparsemax(scores)
)

"""Python fallback kernel for linear_attention."""

import torch
import torch.nn.functional as F


class ComponentHandler:
    """Fallback handler for linear_attention: ELU kernel linear attention O(SD^2)."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]  # (B, S, D)
        # ELU+1 feature map for linear attention
        phi = F.elu(x) + 1  # (B, S, D)
        # Causal linear attention via cumulative sums
        kv = torch.cumsum(torch.einsum("bsd,bse->bsde", phi, x), dim=1)  # (B, S, D, D)
        k_sum = torch.cumsum(phi, dim=1)  # (B, S, D)
        num = torch.einsum("bsd,bsde->bse", phi, kv)  # (B, S, D)
        den = (phi * k_sum).sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return {"y": num / den}

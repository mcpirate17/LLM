"""Python fallback kernel for entmax_attention."""

import torch


def _entmax(logits, alpha=1.5):
    alpha = max(1.01, min(2.0, float(alpha)))
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    shifted = shifted.clamp(min=-20.0)
    scaled = shifted * (alpha - 1.0)
    tau_lo = scaled.min(dim=-1, keepdim=True).values - 1.0
    tau_hi = scaled.max(dim=-1, keepdim=True).values
    power = 1.0 / (alpha - 1.0)
    for _ in range(24):
        tau = (tau_lo + tau_hi) * 0.5
        probs = torch.clamp(scaled - tau, min=0).pow(power)
        too_large = probs.sum(dim=-1, keepdim=True) > 1.0
        tau_lo = torch.where(too_large, tau, tau_lo)
        tau_hi = torch.where(too_large, tau_hi, tau)
    probs = torch.clamp(scaled - tau_hi, min=0).pow(power)
    return probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)


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
        weights = _entmax(scores.masked_fill(mask, -1e9), config.get("alpha", 1.5))
        return {"y": torch.matmul(weights, x)}

"""Python fallback kernel for integral_kernel."""

import torch


class ComponentHandler:
    """Fallback handler for integral_kernel: causal exponential kernel attention."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]  # (B, S, D)
        scale = config.get("kernel_scale", 1.0)
        B, S, D = x.shape
        # Positional distance kernel with causal mask
        pos = torch.arange(S, device=x.device, dtype=x.dtype)
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()  # (S, S)
        kernel = torch.exp(-scale * dist)
        # Causal: zero out future positions
        causal = torch.tril(torch.ones(S, S, device=x.device))
        kernel = kernel * causal
        kernel = kernel / kernel.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return {"y": torch.bmm(kernel.unsqueeze(0).expand(B, -1, -1), x)}

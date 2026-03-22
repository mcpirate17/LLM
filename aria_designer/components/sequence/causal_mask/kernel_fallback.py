"""Python fallback kernel for causal_mask."""

import torch


class ComponentHandler:
    """Fallback handler for causal_mask: cumulative average along sequence dim."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        # Causal integration: cumsum / position count
        cs = torch.cumsum(x, dim=-2)
        seq_len = x.shape[-2]
        divisor = torch.arange(
            1, seq_len + 1, device=x.device, dtype=x.dtype
        ).unsqueeze(-1)
        return {"y": cs / divisor}

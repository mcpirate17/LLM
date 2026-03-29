"""Python fallback kernel for early_exit."""

import torch


class ComponentHandler:
    """Fallback handler for early_exit: confidence-gated output."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]  # (B, S, D)
        # Confidence gate: sigmoid of mean activation
        scores = x.mean(dim=-1, keepdim=True)
        gate = torch.sigmoid(scores)
        return {"y": x * gate}

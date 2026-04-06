"""Python fallback kernel for confidence_token_gate.

This is a true early-exit gate, not the progressive scaling used by
learned_token_gate.
"""

import torch


class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        threshold = float(config.get("threshold", 0.5))

        scores = x.mean(dim=-1)
        gate = torch.sigmoid(scores)
        easy_mask = (gate > threshold).to(dtype=x.dtype)
        gate_ste = easy_mask - gate.detach() + gate

        # Easy tokens are zeroed so an outer residual can cheaply recover them.
        y = x * (1.0 - gate_ste).unsqueeze(-1)
        return {"y": y}

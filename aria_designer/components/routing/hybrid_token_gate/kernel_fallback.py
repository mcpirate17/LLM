"""Python fallback kernel for hybrid_token_gate."""

import torch


class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        threshold = float(config.get("threshold", 0.5))
        scores = x.mean(dim=-1, keepdim=True)
        gate = torch.sigmoid(scores)
        return {"y": x * (gate >= threshold).to(x.dtype)}

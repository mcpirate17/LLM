"""Python fallback kernel for retention_mix."""

import torch


class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        B, S, D = x.shape
        decay = torch.linspace(0.55, 0.98, D, device=x.device, dtype=x.dtype)
        state = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(S):
            state = decay * state + x[:, t]
            outputs.append(state / (1.0 + t) ** 0.5)
        return {"y": torch.stack(outputs, dim=1)}

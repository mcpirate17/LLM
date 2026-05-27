"""Python fallback kernel for dplr_gated_delta."""

import torch


class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        B, S, D = x.shape
        prev = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        outputs = []
        decay = torch.sigmoid(x)
        for t in range(S):
            update = torch.tanh(x[:, t])
            prev = decay[:, t] * prev + (1.0 - decay[:, t]) * update
            outputs.append(prev)
        return {"y": torch.stack(outputs, dim=1)}

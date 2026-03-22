"""Python fallback kernel for split3."""

import torch


class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        scope = str(config.get("split_scope", "feature"))
        dim = 1 if scope == "token" and x.ndim >= 2 else -1
        parts = torch.tensor_split(x, 3, dim=dim)
        return {"y0": parts[0], "y1": parts[1], "y2": parts[2]}

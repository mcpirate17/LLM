"""Python fallback kernel for softmax_last."""

import torch.nn.functional as F


class ComponentHandler:
    """Fallback handler for softmax_last: softmax over last dim."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        return {"y": F.softmax(x, dim=-1)}

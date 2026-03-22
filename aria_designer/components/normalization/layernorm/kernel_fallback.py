"""Python fallback kernel for layernorm."""

import torch.nn.functional as F


class ComponentHandler:
    """Fallback handler for layernorm."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        return {"y": F.layer_norm(x, [x.shape[-1]])}

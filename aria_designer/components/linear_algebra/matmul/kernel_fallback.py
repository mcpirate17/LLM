"""Auto-generated Python fallback kernel for matmul."""

import torch.nn as nn


class ComponentHandler:
    """Fallback handler for matmul."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return nn.Identity()

    def forward(self, inputs, config):
        a = inputs["a"]
        b = inputs["b"]
        # TODO: implement matmul
        return {"y": a}

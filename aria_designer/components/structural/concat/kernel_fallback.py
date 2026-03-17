"""Auto-generated Python fallback kernel for concat."""

import torch.nn as nn


class ComponentHandler:
    """Fallback handler for concat."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return nn.Identity()

    def forward(self, inputs, config):
        a = inputs["a"]
        b = inputs["b"]
        # TODO: implement concat
        return {"y": a}

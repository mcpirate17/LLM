"""Auto-generated Python fallback kernel for outer_product."""

import torch.nn as nn


class ComponentHandler:
    """Fallback handler for outer_product."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return nn.Identity()

    def forward(self, inputs, config):
        a = inputs["a"]
        b = inputs["b"]
        # TODO: implement outer_product
        return {"y": a}

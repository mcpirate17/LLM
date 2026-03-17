"""Auto-generated Python fallback kernel for linear_proj."""

import torch.nn as nn


class ComponentHandler:
    """Fallback handler for linear_proj."""

    def validate_config(self, config):
        return []

    def build(self, config):
        # TODO: implement parameterized module
        return nn.Identity()

    def forward(self, inputs, config):
        x = inputs["x"]
        # TODO: implement linear_proj
        return {"y": x}

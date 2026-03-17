"""Auto-generated Python fallback kernel for conv1d_seq."""

import torch.nn as nn


class ComponentHandler:
    """Fallback handler for conv1d_seq."""

    def validate_config(self, config):
        return []

    def build(self, config):
        # TODO: implement parameterized module
        return nn.Identity()

    def forward(self, inputs, config):
        x = inputs["x"]
        # TODO: implement conv1d_seq
        return {"y": x}

"""Auto-generated Python fallback kernel for input."""

import torch.nn as nn


class ComponentHandler:
    """Fallback handler for input."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return nn.Identity()

    def forward(self, inputs, config):
        # Input usually doesn't have inputs from other nodes, it gets it from the 'inputs' arg of forward
        return {"y": list(inputs.values())[0] if inputs else None}

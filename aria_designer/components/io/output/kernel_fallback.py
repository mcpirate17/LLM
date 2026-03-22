"""Python fallback kernel for output (graph output passthrough)."""

import torch.nn as nn


class ComponentHandler:
    """Passthrough handler for graph output."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return nn.Identity()

    def forward(self, inputs, config):
        return {"y": list(inputs.values())[0] if inputs else None}

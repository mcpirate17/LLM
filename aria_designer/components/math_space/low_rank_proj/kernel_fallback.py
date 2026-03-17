"""Python fallback kernel for low_rank_proj.

Current behavior is an identity pass-through placeholder so workflows remain
executable in preview/eval when a native implementation is unavailable.
"""

import torch.nn as nn


class ComponentHandler:
    """Fallback handler for low_rank_proj."""

    def validate_config(self, config):
        return []

    def build(self, config):
        # Placeholder until parameterized low-rank projection is implemented.
        return nn.Identity()

    def forward(self, inputs, config):
        x = inputs["x"]
        return {"y": x}

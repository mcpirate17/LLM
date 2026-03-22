"""Python fallback kernel for route_recursion."""

import torch


class ComponentHandler:
    """Fallback handler for route_recursion."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        scores = inputs["scores"]
        if scores.dim() != 3:
            raise ValueError("route_recursion expects scores with shape [B, S, Dp]")
        max_depth = max(
            1, min(int(config.get("max_depth", scores.size(-1))), scores.size(-1))
        )
        depth = torch.argmax(scores[..., :max_depth], dim=-1)
        return {"depth": depth}

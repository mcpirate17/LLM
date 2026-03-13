"""Python fallback kernel for route_lanes."""

import torch

class ComponentHandler:
    """Fallback handler for route_lanes."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        scores = inputs["scores"]
        n_lanes = max(1, int(config.get("n_lanes", 2)))
        if scores.dim() < 2:
            raise ValueError("route_lanes expects scores with shape [B, S, L]")
        if scores.size(-1) != n_lanes:
            n_lanes = scores.size(-1)
        lane_indices = torch.argmax(scores[..., :n_lanes], dim=-1)
        return {"lane_indices": lane_indices}

"""Python fallback kernel for route_topk."""

import torch


class ComponentHandler:
    """Fallback handler for route_topk."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        scores = inputs["scores"]
        if scores.dim() != 3:
            raise ValueError("route_topk expects scores with shape [B, S, K]")
        k = max(1, min(int(config.get("k", 1)), scores.size(-1)))
        weights, indices = torch.topk(scores, k=k, dim=-1)
        return {"indices": indices, "weights": weights}

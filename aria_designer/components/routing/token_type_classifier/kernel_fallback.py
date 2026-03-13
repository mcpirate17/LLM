"""Python fallback kernel for token_type_classifier."""

import torch

class ComponentHandler:
    """Fallback handler for token_type_classifier."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        n_classes = max(1, int(config.get("n_classes", 2)))
        pooled = x.mean(dim=-1, keepdim=True)
        scores = pooled.expand(*pooled.shape[:-1], n_classes).contiguous()
        return {"scores": scores}

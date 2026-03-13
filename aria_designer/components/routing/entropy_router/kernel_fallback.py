"""Python fallback kernel for entropy_router."""

import torch

class ComponentHandler:
    """Fallback handler for entropy_router."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        scores = inputs["scores"]
        probs = torch.softmax(scores, dim=-1)
        entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=-1, keepdim=True)
        return {"routing_signal": entropy}

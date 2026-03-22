"""Python fallback kernel for matmul."""

import torch


class ComponentHandler:
    """Fallback handler for matmul: a[B,S,D] @ b[B,D,K] -> y[B,S,K]."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        a = inputs["a"]
        b = inputs["b"]
        return {"y": torch.bmm(a, b)}

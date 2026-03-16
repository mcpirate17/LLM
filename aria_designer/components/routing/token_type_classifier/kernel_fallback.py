"""Python fallback kernel for token_type_classifier."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComponentHandler:
    """Token type classifier: learned D → n_classes projection with GELU nonlinearity."""

    def __init__(self):
        self._classifier = None
        self._proj_back = None

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        D = x.shape[-1]
        n_classes = max(1, int(config.get("n_classes", 2)))

        # Lazy init with proper learned parameters
        if self._classifier is None or self._classifier.in_features != D:
            self._classifier = nn.Linear(D, n_classes, bias=False)
            self._proj_back = nn.Linear(n_classes, D, bias=False)
            nn.init.normal_(self._classifier.weight, std=0.02)
            nn.init.normal_(self._proj_back.weight, std=0.02)
            self._classifier.to(device=x.device, dtype=x.dtype)
            self._proj_back.to(device=x.device, dtype=x.dtype)

        # Classify with nonlinearity, then project back
        scores = F.gelu(self._classifier(x))  # (B, S, n_classes)
        out = self._proj_back(scores)  # (B, S, D)
        return {"scores": scores, "y": out}

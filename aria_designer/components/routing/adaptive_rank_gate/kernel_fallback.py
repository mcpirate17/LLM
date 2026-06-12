"""Python fallback kernel for adaptive_rank_gate."""

import torch

from aria_designer.components.base import LazyLowRankLanesHandler


class ComponentHandler(LazyLowRankLanesHandler):
    """Adaptive low-rank gating fallback with cached weights."""

    def forward(self, inputs, config):
        x = inputs["x"]
        self._ensure_weights(x)
        gate = torch.sigmoid(x.pow(2).mean(dim=-1, keepdim=True))
        return {"y": self.low_rank_lane(x) * (1.0 - gate) + self.dense_lane(x) * gate}

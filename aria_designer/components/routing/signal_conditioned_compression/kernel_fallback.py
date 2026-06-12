"""Python fallback kernel for signal_conditioned_compression."""

import torch

from aria_designer.components.base import LazyLowRankLanesHandler


class ComponentHandler(LazyLowRankLanesHandler):
    """Blend dense and low-rank projections from an external routing signal."""

    def forward(self, inputs, config):
        x = inputs["x"]
        routing_signal = inputs.get("routing_signal")
        if routing_signal is None:
            routing_signal = x.mean(dim=-1, keepdim=True)
        self._ensure_weights(x)
        gate = torch.sigmoid(routing_signal.to(dtype=x.dtype))
        return {"y": self.low_rank_lane(x) * (1.0 - gate) + self.dense_lane(x) * gate}

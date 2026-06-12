"""Python fallback kernel for difficulty_blend_3way."""

import torch.nn.functional as F

from aria_designer.components.base import LazyLowRankLanesHandler


class ComponentHandler(LazyLowRankLanesHandler):
    """Three-lane routing fallback: skip, low-rank, and dense paths."""

    router_lanes = 3

    def forward(self, inputs, config):
        x = inputs["x"]
        self._ensure_weights(x)
        lane_weights = F.softmax(x @ self._router, dim=-1)
        y = (
            x * lane_weights[..., 0:1]
            + self.low_rank_lane(x) * lane_weights[..., 1:2]
            + F.gelu(self.dense_lane(x)) * lane_weights[..., 2:3]
        )
        return {"y": y}

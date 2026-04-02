"""Python fallback kernel for score_depth_blend."""

import torch
import torch.nn.functional as F


class ComponentHandler:
    """Depth-aware iterative transform using provided recursion scores."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        max_depth = max(1, int(config.get("max_depth", 3)))
        scores = inputs.get("scores")
        if scores is None:
            base = x.pow(2).mean(dim=-1, keepdim=True)
            scores = torch.cat([base / float(i + 1) for i in range(max_depth)], dim=-1)
        else:
            scores = scores[..., :max_depth]
        weights = F.softmax(scores, dim=-1)
        states = []
        z = x
        for depth in range(max_depth):
            z = torch.tanh(z + (depth + 1) * 0.1 * x)
            states.append(z)
        stacked = torch.stack(states, dim=-2)
        y = (stacked * weights.unsqueeze(-1)).sum(dim=-2)
        return {"y": y}

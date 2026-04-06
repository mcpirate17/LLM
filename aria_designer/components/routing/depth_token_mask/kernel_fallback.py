"""Fallback kernel for depth_token_mask.

Align this path with the research compiler op:
- causal, position-based token dropping
- soft score gate for gradient flow
- no hidden learned state inside the handler
"""

import torch


class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        B, S, D = x.shape
        capacity = float(config.get("capacity_factor", 0.75))

        # Match research.synthesis.compiler_ops_routing._op_depth_token_mask:
        # deterministic causal keep/drop structure plus a soft score gate.
        scores = x.mean(dim=-1)
        stride = max(1, int(1.0 / max(1.0 - capacity, 0.01)))
        pos = torch.arange(S, device=x.device)
        keep_mask = ((pos % stride) != (stride - 1)).to(dtype=x.dtype)
        keep_mask = keep_mask.unsqueeze(0).expand(B, -1)

        cumsum = scores.cumsum(dim=-1)
        counts = torch.arange(1, S + 1, device=x.device, dtype=x.dtype)
        causal_mean = cumsum / counts
        soft_gate = torch.sigmoid(4.0 * (scores - causal_mean))
        gate = soft_gate * keep_mask

        y = x * gate.unsqueeze(-1)
        return {"y": y}

"""Kernel handler for relu_gate_routing — ReLU-gated MoE with learned experts."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComponentHandler:
    def __init__(self):
        self._gate = None
        self._experts = None

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        D = x.shape[-1]
        n_experts = int(config.get("n_experts", 4))

        if self._gate is None or self._gate.in_features != D:
            self._gate = nn.Linear(D, n_experts, bias=False)
            self._experts = nn.ModuleList(
                [nn.Linear(D, D, bias=False) for _ in range(n_experts)]
            )
            nn.init.normal_(self._gate.weight, std=0.02)
            for expert in self._experts:
                nn.init.normal_(expert.weight, std=0.02)
            self._gate.to(device=x.device, dtype=x.dtype)
            self._experts.to(device=x.device, dtype=x.dtype)

        # ReLU gate → sparse activation → normalize
        gate_logits = F.relu(self._gate(x))  # (B, S, n_experts)
        gate_sum = gate_logits.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        gate_weights = gate_logits / gate_sum  # (B, S, n_experts)

        y = torch.zeros_like(x)
        for i, expert in enumerate(self._experts):
            y = y + gate_weights[..., i : i + 1] * expert(x)
        return {"y": y}

"""Python fallback kernel for gated_delta."""

import torch
import torch.nn as nn


class ComponentHandler:
    """Gated delta rule: linear recurrence with decay + update gates."""

    def __init__(self):
        self._projs = None

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        B, S, D = x.shape

        if self._projs is None or self._projs["q"].in_features != D:
            self._projs = {
                name: nn.Linear(D, D, bias=False).to(device=x.device, dtype=x.dtype)
                for name in ("q", "k", "v", "o", "alpha", "beta")
            }

        q = self._projs["q"](x)
        k = self._projs["k"](x)
        v = self._projs["v"](x)
        alpha = torch.sigmoid(self._projs["alpha"](x))
        beta = torch.sigmoid(self._projs["beta"](x))

        h = torch.zeros(B, D, D, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(S):
            vk = v[:, t, :].unsqueeze(-1) * k[:, t, :].unsqueeze(-2)
            h = alpha[:, t, :].unsqueeze(-1) * h + beta[:, t, :].unsqueeze(-1) * (
                vk - h
            )
            out_t = (q[:, t, :].unsqueeze(-2) @ h).squeeze(-2)
            outputs.append(out_t)

        return {"y": self._projs["o"](torch.stack(outputs, dim=1))}

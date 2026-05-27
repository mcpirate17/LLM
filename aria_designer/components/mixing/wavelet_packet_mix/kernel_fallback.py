"""Python fallback kernel for wavelet_packet_mix."""

import torch
import torch.nn.functional as F


class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        levels = max(1, min(4, int(config.get("levels", 2))))
        low = x
        high_sum = torch.zeros_like(x)
        inv_sqrt2 = 2.0**-0.5
        for _ in range(levels):
            prev = F.pad(low[:, :-1], (0, 0, 1, 0))
            high = (low - prev) * inv_sqrt2
            low = (low + prev) * inv_sqrt2
            high_sum = high_sum + high
        return {"y": 0.5 * low + 0.5 * high_sum / float(levels)}

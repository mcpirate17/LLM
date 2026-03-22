"""Python fallback for synthetic_data_source."""

import math
import torch


class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        b = int(config.get("batch_size", 2))
        s = int(config.get("seq_len", 64))
        d = int(config.get("feature_dim", 64))
        pattern = str(config.get("pattern", "sine"))
        amp = float(config.get("amplitude", 1.0))
        freq = float(config.get("frequency", 1.0))
        phase = float(config.get("phase", 0.0))
        impulse_index = int(config.get("impulse_index", 0))

        t = torch.arange(s, dtype=torch.float32)
        if pattern == "sawtooth":
            base = ((t * freq / max(float(s), 1.0) + phase) % 1.0) * 2.0 - 1.0
        elif pattern == "impulse":
            base = torch.zeros(s, dtype=torch.float32)
            base[max(0, min(s - 1, impulse_index))] = 1.0
        elif pattern == "checkerboard":
            base = ((torch.arange(s) % 2) * 2 - 1).to(torch.float32)
        else:
            base = torch.sin(2.0 * math.pi * freq * t / max(float(s), 1.0) + phase)

        base = amp * base
        data = base.view(1, s, 1).repeat(b, 1, d)
        return {"data": data}

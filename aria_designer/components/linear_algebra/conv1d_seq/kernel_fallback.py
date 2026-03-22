"""Python fallback kernel for conv1d_seq."""

import torch
import torch.nn.functional as F


class ComponentHandler:
    """Fallback handler for conv1d_seq: depthwise causal conv1d."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]  # (B, S, D)
        kernel_size = config.get("kernel_size", 4)
        B, S, D = x.shape
        # Depthwise causal conv: pad left, no pad right
        x_t = x.transpose(1, 2)  # (B, D, S)
        x_pad = F.pad(x_t, (kernel_size - 1, 0))
        # Unit impulse kernel for preview (identity convolution)
        weight = torch.zeros(D, 1, kernel_size, device=x.device, dtype=x.dtype)
        weight[:, 0, -1] = 1.0  # last position = identity
        y = F.conv1d(x_pad, weight, groups=D)
        return {"y": y.transpose(1, 2)}

"""Python fallback kernel for conv_only."""

import torch
import torch.nn.functional as F


class ComponentHandler:
    """Fallback handler for conv_only: depthwise causal conv1d."""

    def __init__(self):
        self._weight = None
        self._weight_key = None  # (D, kernel_size, device, dtype)

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]  # (B, S, D)
        kernel_size = config.get("kernel_size", 4)
        B, S, D = x.shape

        # Cache identity conv weight by (D, kernel_size, device, dtype)
        key = (D, kernel_size, x.device, x.dtype)
        if self._weight_key != key:
            weight = torch.zeros(D, 1, kernel_size, device=x.device, dtype=x.dtype)
            weight[:, 0, -1] = 1.0
            self._weight = weight
            self._weight_key = key

        x_t = x.transpose(1, 2)  # (B, D, S)
        x_pad = F.pad(x_t, (kernel_size - 1, 0))
        y = F.conv1d(x_pad, self._weight, groups=D)
        return {"y": y.transpose(1, 2)}

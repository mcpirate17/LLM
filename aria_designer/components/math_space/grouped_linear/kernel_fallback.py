"""Kernel handler for grouped_linear — delegates to research.mathspaces.compression."""

import torch
import torch.nn as nn
from aria_designer.components.base import NativeComponentHandler


class ComponentHandler(NativeComponentHandler):
    native_op_name = "grouped_linear"

    def _get_native_args(self, inputs, config):
        x = inputs["x"].detach().contiguous().float()
        return (x,)

    def _fallback(self, inputs, config):
        from research.mathspaces.compression import execute_grouped_linear

        x = inputs["x"]
        D = x.shape[-1]
        g = config.get("n_groups", 4)
        group_dim = D // g
        stub = nn.Module()
        stub.n_groups = g
        stub.weight = nn.Parameter(
            torch.randn(g, group_dim, group_dim, device=x.device, dtype=x.dtype)
            * (group_dim**-0.5)
        )
        return {"y": execute_grouped_linear(stub, x)}

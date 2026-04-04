"""Kernel handler for rotor_transform — delegates to research.mathspaces.clifford."""

import torch
import torch.nn as nn
from aria_designer.components.base import NativeComponentHandler


class ComponentHandler(NativeComponentHandler):
    native_op_name = "rotor_transform"

    def _get_native_args(self, inputs, config):
        x = inputs["x"].detach().contiguous().float()
        return (x,)

    def _fallback(self, inputs, config):
        from research.mathspaces.clifford import execute_rotor_transform, N_BASIS

        x = inputs["x"]
        stub = nn.Module()
        stub.weight = nn.Parameter(torch.zeros(N_BASIS, device=x.device, dtype=x.dtype))
        stub.weight.data[0] = 1.0  # identity rotor
        return {"y": execute_rotor_transform(stub, x)}

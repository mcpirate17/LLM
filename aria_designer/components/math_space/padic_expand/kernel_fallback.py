"""Kernel handler for padic_expand — dispatches to aria_core.padic_expand_f32."""
import torch
import torch.nn as nn
from components.base import NativeComponentHandler

class ComponentHandler(NativeComponentHandler):
    native_op_name = "padic_expand"

    def _get_native_args(self, inputs, config):
        x = inputs["x"].detach().contiguous().float()
        p = config.get("p", 2)
        n_digits = config.get("n_digits", 4)
        return (x, p, n_digits)

    def _fallback(self, inputs, config):
        x = inputs["x"]
        p = config.get("p", 2)
        n_digits = config.get("n_digits", 4)
        digits = []
        val = x.abs()
        for _ in range(n_digits):
            digits.append(val % p)
            val = val.div(p, rounding_mode='floor')
        return {"y": torch.stack(digits, dim=-1).reshape(*x.shape[:-1], -1)}

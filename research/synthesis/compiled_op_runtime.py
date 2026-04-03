from __future__ import annotations

import torch


class CompiledOpRuntimeMixin:
    def _cast_params_to(self, dtype: torch.dtype) -> None:
        if getattr(self, "_last_cast_dtype", None) == dtype:
            return
        for param in self._parameters.values():
            if param is not None and param.dtype != dtype:
                param.data = param.data.to(dtype)
        for child in self._modules.values():
            if isinstance(child, torch.nn.ParameterList):
                for param in child:
                    if param.dtype != dtype:
                        param.data = param.data.to(dtype)
        self._last_cast_dtype = dtype

    def _record_op_timing(self, elapsed: float) -> None:
        timing = getattr(self, "op_timing", None)
        if timing is None:
            timing = {"calls": 0, "total_us": 0.0, "max_us": 0.0}
            object.__setattr__(self, "op_timing", timing)
        elapsed_us = elapsed * 1e6
        timing["calls"] += 1
        timing["total_us"] += elapsed_us
        if elapsed_us > timing["max_us"]:
            timing["max_us"] = elapsed_us

"""aria_core — Unified high-performance kernel library for Aria."""
import warnings
import torch  # ensure torch libs are loaded first

_C_AVAILABLE = False
_C_IMPORT_ERROR = None
try:
    # Canonical extension location from setup.py: aria_core._C
    from ._C import *  # noqa: F401,F403
    _C_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as exc_primary:
    try:
        # Backward-compat fallback for historical package layout.
        from .aria_core._C import *  # noqa: F401,F403
        _C_AVAILABLE = True
    except (ImportError, ModuleNotFoundError) as exc_fallback:
        _C_IMPORT_ERROR = exc_fallback
        warnings.warn(
            "aria_core native extension unavailable; import succeeded in fallback mode. "
            "Build/install aria_core._C for native kernels.",
            RuntimeWarning,
            stacklevel=2,
        )


def __getattr__(name: str):
    if not _C_AVAILABLE:
        if name == "relu_f32":
            return lambda x: torch.relu(x)
        if name == "gelu_f32":
            return lambda x: torch.nn.functional.gelu(x)
        if name == "add_f32":
            return lambda a, b: a + b
        if name == "sub_f32":
            return lambda a, b: a - b
        if name == "mul_f32":
            return lambda a, b: a * b
        if name == "matmul_f32":
            return lambda a, b: a @ b
        if name == "linear_f32":
            return lambda x, w, b=None: torch.nn.functional.linear(x, w, b)
        if name == "softmax_f32":
            return lambda x: torch.nn.functional.softmax(x, dim=-1)
        if name == "rmsnorm_f32":
            def _rmsnorm_f32(x, w, eps=1e-5):
                rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
                return (x / rms) * w

            return _rmsnorm_f32
        raise ModuleNotFoundError(
            "aria_core._C is not available; native kernels are not loaded. "
            "Install/build aria_core extension before using kernel symbols."
        ) from _C_IMPORT_ERROR
    raise AttributeError(name)

__version__ = "0.1.0"

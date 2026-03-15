"""aria_core.aria_core — inner package for backward compat."""
import warnings
import torch  # noqa: F401 — ensure torch libs loaded first

_C_AVAILABLE = False
_C_IMPORT_ERROR = None
try:
    from ._C import *  # noqa: F401,F403
    _C_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as exc:
    _C_IMPORT_ERROR = exc
    warnings.warn(
        "aria_core.aria_core._C is unavailable; using import-safe mode only.",
        RuntimeWarning,
        stacklevel=2,
    )

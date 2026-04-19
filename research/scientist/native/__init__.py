from __future__ import annotations

import importlib

_SUBMODULES = {
    "core",
    "abi",
    "telemetry",
    "guardrails",
    "dispatch",
    "profiling",
    "designer",
    "autograd",
    "compiler",
    "intelligent_router",
}

__all__ = sorted(_SUBMODULES)


def __getattr__(name: str):
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

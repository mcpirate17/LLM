from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any


_NATIVE_RUNTIME_CANDIDATES = (
    Path(__file__).resolve().parent
    / "runtime"
    / "native"
    / "build"
    / "libaria_native_runtime.so",
    Path(__file__).resolve().parent
    / "runtime"
    / "native"
    / "build_current"
    / "libaria_native_runtime.so",
)


def load_native_runtime_lib(required_symbols: tuple[str, ...], logger: Any) -> Any:
    for path in _NATIVE_RUNTIME_CANDIDATES:
        if not path.exists():
            continue
        try:
            candidate = ctypes.CDLL(
                str(path), mode=os.RTLD_LOCAL | getattr(os, "RTLD_LAZY", 1)
            )
        except OSError as exc:
            logger.debug("Failed to load native runtime at %s: %s", path, exc)
            continue
        if all(hasattr(candidate, name) for name in required_symbols):
            return candidate
    return None

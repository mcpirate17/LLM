"""Shared native-extension bootstrap for both aria_core package entrypoints."""

from __future__ import annotations

import importlib

import torch  # ensure torch libs are loaded first  # noqa: F401


def load_native_extension(namespace: dict) -> None:
    try:
        module = importlib.import_module(f"{namespace['__package__']}._C")
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "aria_core native extension unavailable; build/install aria_core._C "
            "before importing aria_core."
        ) from exc

    namespace.update(
        {
            name: value
            for name, value in module.__dict__.items()
            if not name.startswith("__")
        }
    )

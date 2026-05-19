"""Shared optional dependency probes for integration-style tests."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


def import_module(dotted_path: str):
    """Import a submodule without triggering unrelated helper logic."""
    return importlib.import_module(dotted_path)


@dataclass(frozen=True)
class IntegrationDependencies:
    torch: Any
    nn: Any
    lab_notebook: Any
    has_torch: bool
    has_flask: bool
    has_notebook: bool
    has_persona: bool
    has_prompts: bool
    has_context: bool


def probe_integration_dependencies(*, need_nn: bool = False) -> IntegrationDependencies:
    try:
        import torch

        nn = None
        if need_nn:
            import torch.nn as nn

        has_torch = True
    except ImportError:
        torch = None
        nn = None
        has_torch = False

    try:
        has_flask = True
    except ImportError:
        has_flask = False

    try:
        from research.scientist.notebook import LabNotebook  # noqa: F401

        has_notebook = True
    except Exception as exc:  # noqa: BLE001
        has_notebook = False
        LabNotebook = None
        print(f"Notebook import failed: {exc}")

    try:
        has_persona = True
    except Exception as exc:  # noqa: BLE001
        has_persona = False
        print(f"Persona import failed: {exc}")

    try:
        import research.scientist.llm.prompts as _prompts_mod  # noqa: F401

        has_prompts = True
    except Exception as exc:  # noqa: BLE001
        has_prompts = False
        print(f"Prompts import failed: {exc}")

    try:
        import research.scientist.llm.context as _context_mod  # noqa: F401

        has_context = True
    except Exception as exc:  # noqa: BLE001
        has_context = False
        print(f"Context import failed: {exc}")

    return IntegrationDependencies(
        torch=torch,
        nn=nn,
        lab_notebook=LabNotebook,
        has_torch=has_torch,
        has_flask=has_flask,
        has_notebook=has_notebook,
        has_persona=has_persona,
        has_prompts=has_prompts,
        has_context=has_context,
    )

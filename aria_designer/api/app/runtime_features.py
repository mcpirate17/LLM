from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ARIA_DESIGNER_ROOT = _PROJECT_ROOT / "aria_designer"
_ARIA_CORE_ROOT = _PROJECT_ROOT / "aria_core"
_RESEARCH_ROOT = _PROJECT_ROOT / "research"


def _candidates(module: str):
    """Yield the import paths to try for a runtime feature module.

    Tries the bare path and an ``aria_designer.`` prefix so the same lookup
    works whether the package is installed or run from a checkout.
    """
    yield module
    if module.startswith("runtime."):
        yield f"aria_designer.{module}"


def _spec_exists(module: str) -> bool:
    """True if *module* is importable, WITHOUT importing it (no torch load)."""
    for candidate in _candidates(module):
        try:
            if importlib.util.find_spec(candidate) is not None:
                return True
        except (ImportError, ValueError, ModuleNotFoundError):
            continue
    return False


# Capability flags resolved via find_spec — cheap, importing this module no longer
# pulls torch (runtime.dispatch) or the research stack (runtime.bridge). The heavy
# modules load only when a callable below is first *accessed* (PEP 562 __getattr__),
# i.e. inside an eval/compile/import request handler — never at server startup.
HAS_BRIDGE: bool = _spec_exists("runtime.bridge")
HAS_PROFILER: bool = _spec_exists("runtime.profiler")
HAS_IMPORTER: bool = _spec_exists("runtime.importer")
HAS_CONSTRAINTS: bool = _spec_exists("runtime.constraints")
HAS_SUBGRAPH: bool = _spec_exists("runtime.subgraph")

# Public callable / constant name -> (module, attribute). Resolved lazily.
_LAZY: dict[str, tuple[str, str]] = {
    "KernelDispatcher": ("runtime.dispatch", "KernelDispatcher"),
    "runtime_compile": ("runtime.compiler", "compile_workflow"),
    "find_unsupported_edge_dtype_pairings": (
        "runtime.port_dtypes",
        "find_unsupported_edge_dtype_pairings",
    ),
    "export_onnx": ("runtime.export", "export_onnx"),
    "bridge_evaluate": ("runtime.bridge", "evaluate_workflow"),
    "bridge_validate": ("runtime.bridge", "validate_workflow_graph"),
    "bridge_estimate": ("runtime.bridge", "estimate_performance"),
    "bridge_list_primitives": ("runtime.bridge", "list_available_primitives"),
    "bridge_analyze_compression": ("runtime.bridge", "analyze_compression"),
    "bridge_analyze_routing": ("runtime.bridge", "bridge_analyze_routing"),
    "bridge_component_capability": (
        "runtime.bridge",
        "get_component_execution_capability",
    ),
    "bridge_profile": ("runtime.profiler", "profile_workflow"),
    "import_survivors": ("runtime.importer", "import_survivors"),
    "import_single": ("runtime.importer", "import_single"),
    "graph_to_workflow": ("runtime.importer", "graph_to_workflow"),
    "check_compatibility": ("runtime.constraints", "check_compatibility"),
    "compute_palette_constraints": (
        "runtime.constraints",
        "compute_palette_constraints",
    ),
    "extract_block": ("runtime.subgraph", "extract_block"),
    "expand_block": ("runtime.subgraph", "expand_block"),
    "list_builtin_blocks": ("runtime.subgraph", "list_builtin_blocks"),
    "BUILTIN_BLOCKS": ("runtime.subgraph", "BUILTIN_BLOCKS"),
}


def _resolve(module: str, attr: str):
    """Import *module* and return *attr*, or ``None`` if the module is absent.

    Mirrors the old ``_optional_import`` contract: a missing optional runtime
    module yields ``None`` (callers gate on the matching ``HAS_*`` flag); an
    importable module that lacks the attribute still raises, loudly.
    """
    for candidate in _candidates(module):
        try:
            mod = importlib.import_module(candidate)
        except ImportError:
            continue
        return getattr(mod, attr)
    return None


def __getattr__(name: str):
    spec = _LAZY.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return _resolve(*spec)


if TYPE_CHECKING:  # give static analysis the symbol names without importing torch
    from collections.abc import Callable

    KernelDispatcher: type
    runtime_compile: Callable[..., Any]
    find_unsupported_edge_dtype_pairings: Callable[..., Any]
    export_onnx: Callable[..., Any]
    bridge_evaluate: Callable[..., Any]
    bridge_validate: Callable[..., Any]
    bridge_estimate: Callable[..., Any]
    bridge_list_primitives: Callable[..., Any]
    bridge_analyze_compression: Callable[..., Any]
    bridge_analyze_routing: Callable[..., Any]
    bridge_component_capability: Callable[..., Any]
    bridge_profile: Callable[..., Any]
    import_survivors: Callable[..., Any]
    import_single: Callable[..., Any]
    graph_to_workflow: Callable[..., Any]
    check_compatibility: Callable[..., Any]
    compute_palette_constraints: Callable[..., Any]
    extract_block: Callable[..., Any]
    expand_block: Callable[..., Any]
    list_builtin_blocks: Callable[..., Any]
    BUILTIN_BLOCKS: dict[str, Any]

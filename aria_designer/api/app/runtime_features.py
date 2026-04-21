from __future__ import annotations

from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ARIA_DESIGNER_ROOT = _PROJECT_ROOT / "aria_designer"
_ARIA_CORE_ROOT = _PROJECT_ROOT / "aria_core"
_RESEARCH_ROOT = _PROJECT_ROOT / "research"


def _optional_import(
    module: str,
    names: list[str],
    *,
    aliases: dict[str, str] | None = None,
) -> tuple[Any, ...]:
    """Import *names* from *module*, returning ``None`` for each on ImportError.

    Tries both the bare module path and an ``aria_designer.`` prefix so the
    same call works whether the package is installed or run from a checkout.
    """
    aliases = aliases or {}
    for candidate in (
        module,
        f"aria_designer.{module}" if module.startswith("runtime.") else None,
    ):
        if not candidate:
            continue
        try:
            mod = __import__(candidate, fromlist=names)
            return tuple(getattr(mod, aliases.get(name, name)) for name in names)
        except ImportError:
            continue
    return tuple(None for _ in names)


(KernelDispatcher, runtime_compile, find_unsupported_edge_dtype_pairings) = (
    _optional_import("runtime.dispatch", ["KernelDispatcher"])
    + _optional_import("runtime.compiler", ["compile_workflow"])
    + _optional_import("runtime.port_dtypes", ["find_unsupported_edge_dtype_pairings"])
)

(export_onnx,) = _optional_import("runtime.export", ["export_onnx"])

(
    bridge_evaluate,
    bridge_validate,
    bridge_estimate,
    bridge_list_primitives,
    bridge_analyze_compression,
    bridge_analyze_routing,
    bridge_component_capability,
) = _optional_import(
    "runtime.bridge",
    [
        "evaluate_workflow",
        "validate_workflow_graph",
        "estimate_performance",
        "list_available_primitives",
        "analyze_compression",
        "bridge_analyze_routing",
        "get_component_execution_capability",
    ],
)
HAS_BRIDGE: bool = bridge_evaluate is not None

(bridge_profile,) = _optional_import("runtime.profiler", ["profile_workflow"])
HAS_PROFILER: bool = bridge_profile is not None

(import_survivors, import_single, graph_to_workflow) = _optional_import(
    "runtime.importer", ["import_survivors", "import_single", "graph_to_workflow"]
)
HAS_IMPORTER: bool = import_survivors is not None

(check_compatibility, compute_palette_constraints) = _optional_import(
    "runtime.constraints", ["check_compatibility", "compute_palette_constraints"]
)
HAS_CONSTRAINTS: bool = check_compatibility is not None

(extract_block, expand_block, list_builtin_blocks, BUILTIN_BLOCKS) = _optional_import(
    "runtime.subgraph",
    ["extract_block", "expand_block", "list_builtin_blocks", "BUILTIN_BLOCKS"],
)
HAS_SUBGRAPH: bool = extract_block is not None

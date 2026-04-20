"""Program synthesis public API.

Keep package import cheap. Runtime-heavy modules such as the compiler should
only load when their symbols are actually requested.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "PrimitiveOp",
    "PRIMITIVE_REGISTRY",
    "get_primitive",
    "list_primitives",
    "OpNode",
    "ComputationGraph",
    "GrammarConfig",
    "generate_layer_graph",
    "compile_graph",
    "CompiledLayer",
    "SynthesizedModel",
    "validate_graph",
    "ValidationResult",
    "graph_to_json",
    "graph_from_json",
]

_EXPORTS = {
    "PrimitiveOp": ("primitives", "PrimitiveOp"),
    "PRIMITIVE_REGISTRY": ("primitives", "PRIMITIVE_REGISTRY"),
    "get_primitive": ("primitives", "get_primitive"),
    "list_primitives": ("primitives", "list_primitives"),
    "OpNode": ("graph", "OpNode"),
    "ComputationGraph": ("graph", "ComputationGraph"),
    "GrammarConfig": ("grammar", "GrammarConfig"),
    "generate_layer_graph": ("grammar", "generate_layer_graph"),
    "compile_graph": ("compiler", "compile_graph"),
    "CompiledLayer": ("compiler", "CompiledLayer"),
    "SynthesizedModel": ("compiler", "SynthesizedModel"),
    "validate_graph": ("validator", "validate_graph"),
    "ValidationResult": ("validator", "ValidationResult"),
    "graph_to_json": ("serializer", "graph_to_json"),
    "graph_from_json": ("serializer", "graph_from_json"),
}


def __getattr__(name: str):
    export = _EXPORTS.get(name)
    if export is not None:
        module_name, attr_name = export
        module = import_module(f".{module_name}", __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    # Fall back to submodule lookup so `from research.synthesis import <submodule>`
    # and `from . import <submodule>` keep working alongside the lazy _EXPORTS map.
    try:
        module = import_module(f".{name}", __name__)
    except ImportError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    globals()[name] = module
    return module

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from . import dispatch as native_dispatch


@dataclass(frozen=True, slots=True)
class GraphNativeCapability:
    all_ops: frozenset[str]
    supported_ops: frozenset[str]
    unsupported_ops: frozenset[str]
    scheduler_supported_ops: frozenset[str]
    scheduler_unsupported_ops: frozenset[str]
    native_coverage: float
    refusal_reason: str | None

    @property
    def all_native(self) -> bool:
        return self.refusal_reason is None


def probe_supported_native_ops(graph: Any, *, native_lib: Any = None) -> frozenset[str]:
    cached = getattr(graph, "_native_supported_ops_cache", None)
    if cached is not None and native_lib is None:
        return frozenset(cached)

    op_support = native_dispatch._check_native_op_support(
        [graph], native_lib=native_lib
    )
    supported = frozenset(
        op
        for op in (op_support.get("supported") or ())
        if op not in native_dispatch.NATIVE_STRUCTURAL_OPS
    )
    supported = _augment_composite_supported_ops(graph, supported)
    if native_lib is None:
        setattr(graph, "_native_supported_ops_cache", supported)
    return supported


def _augment_composite_supported_ops(
    graph: Any,
    supported_ops: frozenset[str],
) -> frozenset[str]:
    nodes = getattr(graph, "nodes", None)
    if not isinstance(nodes, dict):
        return supported_ops

    graph_ops = {
        getattr(node, "op_name", "")
        for node in nodes.values()
        if not getattr(node, "is_input", False)
    }
    augmented = set(supported_ops)
    if "conv_only" in graph_ops and {"conv1d_seq", "linear"}.issubset(augmented):
        augmented.add("conv_only")
    return frozenset(augmented)


def classify_graph_native_capability(
    graph: Any,
    supported_ops: Iterable[str],
) -> GraphNativeCapability:
    nodes = getattr(graph, "nodes", None)
    if not isinstance(nodes, dict):
        return GraphNativeCapability(
            all_ops=frozenset(),
            supported_ops=frozenset(),
            unsupported_ops=frozenset(),
            scheduler_supported_ops=frozenset(native_dispatch.NATIVE_STRUCTURAL_OPS),
            scheduler_unsupported_ops=frozenset(),
            native_coverage=0.0,
            refusal_reason="invalid_graph",
        )

    all_ops = frozenset(
        getattr(node, "op_name", "")
        for node in nodes.values()
        if not getattr(node, "is_input", False)
    )
    supported = frozenset(
        set(supported_ops) | set(native_dispatch.NATIVE_STRUCTURAL_OPS)
    )
    scheduler_supported = frozenset(
        native_dispatch.scheduler_compatible_ops(set(supported))
    )
    unsupported = frozenset(op for op in all_ops if op not in supported)
    scheduler_unsupported = frozenset(
        op for op in all_ops if op not in scheduler_supported
    )

    refusal_reason: str | None = None
    if unsupported:
        refusal_reason = "graph_not_fully_native"
    elif scheduler_unsupported:
        refusal_reason = "scheduler_incompatible_ops"

    native_coverage = 1.0
    if all_ops:
        native_coverage = float(len(all_ops - unsupported)) / float(len(all_ops))

    return GraphNativeCapability(
        all_ops=all_ops,
        supported_ops=supported,
        unsupported_ops=unsupported,
        scheduler_supported_ops=scheduler_supported,
        scheduler_unsupported_ops=scheduler_unsupported,
        native_coverage=native_coverage,
        refusal_reason=refusal_reason,
    )

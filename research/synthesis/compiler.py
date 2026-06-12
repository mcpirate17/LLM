"""
Computation Graph Compiler

Compiles a ComputationGraph into a live PyTorch nn.Module.
Each OpNode becomes a concrete tensor operation, with learnable
parameters allocated for parameterized ops.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

from .graph import ComputationGraph
from . import native_compile
from .compiler_registry import OP_DISPATCH, load_split_op_modules
from .primitives import load_primitives_from_designer

logger = logging.getLogger(__name__)


try:
    from . import kernels  # noqa: F401 — side-effect import sets HAS_KERNELS

    HAS_KERNELS = True
except ImportError:
    HAS_KERNELS = False

from research.defaults import VOCAB_SIZE, VALIDATION_SEQ_LEN

load_split_op_modules()

from .compiled_model import CompiledLayer, SynthesizedModel
from .compiled_op import CompiledOp
from .compiler_ops_sequence import _op_gated_delta, _parallel_associative_scan

_OP_DISPATCH = OP_DISPATCH

__all__ = [
    "CompiledLayer",
    "CompiledOp",
    "SynthesizedModel",
    "_OP_DISPATCH",
    "_op_gated_delta",
    "_parallel_associative_scan",
    "compile_graph",
    "compile_model",
    "torch",
]


def compile_graph(
    graph: ComputationGraph, use_ir: bool = True, executor: str = "default"
) -> nn.Module:
    """Compile a graph to a PyTorch module.

    Args:
        graph: The computation graph to compile.
        use_ir: If True, prefer the native/IR fast path. If False, use the
            standard CompiledLayer fallback.
        executor: ``default`` resolves to native subgraph dispatch with
            IRExecutorV2 fallback. Use ``compiled`` to force CompiledLayer.
    """
    from .graph_validator import annotate_kv_cacheable

    annotate_kv_cacheable(graph)
    return _compile_layer_module(
        graph, prefer_fast_path=use_ir, executor_variant=executor
    )


def compile_model(
    layer_graphs: List[ComputationGraph],
    vocab_size: int = VOCAB_SIZE,
    max_seq_len: int = VALIDATION_SEQ_LEN,
    use_ir: bool = True,
    executor: str = "default",
) -> SynthesizedModel:
    if not layer_graphs:
        raise ValueError("Empty layer_graphs list")
    model = SynthesizedModel(
        layer_graphs, vocab_size, layer_graphs[0].model_dim, max_seq_len
    )
    model.layers = nn.ModuleList(
        [
            _compile_layer_module(g, prefer_fast_path=use_ir, executor_variant=executor)
            for g in layer_graphs
        ]
    )
    return model


def _resolve_executor_variant(executor_variant: str) -> str:
    normalized = str(executor_variant or "default").strip().lower()
    if normalized in {"default", "auto", "ir_v2", "v2", "native_ir_v2"}:
        return "ir_v2"
    if normalized in {
        "compiled",
        "compiled_layer",
        "eager",
        "standard",
        "torch",
        "ir_v1",
        "v1",
        "legacy",
    }:
        return "compiled"
    raise ValueError(f"Unknown executor variant: {executor_variant}")


def _compile_layer_module(
    graph: ComputationGraph,
    *,
    prefer_fast_path: bool = True,
    executor_variant: str = "default",
) -> nn.Module:
    executor_variant = _resolve_executor_variant(executor_variant)
    if not prefer_fast_path or executor_variant == "compiled":
        layer = CompiledLayer(graph)
        native_compile.attach_partial_native_wrapper(layer, graph)
        return layer

    native_layer = native_compile.try_compile_native_subgraph_layer(
        CompiledLayer, graph
    )
    if native_layer is not None:
        return native_layer

    from .ir_executor_v2 import IRExecutorV2

    layer = IRExecutorV2(graph.lower_to_ir(), source_graph=graph)
    native_compile.attach_partial_native_wrapper(layer, graph)
    return layer


# Register designer-manifest primitives now that the op dispatch table is
# populated. This used to live at the bottom of primitives.py, where it always
# died on a circular import (primitives -> compiler -> graph -> primitives)
# that was silently swallowed, so designer ops never registered in pipeline
# import order.
_DESIGNER_COMPONENTS = (
    Path(__file__).resolve().parents[2] / "aria_designer" / "components"
)
if _DESIGNER_COMPONENTS.exists():
    load_primitives_from_designer(_DESIGNER_COMPONENTS)

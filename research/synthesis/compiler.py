"""
Computation Graph Compiler

Compiles a ComputationGraph into a live PyTorch nn.Module.
Each OpNode becomes a concrete tensor operation, with learnable
parameters allocated for parameterized ops.
"""

from __future__ import annotations

import logging
from typing import List

import torch
import torch.nn as nn

from .graph import ComputationGraph
from . import native_compile
from .compiler_registry import OP_DISPATCH, load_split_op_modules

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
        use_ir: If True (default), prefers native subgraph dispatch and falls
            back to IRExecutor when native execution is unavailable.
        executor: `default` preserves the current path. `ir_v2` enables the
            isolated experimental executor instance.
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
    if use_ir:
        model.layers = nn.ModuleList(
            [
                _compile_layer_module(
                    g, prefer_fast_path=True, executor_variant=executor
                )
                for g in layer_graphs
            ]
        )
    return model


def _compile_layer_module(
    graph: ComputationGraph,
    *,
    prefer_fast_path: bool,
    executor_variant: str = "default",
) -> nn.Module:
    if not prefer_fast_path:
        layer = CompiledLayer(graph)
        native_compile.attach_partial_native_wrapper(layer, graph)
        return layer

    native_layer = native_compile.try_compile_native_subgraph_layer(
        CompiledLayer, graph
    )
    if native_layer is not None:
        return native_layer

    from .ir_executor import IRExecutor
    from .ir_executor_v2 import IRExecutorV2

    if executor_variant == "ir_v2":
        layer = IRExecutorV2(graph.lower_to_ir(), source_graph=graph)
    else:
        layer = IRExecutor(graph.lower_to_ir(), source_graph=graph)
    native_compile.attach_partial_native_wrapper(layer, graph)
    return layer

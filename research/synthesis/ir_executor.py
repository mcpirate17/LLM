"""
IR Executor

High-performance execution of ComputationGraphIR using torch.compile
and registry-based dispatch. Minimizes Python overhead by lowering
the entire IR traversal into a single compiled kernel.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Tuple

import torch
import torch.nn as nn

from .graph import ComputationGraphIR
from .ir_executor_native import configure_native_execution
from .ir_executor_plan import build_executor_plan
from .ir_executor_runtime import (
    execute_plan_loop,
    execute_plan_loop_with_native_segments,
    initialize_execution_state,
)

logger = logging.getLogger(__name__)


class IRExecutor(nn.Module):
    """Executes ComputationGraphIR with minimal overhead."""

    def __init__(self, ir: ComputationGraphIR, source_graph=None):
        super().__init__()
        self.source_graph = source_graph
        self.model_dim = ir.model_dim
        self.op_codes = ir.op_codes
        self.input_indices = ir.input_indices
        self.output_node_idx = ir.output_node_idx
        self.configs = ir.configs
        self._subgraph_dispatcher = None
        self._native_chain_segments = ()
        self._native_chain_segment_slots = ()
        self._native_forward_wrapper = None
        self._native_setup_reason = None
        self._native_setup_detail = None
        self._output_idx_int = int(ir.output_node_idx)
        self._has_native_chain_slots = False
        self._last_execution_path = "uninitialized"
        self._execution_stats = {
            "native_subgraph_dispatches": 0,
            "native_chain_dispatches": 0,
            "hybrid_native_python_ir_loops": 0,
            "python_ir_loop_fallbacks": 0,
        }

        plan = build_executor_plan(ir)
        self.consumer_counts = plan.consumer_counts
        if self.output_node_idx is not None:
            for i in range(len(self.op_codes)):
                if self.op_codes[i] == 0:
                    continue
                if i != int(self.output_node_idx) and self.consumer_counts[i] == 0:
                    op_name = getattr(plan.flat_ops[i], "op_name", "unknown")
                    logger.warning(
                        "IRExecutor: node %d (%s) has zero consumers (possible dead branch)",
                        i,
                        op_name,
                    )

        self.ops = plan.ops
        self.idx_to_op_idx = plan.idx_to_op_idx
        self._flat_ops = plan.flat_ops
        self._n_nodes = plan.n_nodes
        self._counts_original = plan.counts_original
        self._counts_buf = plan.counts_buf
        self._input_node_indices = plan.input_node_indices
        self._exec_node_indices = plan.exec_node_indices
        self._exec_in1_indices = plan.exec_in1_indices
        self._exec_in2_indices = plan.exec_in2_indices
        self._exec_ops = plan.exec_ops
        self._node_outputs_buf = [None] * self._n_nodes

        # torch.compile can dominate runtime for short-lived candidate models.
        # Keep it opt-in so screening throughput does not get bottlenecked by
        # per-architecture compile/recompile overhead.
        enable_compile = os.getenv(
            "RESEARCH_ENABLE_TORCH_COMPILE", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        if enable_compile:
            try:
                self.forward = torch.compile(self.forward)
            except Exception as e:
                logger.debug("torch.compile failed for IRExecutor: %s", e)

        native_cfg = configure_native_execution(
            self,
            self.source_graph,
            flat_ops=self._flat_ops,
            n_nodes=self._n_nodes,
            exec_plan_node_indices=self._exec_node_indices,
        )
        self._subgraph_dispatcher = native_cfg.subgraph_dispatcher
        self._native_chain_segments = native_cfg.native_chain_segments
        self._native_chain_segment_slots = native_cfg.native_chain_segment_slots
        self._has_native_chain_slots = bool(self._native_chain_segment_slots)
        self._native_forward_wrapper = native_cfg.native_forward_wrapper
        self._native_setup_reason = native_cfg.setup_reason
        self._native_setup_detail = native_cfg.setup_detail

    def _wrapper_stats(self) -> Tuple[int, int]:
        wrapper = self._native_forward_wrapper
        if wrapper is None:
            return (0, 0)
        stats = wrapper.stats
        return (
            int(stats.get("native_dispatches", 0)),
            int(stats.get("fallbacks", 0)),
        )

    def forward(
        self, x: torch.Tensor, capture_intermediates: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """Lowered execution loop. torch.compile fuses this into a single kernel."""
        if not capture_intermediates and self._subgraph_dispatcher is not None:
            native_result = self._subgraph_dispatcher.try_dispatch(x)
            if native_result is not None:
                self._last_execution_path = "native_subgraph"
                self._execution_stats["native_subgraph_dispatches"] += 1
                return native_result

        wrapper_dispatches_before, _ = self._wrapper_stats()
        native_chain_dispatches_before = self._execution_stats[
            "native_chain_dispatches"
        ]
        counts, node_outputs, captured = initialize_execution_state(
            n_nodes=self._n_nodes,
            counts_buf=self._counts_buf,
            counts_original=self._counts_original,
            input_node_indices=self._input_node_indices,
            x=x,
            capture_intermediates=capture_intermediates,
            node_outputs_buf=None if capture_intermediates else self._node_outputs_buf,
        )

        if capture_intermediates or not self._has_native_chain_slots:
            execute_plan_loop(
                counts=counts,
                node_outputs=node_outputs,
                captured=captured,
                output_idx=self._output_idx_int,
                exec_node_indices=self._exec_node_indices,
                exec_in1_indices=self._exec_in1_indices,
                exec_in2_indices=self._exec_in2_indices,
                exec_ops=self._exec_ops,
            )
        else:
            self._execution_stats["native_chain_dispatches"] += (
                execute_plan_loop_with_native_segments(
                    counts=counts,
                    node_outputs=node_outputs,
                    captured=captured,
                    output_idx=self._output_idx_int,
                    exec_node_indices=self._exec_node_indices,
                    exec_in1_indices=self._exec_in1_indices,
                    exec_in2_indices=self._exec_in2_indices,
                    exec_ops=self._exec_ops,
                    chain_segment_slots=self._native_chain_segment_slots,
                )
            )

        res = node_outputs[self._output_idx_int]
        if res is None:
            logger.warning(
                "IRExecutor: output node %d produced None, returning input",
                self._output_idx_int,
            )
            res = x

        wrapper_dispatches_after, _ = self._wrapper_stats()
        native_chain_dispatches_after = self._execution_stats["native_chain_dispatches"]
        if (
            wrapper_dispatches_after > wrapper_dispatches_before
            or native_chain_dispatches_after > native_chain_dispatches_before
        ):
            self._last_execution_path = "hybrid_native_python_ir_loop"
            self._execution_stats["hybrid_native_python_ir_loops"] += 1
        else:
            self._last_execution_path = "python_ir_loop"
            self._execution_stats["python_ir_loop_fallbacks"] += 1

        if captured is not None:
            return res, captured
        node_outputs[self._output_idx_int] = None
        return res

    @property
    def execution_stats(self) -> Dict[str, int | str | bool]:
        wrapper_dispatches, wrapper_fallbacks = self._wrapper_stats()
        dispatcher_stats = (
            self._subgraph_dispatcher.stats
            if self._subgraph_dispatcher is not None
            and hasattr(self._subgraph_dispatcher, "stats")
            else {}
        )
        return {
            "last_execution_path": self._last_execution_path,
            "native_subgraph_available": self._subgraph_dispatcher is not None,
            "native_setup_reason": self._native_setup_reason,
            "native_setup_detail": self._native_setup_detail,
            "native_subgraph_refusal_reason": dispatcher_stats.get(
                "last_refusal_reason"
            ),
            "native_chain_segments": len(self._native_chain_segments),
            "partial_native_available": self._native_forward_wrapper is not None,
            "partial_native_dispatches": wrapper_dispatches,
            "partial_native_fallbacks": wrapper_fallbacks,
            **self._execution_stats,
        }

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
    execute_plan_entry,
    initialize_execution_frame,
    maybe_dispatch_native_segment,
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
        self._native_chain_segments_by_plan_index = {}
        self._native_forward_wrapper = None
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
        self._op_codes_list = plan.op_codes_list
        self._in1_list = plan.in1_list
        self._in2_list = plan.in2_list
        self._counts_original = plan.counts_original
        self._counts_buf = plan.counts_buf
        self._input_node_indices = plan.input_node_indices
        self._exec_plan = plan.exec_plan
        self._exec_plan_node_indices = plan.exec_plan_node_indices

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
            op_codes_list=self._op_codes_list,
            exec_plan_node_indices=self._exec_plan_node_indices,
        )
        self._subgraph_dispatcher = native_cfg.subgraph_dispatcher
        self._native_chain_segments = native_cfg.native_chain_segments
        self._native_chain_segments_by_plan_index = (
            native_cfg.native_chain_segments_by_plan_index or {}
        )
        self._native_forward_wrapper = native_cfg.native_forward_wrapper

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
        frame = initialize_execution_frame(
            n_nodes=len(self._op_codes_list),
            counts_buf=self._counts_buf,
            counts_original=self._counts_original,
            output_idx=int(self.output_node_idx),
            input_node_indices=self._input_node_indices,
            x=x,
            capture_intermediates=capture_intermediates,
        )

        plan_index = 0
        while plan_index < len(self._exec_plan):
            segment = (
                self._native_chain_segments_by_plan_index.get(plan_index)
                if not capture_intermediates
                else None
            )
            if segment is not None and maybe_dispatch_native_segment(frame, segment):
                plan_index = segment.end_plan_index + 1
                self._execution_stats["native_chain_dispatches"] += 1
                continue

            execute_plan_entry(frame, self._exec_plan[plan_index])
            plan_index += 1

        res = frame.node_outputs[frame.output_idx]
        if res is None:
            logger.warning(
                "IRExecutor: output node %d produced None, returning input",
                frame.output_idx,
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

        if frame.captured is not None:
            return res, frame.captured
        return res

    @property
    def execution_stats(self) -> Dict[str, int | str | bool]:
        wrapper_dispatches, wrapper_fallbacks = self._wrapper_stats()
        return {
            "last_execution_path": self._last_execution_path,
            "native_subgraph_available": self._subgraph_dispatcher is not None,
            "native_chain_segments": len(self._native_chain_segments),
            "partial_native_available": self._native_forward_wrapper is not None,
            "partial_native_dispatches": wrapper_dispatches,
            "partial_native_fallbacks": wrapper_fallbacks,
            **self._execution_stats,
        }

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .graph import ComputationGraphIR
from . import ir_executor_v2_native
from .ir_executor_v2_plan import build_ir_executor_v2_plan
from .ir_executor_runtime import execute_plan_loop, initialize_execution_state


class IRExecutorV2(nn.Module):
    """Experimental executor instance.

    This class is intentionally isolated from the default compiler path.
    It tries a whole-graph native dispatch first and falls back to the
    existing IRExecutor implementation for parity.
    """

    def __init__(self, ir: ComputationGraphIR, source_graph=None):
        super().__init__()
        self._ir = ir
        self.source_graph = source_graph
        self._has_bound_params = bool(
            source_graph is not None
            and ir_executor_v2_native.graph_has_bound_params(source_graph)
        )
        self._plan = None
        self.ops = nn.ModuleList()
        self._flat_ops = ()
        self._n_nodes = 0
        self._counts_original = ()
        self._counts_buf = ()
        self._input_node_indices = ()
        self._exec_node_indices = ()
        self._exec_in1_indices = ()
        self._exec_in2_indices = ()
        self._exec_ops = ()
        if self._has_bound_params:
            self._ensure_plan()
        native_cfg = ir_executor_v2_native.configure_ir_executor_v2_native(
            source_graph,
            **self._bound_native_inputs(),
        )
        self._native_dispatcher = native_cfg.dispatcher
        self._native_setup_reason = native_cfg.setup_reason
        self._native_setup_detail = native_cfg.setup_detail
        self._last_execution_path = "uninitialized"
        self._execution_stats = {
            "v2_native_dispatches": 0,
            "v2_fallback_dispatches": 0,
        }

    def _ensure_plan(self):
        plan = self._plan
        if plan is None:
            plan = build_ir_executor_v2_plan(self._ir)
            exec_plan = plan.executor_plan
            self._plan = plan
            self.ops = exec_plan.ops
            self._flat_ops = exec_plan.flat_ops
            self._n_nodes = exec_plan.n_nodes
            self._counts_original = exec_plan.counts_original
            self._counts_buf = exec_plan.counts_buf
            self._input_node_indices = exec_plan.input_node_indices
            self._exec_node_indices = exec_plan.exec_node_indices
            self._exec_in1_indices = exec_plan.exec_in1_indices
            self._exec_in2_indices = exec_plan.exec_in2_indices
            self._exec_ops = exec_plan.exec_ops
        return plan

    def _bound_native_inputs(self) -> Dict[str, list[object | None] | list[int]]:
        if not self._has_bound_params:
            return {}
        plan = self._ensure_plan()
        ir_node_ids = (
            plan.ir.node_ids.tolist()
            if plan.ir.node_ids is not None
            else list(range(plan.n_nodes))
        )
        return {"flat_ops": self._flat_ops, "ir_node_ids": ir_node_ids}

    def forward(
        self, x: torch.Tensor, capture_intermediates: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        if not capture_intermediates and self._native_dispatcher is not None:
            native_result = self._native_dispatcher.try_dispatch(x)
            if native_result is not None:
                self._last_execution_path = "v2_native_subgraph"
                self._execution_stats["v2_native_dispatches"] += 1
                return native_result
        self._last_execution_path = "v2_fallback"
        self._execution_stats["v2_fallback_dispatches"] += 1
        plan = self._ensure_plan()
        counts, node_outputs, captured = initialize_execution_state(
            n_nodes=self._n_nodes,
            counts_buf=self._counts_buf,
            counts_original=self._counts_original,
            input_node_indices=self._input_node_indices,
            x=x,
            capture_intermediates=capture_intermediates,
        )
        execute_plan_loop(
            counts=counts,
            node_outputs=node_outputs,
            captured=captured,
            output_idx=plan.output_node_idx,
            exec_node_indices=self._exec_node_indices,
            exec_in1_indices=self._exec_in1_indices,
            exec_in2_indices=self._exec_in2_indices,
            exec_ops=self._exec_ops,
        )
        result = node_outputs[plan.output_node_idx]
        if captured is not None:
            return result, captured
        return result

    @property
    def execution_stats(self) -> Dict[str, int | str | bool]:
        dispatcher_stats = (
            self._native_dispatcher.stats
            if self._native_dispatcher is not None
            and hasattr(self._native_dispatcher, "stats")
            else {}
        )
        return {
            "last_execution_path": self._last_execution_path,
            "native_subgraph_available": self._native_dispatcher is not None,
            "native_setup_reason": self._native_setup_reason,
            "native_setup_detail": self._native_setup_detail,
            "native_subgraph_refusal_reason": dispatcher_stats.get(
                "last_refusal_reason"
            ),
            "plan_initialized": self._plan is not None,
            **self._execution_stats,
        }

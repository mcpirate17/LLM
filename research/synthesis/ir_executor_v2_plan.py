from __future__ import annotations

from dataclasses import dataclass

from .graph import ComputationGraphIR
from .ir_executor_plan import ExecutorPlan, build_executor_plan


@dataclass(slots=True)
class IRExecutorV2Plan:
    ir: ComputationGraphIR
    executor_plan: ExecutorPlan
    n_nodes: int
    output_node_idx: int


def build_ir_executor_v2_plan(ir: ComputationGraphIR) -> IRExecutorV2Plan:
    executor_plan = build_executor_plan(ir)
    return IRExecutorV2Plan(
        ir=ir,
        executor_plan=executor_plan,
        n_nodes=executor_plan.n_nodes,
        output_node_idx=int(ir.output_node_idx),
    )

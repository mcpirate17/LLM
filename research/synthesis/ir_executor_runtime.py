from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .ir_executor_plan import ExecPlanEntry


@dataclass(slots=True)
class ExecutionFrame:
    counts: list[int]
    output_idx: int
    node_outputs: list[Optional[torch.Tensor]]
    captured: dict[int, torch.Tensor] | None


def initialize_execution_frame(
    *,
    n_nodes: int,
    counts_buf: list[int],
    counts_original: list[int],
    output_idx: int,
    input_node_indices: tuple[int, ...],
    x: torch.Tensor,
    capture_intermediates: bool,
) -> ExecutionFrame:
    node_outputs: list[Optional[torch.Tensor]] = [None] * n_nodes
    counts_buf[:] = counts_original
    for input_idx in input_node_indices:
        node_outputs[input_idx] = x
    return ExecutionFrame(
        counts=counts_buf,
        output_idx=output_idx,
        node_outputs=node_outputs,
        captured={} if capture_intermediates else None,
    )


def maybe_dispatch_native_segment(frame: ExecutionFrame, segment) -> bool:
    seg_input = frame.node_outputs[segment.input_ir_idx]
    native_result = segment.dispatcher.try_dispatch(seg_input)
    if native_result is None:
        return False

    frame.node_outputs[segment.output_ir_idx] = native_result
    frame.counts[segment.input_ir_idx] -= 1
    if (
        frame.counts[segment.input_ir_idx] <= 0
        and segment.input_ir_idx != frame.output_idx
        and frame.captured is None
    ):
        frame.node_outputs[segment.input_ir_idx] = None

    for release_idx in segment.release_ir_indices:
        frame.counts[release_idx] -= 1
        if release_idx != frame.output_idx and frame.captured is None:
            frame.node_outputs[release_idx] = None
    return True


def execute_plan_entry(frame: ExecutionFrame, entry: ExecPlanEntry) -> None:
    t1 = frame.node_outputs[entry.in1_idx]
    if entry.in2_idx != -1:
        t2 = frame.node_outputs[entry.in2_idx]
        out = entry.op(t1, t2)
        frame.counts[entry.in2_idx] -= 1
        if (
            frame.counts[entry.in2_idx] <= 0
            and entry.in2_idx != frame.output_idx
            and frame.captured is None
        ):
            frame.node_outputs[entry.in2_idx] = None
    else:
        out = entry.op(t1)

    frame.node_outputs[entry.node_idx] = out
    frame.counts[entry.in1_idx] -= 1
    if (
        frame.counts[entry.in1_idx] <= 0
        and entry.in1_idx != frame.output_idx
        and frame.captured is None
    ):
        frame.node_outputs[entry.in1_idx] = None

    if frame.captured is not None:
        frame.captured[entry.node_idx] = out

from __future__ import annotations

from typing import Optional

import torch

# Each helper below orchestrates per-block op dispatch via a Python loop that
# reads Python ints (output_idx, in2_idx == -1, counts[i] <= 0) inside the
# body. torch.dynamo specializes on those ints, so an interleaved hybrid with
# 18 IRExecutor blocks burns through the recompile cache (the 2026-05-20
# benchmark crashed at 1078 ms/step from this; raising the cache made it 8×
# worse). Disabling dynamo at the loop boundary keeps the leaf ops eager —
# still cuBLAS/cuDNN-fast — while the outer TinyLM still benefits from
# torch.compile. Don't add @torch.compile to anything in this file.
_disable = getattr(torch, "_dynamo", None) and torch._dynamo.disable
if _disable is None:  # very old torch — no dynamo to disable

    def _disable(fn):  # type: ignore[no-redef]
        return fn


@_disable
def initialize_execution_state(
    *,
    n_nodes: int,
    counts_buf: list[int],
    counts_original: list[int],
    input_node_indices: tuple[int, ...],
    x: torch.Tensor,
    capture_intermediates: bool,
    node_outputs_buf: list[Optional[torch.Tensor]] | None = None,
) -> tuple[list[int], list[Optional[torch.Tensor]], dict[int, torch.Tensor] | None]:
    if node_outputs_buf is None or len(node_outputs_buf) != n_nodes:
        node_outputs: list[Optional[torch.Tensor]] = [None] * n_nodes
    else:
        node_outputs = node_outputs_buf
        for idx in range(n_nodes):
            node_outputs[idx] = None
    counts_buf[:] = counts_original
    for input_idx in input_node_indices:
        node_outputs[input_idx] = x
    return counts_buf, node_outputs, {} if capture_intermediates else None


@_disable
def dispatch_native_segment(
    *,
    counts: list[int],
    node_outputs: list[Optional[torch.Tensor]],
    captured: dict[int, torch.Tensor] | None,
    output_idx: int,
    segment,
) -> bool:
    release_outputs = captured is None
    seg_input = node_outputs[segment.input_ir_idx]
    native_result = segment.dispatcher.try_dispatch(seg_input)
    if native_result is None:
        return False

    node_outputs[segment.output_ir_idx] = native_result
    input_ir_idx = segment.input_ir_idx
    counts[input_ir_idx] -= segment.input_consume_count
    if counts[input_ir_idx] <= 0 and input_ir_idx != output_idx and release_outputs:
        node_outputs[input_ir_idx] = None

    for release_idx, release_count in segment.release_ir_counts:
        counts[release_idx] -= release_count
        if release_idx != output_idx and release_outputs:
            node_outputs[release_idx] = None
    return True


@_disable
def execute_plan_loop(
    *,
    counts: list[int],
    node_outputs: list[Optional[torch.Tensor]],
    captured: dict[int, torch.Tensor] | None,
    output_idx: int,
    exec_node_indices: tuple[int, ...],
    exec_in1_indices: tuple[int, ...],
    exec_in2_indices: tuple[int, ...],
    exec_ops: tuple,
) -> None:
    release_outputs = captured is None
    for plan_index in range(len(exec_node_indices)):
        node_idx = exec_node_indices[plan_index]
        in1_idx = exec_in1_indices[plan_index]
        in2_idx = exec_in2_indices[plan_index]
        op = exec_ops[plan_index]
        t1 = node_outputs[in1_idx]
        if in2_idx != -1:
            t2 = node_outputs[in2_idx]
            out = op(t1, t2)
            counts[in2_idx] -= 1
            if counts[in2_idx] <= 0 and in2_idx != output_idx and release_outputs:
                node_outputs[in2_idx] = None
        else:
            out = op(t1)

        node_outputs[node_idx] = out
        counts[in1_idx] -= 1
        if counts[in1_idx] <= 0 and in1_idx != output_idx and release_outputs:
            node_outputs[in1_idx] = None

        if captured is not None:
            captured[node_idx] = out


@_disable
def execute_plan_loop_with_native_segments(
    *,
    counts: list[int],
    node_outputs: list[Optional[torch.Tensor]],
    captured: dict[int, torch.Tensor] | None,
    output_idx: int,
    exec_node_indices: tuple[int, ...],
    exec_in1_indices: tuple[int, ...],
    exec_in2_indices: tuple[int, ...],
    exec_ops: tuple,
    chain_segment_slots: tuple,
) -> int:
    native_dispatches = 0
    release_outputs = captured is None
    plan_len = len(exec_node_indices)
    plan_index = 0
    while plan_index < plan_len:
        segment = chain_segment_slots[plan_index]
        if segment is not None and dispatch_native_segment(
            counts=counts,
            node_outputs=node_outputs,
            captured=captured,
            output_idx=output_idx,
            segment=segment,
        ):
            native_dispatches += 1
            plan_index = segment.end_plan_index + 1
            continue

        node_idx = exec_node_indices[plan_index]
        in1_idx = exec_in1_indices[plan_index]
        in2_idx = exec_in2_indices[plan_index]
        op = exec_ops[plan_index]
        t1 = node_outputs[in1_idx]
        if in2_idx != -1:
            t2 = node_outputs[in2_idx]
            out = op(t1, t2)
            counts[in2_idx] -= 1
            if counts[in2_idx] <= 0 and in2_idx != output_idx and release_outputs:
                node_outputs[in2_idx] = None
        else:
            out = op(t1)

        node_outputs[node_idx] = out
        counts[in1_idx] -= 1
        if counts[in1_idx] <= 0 and in1_idx != output_idx and release_outputs:
            node_outputs[in1_idx] = None

        if captured is not None:
            captured[node_idx] = out
        plan_index += 1
    return native_dispatches

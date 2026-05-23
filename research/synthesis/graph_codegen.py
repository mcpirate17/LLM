"""Code-gen a flat-Python forward() for a synthesis ComputationGraph.

Replaces the per-step Python orchestration loop in ``CompiledLayer.forward``
(``for nid, is_input, op, input_ids, is_boundary in self._fwd_plan:``) with a
straight-line function whose source is generated at compile time. Each op call
becomes one explicit line, mathspace-boundary RMSNorm fixups are inlined where
needed, and dead intermediates are dropped via plain ``del`` statements.

Why this matters: dynamo cannot trace the orchestrated loop without
specializing on every Python int it reads (``output_id``, ``counts[i] <= 0``,
the ``_fwd_plan`` tuple structure), which is why the loop is currently
``@torch._dynamo.disable``'d. A flat function has none of that — dynamo traces
it cleanly, fuses adjacent ops, and the per-branch Python budget collapses to
the import-time exec overhead (~microseconds, vs ~7 ms for the loop).

API:
    layer = CodegenLayer(graph)             # drop-in for CompiledLayer
    src   = generate_forward_source(graph)  # raw source if you want to inspect
"""

from __future__ import annotations

import textwrap
from typing import Sequence

import torch

from .compiled_model import CompiledLayer
from .compiler_constants import MATHSPACE_OPS
from .graph import ComputationGraph


def _var(nid: int) -> str:
    return f"v{nid}"


def _consumer_counts(graph: ComputationGraph, topo: Sequence[int]) -> dict[int, int]:
    counts: dict[int, int] = {nid: 0 for nid in topo}
    for nid in topo:
        for iid in graph.nodes[nid].input_ids:
            counts[iid] = counts.get(iid, 0) + 1
    return counts


def _mathspace_boundary_nids(graph: ComputationGraph, topo: Sequence[int]) -> set[int]:
    consumers: dict[int, list[int]] = {nid: [] for nid in graph.nodes}
    for nid in topo:
        for iid in graph.nodes[nid].input_ids:
            consumers[iid].append(nid)
    output_id = graph._output_node_id
    boundary: set[int] = set()
    for nid in topo:
        node = graph.nodes[nid]
        if node.is_input or node.op_name not in MATHSPACE_OPS:
            continue
        node_consumers = consumers.get(nid, [])
        if not node_consumers and nid == output_id:
            boundary.add(nid)
            continue
        for cid in node_consumers:
            if graph.nodes[cid].op_name not in MATHSPACE_OPS:
                boundary.add(nid)
                break
    return boundary


def generate_forward_source(graph: ComputationGraph) -> str:
    """Return the source for a CompiledLayer-compatible flat forward().

    The emitted function reads ``self.ops`` (a ``nn.ModuleDict`` keyed by
    stringified node id, just like ``CompiledLayer``) and produces an output
    tensor. No mutable counts, no plan-tuple unpacking, no Python loop.
    """
    topo = graph.topological_order()
    if not topo:
        raise ValueError("Cannot generate forward() for an empty graph")
    output_id = graph._output_node_id
    if output_id is None:
        raise ValueError("Graph has no output node set")

    consumer_counts = _consumer_counts(graph, topo)
    boundary_nids = _mathspace_boundary_nids(graph, topo)

    lines: list[str] = ["def forward(self, x):"]
    pending_uses = {nid: consumer_counts.get(nid, 0) for nid in topo}

    for nid in topo:
        node = graph.nodes[nid]
        var = _var(nid)
        if node.is_input:
            lines.append(f"    {var} = x")
            continue

        args = ", ".join(_var(iid) for iid in node.input_ids)
        op_key = f"self.ops[{str(nid)!r}]"
        lines.append(f"    {var} = {op_key}({args})")

        if nid in boundary_nids:
            # Mirror CompiledLayer's mathspace-boundary RMSNorm: cast to fp32,
            # rsqrt of mean-of-squares + eps, then cast back to the original
            # dtype. Hoisting this into source lets dynamo fuse it with the
            # surrounding op when shapes line up.
            lines.append(
                f"    _rms_f = {var} if {var}.dtype == torch.float32 else {var}.float()"
            )
            lines.append(
                "    _rms = _rms_f.pow(2).mean(dim=-1, keepdim=True).add_(1e-6).rsqrt_()"
            )
            lines.append(
                f"    {var} = ({var} * _rms) if {var}.dtype == torch.float32 "
                f"else ({var} * _rms.to({var}.dtype))"
            )

        # Drop intermediate tensors as soon as their last consumer fires, so
        # the activation memory footprint matches CompiledLayer's per-step
        # release (which calls `outputs[iid] = None` + `del`).
        for iid in node.input_ids:
            pending_uses[iid] -= 1
            if pending_uses[iid] <= 0 and iid != output_id:
                lines.append(f"    del {_var(iid)}")

    lines.append(f"    return {_var(output_id)}")
    return "\n".join(lines)


def _compile_forward_source(source: str) -> "callable":
    namespace: dict = {"torch": torch}
    exec(textwrap.dedent(source), namespace)  # nosec B102  # nosemgrep: python-dangerous-eval-exec - source is generated from a trusted in-process IR, not user input
    fn = namespace.get("forward")
    if fn is None:
        raise RuntimeError("generated source did not define forward()")
    return fn


class CodegenLayer(CompiledLayer):
    """CompiledLayer with a code-genned flat forward().

    Inherits CompiledLayer's __init__ (op registration, mathspace boundary
    detection, counts buffers) so the parameter graph + state_dict layout
    are identical and existing checkpoints load unchanged. We just swap the
    Python orchestration loop for a generated straight-line function.
    """

    def __init__(self, graph: ComputationGraph) -> None:
        super().__init__(graph)
        source = generate_forward_source(graph)
        # Stash for debugging + parity tests.
        self._codegen_source = source
        self._codegen_forward = _compile_forward_source(source).__get__(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dispatcher = getattr(self, "_subgraph_dispatcher", None)
        if dispatcher is not None:
            result = dispatcher.try_dispatch(x)
            if result is not None:
                return result
        return self._codegen_forward(x)

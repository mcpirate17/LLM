from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np

from aria_designer.api.app.workflow_graph_cache import materialize_workflow_graph
from research.synthesis._json_compat import dumps_json, loads_json
from research.synthesis.graph import ComputationGraph
from research.synthesis.native_ir_converter import graph_to_native_ir_json
from research.synthesis.native_topology import (
    _build_canonical_topology_inputs,
    compute_topological_order,
)
from research.synthesis.serializer import (
    _graph_dict_from_json_cached,
    graph_from_json,
    graph_to_json,
)
from research.synthesis.workflow_converter import workflow_to_computation_graph
from research.scientist.native.dispatch import (
    dispatch_graph_backward_native,
    dispatch_graph_forward_native_saved,
)
from research.scientist.intelligence.graph_ops import (
    extract_graph_ops,
    extract_unique_graph_ops,
)


def _bench(fn: Callable[[], Any], repeats: int) -> float:
    started = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - started) * 1000.0 / repeats


def _load_workflow(path: str | Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _build_native_graph(model_dim: int, depth: int) -> ComputationGraph:
    graph = ComputationGraph(model_dim)
    node = graph.add_input()
    for _ in range(depth):
        node = graph.add_op("relu", [node], {})
        node = graph.add_op("silu", [node], {})
    graph.set_output(node)
    return graph


def _speedup(before_ms: float, after_ms: float) -> float:
    if after_ms <= 0.0:
        return float("inf")
    return before_ms / after_ms


def _graph_to_native_ir_json_uncached(graph: ComputationGraph) -> str:
    reachable = graph.get_reachable_nodes()
    if len(reachable) != len(graph.nodes):
        dead_count = len(graph.nodes) - len(reachable)
        raise ValueError(
            f"Graph contains {dead_count} unreachable nodes (dead branches)"
        )

    nodes = []
    edges = []
    for nid in sorted(graph.nodes.keys()):
        node = graph.nodes[nid]
        nodes.append(
            {
                "id": node.id,
                "op_name": node.op_name,
                "input_ids": list(node.input_ids),
                "config": dict(node.config),
                "is_input": node.is_input,
                "is_output": node.is_output,
            }
        )
        for inp_id in node.input_ids:
            edges.append({"source": inp_id, "target": node.id})

    return dumps_json(
        {
            "schema_version": "native_ir.v1",
            "model_dim": graph.model_dim,
            "nodes": nodes,
            "edges": edges,
            "output_node_id": graph._output_node_id,
        }
    )


def _graph_from_json_uncached(json_str: str) -> ComputationGraph:
    return ComputationGraph.from_dict(loads_json(json_str))


def _canonical_topology_inputs_uncached(
    graph: ComputationGraph,
) -> tuple[int, list[tuple[int, int]], list[str], list[str], list[list[int]]]:
    n_nodes = graph._next_id
    op_names = [""] * n_nodes
    config_strs = [""] * n_nodes
    node_inputs = [[] for _ in range(n_nodes)]
    edges: list[tuple[int, int]] = []

    for nid, node in graph.nodes.items():
        op_names[nid] = node.op_name
        if node.config:
            config_items = sorted(f"{k}={v}" for k, v in node.config.items())
            config_strs[nid] = f"[{','.join(config_items)}]"
        node_inputs[nid] = list(node.input_ids)
        for iid in node.input_ids:
            edges.append((iid, nid))

    return n_nodes, edges, op_names, config_strs, node_inputs


def _legacy_graph_ops_from_json(json_str: str) -> list[str]:
    graph_data = loads_json(json_str)
    nodes = graph_data.get("nodes", {}) if isinstance(graph_data, dict) else {}
    ops: list[str] = []
    if isinstance(nodes, dict):
        node_iter = nodes.values()
    elif isinstance(nodes, list):
        node_iter = nodes
    else:
        node_iter = ()
    for node in node_iter:
        if not isinstance(node, dict):
            continue
        op_name = str(
            node.get("op_name") or node.get("op_type") or node.get("op") or ""
        ).strip()
        if op_name and op_name != "input":
            ops.append(op_name)
    return ops


def _legacy_unique_graph_ops_from_json(json_str: str) -> list[str]:
    return sorted(set(_legacy_graph_ops_from_json(json_str)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark graph dataflow hot paths")
    parser.add_argument(
        "--workflow",
        default="aria_designer/ui/public/examples/transformer_mini.json",
        help="Workflow fixture to benchmark",
    )
    parser.add_argument(
        "--convert-repeats",
        type=int,
        default=400,
        help="Number of workflow conversion repetitions",
    )
    parser.add_argument(
        "--repeat-repeats",
        type=int,
        default=2000,
        help="Number of repeated graph/IR calls",
    )
    parser.add_argument(
        "--native-depth",
        type=int,
        default=8,
        help="Depth for the synthetic native dispatch graph",
    )
    parser.add_argument(
        "--native-repeats",
        type=int,
        default=200,
        help="Number of native dispatch repetitions",
    )
    args = parser.parse_args()

    workflow = _load_workflow(args.workflow)
    model_dim = int((workflow.get("metadata") or {}).get("model_dim") or 256)

    convert_baseline_ms = _bench(
        lambda: workflow_to_computation_graph(workflow, model_dim),
        args.convert_repeats,
    )
    materialize_workflow_graph(workflow, model_dim)
    convert_cached_ms = _bench(
        lambda: materialize_workflow_graph(workflow, model_dim),
        args.convert_repeats,
    )
    graph = workflow_to_computation_graph(workflow, model_dim)
    topo_inputs_uncached_ms = _bench(
        lambda: _canonical_topology_inputs_uncached(graph),
        args.repeat_repeats,
    )
    _build_canonical_topology_inputs(graph)
    topo_inputs_cached_ms = _bench(
        lambda: _build_canonical_topology_inputs(graph),
        args.repeat_repeats,
    )
    topo_ms = _bench(
        lambda: compute_topological_order(graph),
        args.repeat_repeats,
    )
    ir_uncached_ms = _bench(
        lambda: _graph_to_native_ir_json_uncached(graph),
        args.repeat_repeats,
    )
    graph_to_native_ir_json(graph)
    ir_cached_ms = _bench(
        lambda: graph_to_native_ir_json(graph),
        args.repeat_repeats,
    )
    graph_json = graph_to_json(graph)
    ops_uncached_ms = _bench(
        lambda: _legacy_graph_ops_from_json(graph_json),
        args.repeat_repeats,
    )
    ops_fast_ms = _bench(
        lambda: extract_graph_ops(graph_json),
        args.repeat_repeats,
    )
    unique_ops_uncached_ms = _bench(
        lambda: _legacy_unique_graph_ops_from_json(graph_json),
        args.repeat_repeats,
    )
    unique_ops_fast_ms = _bench(
        lambda: extract_unique_graph_ops(graph_json),
        args.repeat_repeats,
    )
    parse_uncached_ms = _bench(
        lambda: _graph_from_json_uncached(graph_json),
        args.repeat_repeats,
    )
    _graph_dict_from_json_cached.cache_clear()
    graph_from_json(graph_json)
    parse_cached_ms = _bench(
        lambda: graph_from_json(graph_json),
        args.repeat_repeats,
    )

    native_graph = _build_native_graph(64, args.native_depth)
    x = np.random.randn(2, 8, 64).astype(np.float32)

    native_result: Dict[str, Any]
    try:
        fwd = dispatch_graph_forward_native_saved(native_graph, x)
        grad = np.ones_like(fwd["output"], dtype=np.float32)

        def _native_roundtrip():
            run = dispatch_graph_forward_native_saved(native_graph, x)
            dispatch_graph_backward_native(
                native_graph,
                grad,
                run["saved_activations"],
                ir_json=run.get("ir_json"),
            )

        native_roundtrip_ms = _bench(_native_roundtrip, args.native_repeats)
        native_result = {
            "available": True,
            "forward_backward_ms_per_call": round(native_roundtrip_ms, 6),
            "saved_state_type": type(fwd["saved_activations"]).__name__,
        }
    except Exception as exc:  # noqa: BLE001
        native_result = {
            "available": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    payload = {
        "workflow_fixture": str(args.workflow),
        "workflow_conversion_ms_per_call": round(convert_baseline_ms, 6),
        "workflow_conversion_cached_ms_per_call": round(convert_cached_ms, 6),
        "workflow_conversion_speedup_x": round(
            _speedup(convert_baseline_ms, convert_cached_ms),
            2,
        ),
        "topology_inputs_uncached_ms_per_call": round(topo_inputs_uncached_ms, 6),
        "topology_inputs_cached_ms_per_call": round(topo_inputs_cached_ms, 6),
        "topology_inputs_speedup_x": round(
            _speedup(topo_inputs_uncached_ms, topo_inputs_cached_ms),
            2,
        ),
        "topological_order_ms_per_call": round(topo_ms, 6),
        "native_ir_json_uncached_ms_per_call": round(ir_uncached_ms, 6),
        "native_ir_json_cached_ms_per_call": round(ir_cached_ms, 6),
        "native_ir_json_speedup_x": round(_speedup(ir_uncached_ms, ir_cached_ms), 2),
        "graph_ops_extract_uncached_ms_per_call": round(ops_uncached_ms, 6),
        "graph_ops_extract_fast_ms_per_call": round(ops_fast_ms, 6),
        "graph_ops_extract_speedup_x": round(
            _speedup(ops_uncached_ms, ops_fast_ms),
            2,
        ),
        "graph_unique_ops_uncached_ms_per_call": round(unique_ops_uncached_ms, 6),
        "graph_unique_ops_fast_ms_per_call": round(unique_ops_fast_ms, 6),
        "graph_unique_ops_speedup_x": round(
            _speedup(unique_ops_uncached_ms, unique_ops_fast_ms),
            2,
        ),
        "graph_from_json_uncached_ms_per_call": round(parse_uncached_ms, 6),
        "graph_from_json_cached_ms_per_call": round(parse_cached_ms, 6),
        "graph_from_json_speedup_x": round(
            _speedup(parse_uncached_ms, parse_cached_ms),
            2,
        ),
        "graph_nodes": len(graph.nodes),
        "graph_ops": graph.n_ops(),
        "graph_depth": graph.depth(),
        "native_roundtrip": native_result,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

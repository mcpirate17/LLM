#!/usr/bin/env python3
"""Generate adaptive lane workflow variants and optionally benchmark them."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


ROOT = Path("/home/tim/Projects/LLM/aria_designer")
OUT_DIR = ROOT / "workflows" / "generated"


def node(node_id: str, component_type: str, x: int, y: int, params: dict | None = None) -> dict:
    return {
        "id": node_id,
        "component_type": component_type,
        "params": params or {},
        "ui_meta": {"position": {"x": x, "y": y}},
    }


def edge(src: str, src_port: str, dst: str, dst_port: str) -> dict:
    return {
        "id": f"e_{src}_{src_port}_{dst}_{dst_port}",
        "source": src,
        "source_port": src_port,
        "target": dst,
        "target_port": dst_port,
    }


def base_workflow(workflow_id: str, name: str, hard_lane_nodes: list[dict], hard_lane_edges: list[dict], metadata: dict) -> dict:
    nodes = [
        node(
            "input",
            "io/input",
            440,
            0,
            {
                "source_type": "synthetic",
                "synthetic_pattern": "gaussian",
                "batch_size": 2,
                "seq_len": 64,
                "seed": 42,
                "dim": 256,
                "vocab_size": 32000,
            },
        ),
        node("difficulty", "routing/difficulty_scorer", 440, 120),
        node("lane_router", "routing/lane_router", 440, 240, {"num_lanes": 2, "routing_mode": "threshold"}),
        node("dispatch_fast", "structural/conditional_dispatch", 220, 380, {"num_lanes": 2, "lane": 0}),
        node("dispatch_hard", "structural/conditional_dispatch", 660, 380, {"num_lanes": 2, "lane": 1}),
        node("fast_gelu", "math/gelu", 220, 500),
        *hard_lane_nodes,
        node("combine", "math/add", 440, 860),
        node("compress", "linear_algebra/linear_proj_down", 440, 980),
        node("output", "io/output_head", 440, 1100, {"vocab_size": 32000, "tie_weights": True}),
    ]
    edges = [
        edge("input", "y", "difficulty", "x"),
        edge("difficulty", "y", "lane_router", "x"),
        edge("lane_router", "y", "dispatch_fast", "x"),
        edge("lane_router", "y", "dispatch_hard", "x"),
        edge("dispatch_fast", "y", "fast_gelu", "x"),
        *hard_lane_edges,
        edge("fast_gelu", "y", "combine", "a"),
        edge(hard_lane_nodes[-1]["id"], "y", "combine", "b"),
        edge("combine", "y", "compress", "x"),
        edge("compress", "y", "output", "x"),
    ]
    return {
        "schema_version": "workflow_graph.v1",
        "workflow_id": workflow_id,
        "name": name,
        "metadata": metadata,
        "nodes": nodes,
        "edges": edges,
    }


def variant_specs() -> list[dict]:
    base_meta = {
        "family": "adaptive_lane_split",
        "requested_pattern": "difficulty_split_fast_vs_routed_moe",
        "source_fingerprint": "23a9c75ef7ee1da6",
        "source_note": "Inspired by imported survivor fingerprint 23a9c75ef7ee1da6; mapped to approved Aria Designer components.",
    }
    return [
        {
            "workflow_id": "wf_adaptive_lane_moe_v1",
            "name": "Adaptive Lane Split V1 - Fast GELU vs MoE",
            "metadata": {**base_meta, "variant": "fast_gelu_vs_moe"},
            "hard_lane_nodes": [
                node("hard_moe", "channel_mixing/moe_topk", 660, 500),
                node("hard_gelu", "math/gelu", 660, 620),
            ],
            "hard_lane_edges": [
                edge("dispatch_hard", "y", "hard_moe", "x"),
                edge("hard_moe", "y", "hard_gelu", "x"),
            ],
        },
        {
            "workflow_id": "wf_adaptive_lane_moe_v2",
            "name": "Adaptive Lane Split V2 - Ultrametric Routed-MoE",
            "metadata": {**base_meta, "variant": "ultrametric_then_moe"},
            "hard_lane_nodes": [
                node("hard_ultra", "math_space/ultrametric_attention", 660, 500),
                node("hard_moe", "channel_mixing/moe_topk", 660, 620),
                node("hard_gelu", "math/gelu", 660, 740),
            ],
            "hard_lane_edges": [
                edge("dispatch_hard", "y", "hard_ultra", "x"),
                edge("hard_ultra", "y", "hard_moe", "x"),
                edge("hard_moe", "y", "hard_gelu", "x"),
            ],
        },
        {
            "workflow_id": "wf_adaptive_lane_moe_v3",
            "name": "Adaptive Lane Split V3 - Routed-MoE with Dense Refinement",
            "metadata": {**base_meta, "variant": "moe_then_dense_refine"},
            "hard_lane_nodes": [
                node("hard_moe", "channel_mixing/moe_topk", 660, 500),
                node("hard_proj", "linear_algebra/linear_proj", 660, 620),
                node("hard_gelu", "math/gelu", 660, 740),
            ],
            "hard_lane_edges": [
                edge("dispatch_hard", "y", "hard_moe", "x"),
                edge("hard_moe", "y", "hard_proj", "x"),
                edge("hard_proj", "y", "hard_gelu", "x"),
            ],
        },
        {
            "workflow_id": "wf_adaptive_lane_moe_v4",
            "name": "Adaptive Lane Split V4 - Ultrametric Routed-MoE Bottleneck",
            "metadata": {**base_meta, "variant": "ultrametric_moe_bottleneck"},
            "hard_lane_nodes": [
                node("hard_ultra", "math_space/ultrametric_attention", 660, 500),
                node("hard_moe", "channel_mixing/moe_topk", 660, 620),
                node("hard_bottleneck", "math_space/bottleneck_proj", 660, 740),
                node("hard_gelu", "math/gelu", 660, 860),
            ],
            "hard_lane_edges": [
                edge("dispatch_hard", "y", "hard_ultra", "x"),
                edge("hard_ultra", "y", "hard_moe", "x"),
                edge("hard_moe", "y", "hard_bottleneck", "x"),
                edge("hard_bottleneck", "y", "hard_gelu", "x"),
            ],
        },
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for spec in variant_specs():
        workflow = base_workflow(
            workflow_id=spec["workflow_id"],
            name=spec["name"],
            hard_lane_nodes=deepcopy(spec["hard_lane_nodes"]),
            hard_lane_edges=deepcopy(spec["hard_lane_edges"]),
            metadata=deepcopy(spec["metadata"]),
        )
        path = OUT_DIR / f"{spec['workflow_id']}.json"
        path.write_text(json.dumps(workflow, indent=2) + "\n", encoding="utf-8")
        manifest.append(
            {
                "workflow_id": spec["workflow_id"],
                "name": spec["name"],
                "path": str(path),
                "variant": spec["metadata"]["variant"],
            }
        )
    (OUT_DIR / "adaptive_lane_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest)} workflows to {OUT_DIR}")


if __name__ == "__main__":
    main()

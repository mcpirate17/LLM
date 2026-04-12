from __future__ import annotations

import torch
import torch.nn as nn

from research.eval.stateless_training import (
    clone_module_state,
    functional_micro_train_loop,
)
from research.tools.audit_multiscale_rich_lane_router import (
    build_multiscale_variant,
    summarize_results,
)


def test_build_multiscale_variant_preserves_three_tier_structure():
    graph = build_multiscale_variant(
        model_dim=64,
        span_widths=(2, 3, 4),
        medium_op="conv1d_seq",
        hard_op="moe_topk",
        route_temperature=0.85,
        min_keep_fraction=0.125,
        confidence_threshold=0.55,
    )
    ops = [node.op_name for node in graph.nodes.values()]
    assert ops.count("hybrid_sparse_router") == 3
    assert ops.count("sparse_span_builder") == 3
    assert "default_path" in ops
    assert "token_class_proj" in ops
    assert "signal_conditioned_compression" in ops


def test_summarize_results_selects_lowest_post_ppl_winner():
    results = {
        "span_ablation": [
            {
                "span_widths": [2],
                "medium_op": "conv1d_seq",
                "hard_op": "moe_topk",
                "post_ppl": 120.0,
                "train_final_loss": 4.0,
            },
            {
                "span_widths": [2, 3],
                "medium_op": "conv1d_seq",
                "hard_op": "moe_topk",
                "post_ppl": 100.0,
                "train_final_loss": 3.8,
            },
        ],
        "medium_ablation": [
            {
                "span_widths": [2, 3, 4],
                "medium_op": "adaptive_lane_mixer",
                "hard_op": "moe_topk",
                "post_ppl": 110.0,
                "train_final_loss": 3.9,
            },
        ],
        "hard_ablation": [
            {
                "span_widths": [2, 3, 4],
                "medium_op": "conv1d_seq",
                "hard_op": "dual_compression_blend",
                "post_ppl": 90.0,
                "train_final_loss": 3.7,
            },
        ],
    }

    summary = summarize_results(results)

    assert summary["best_span_variant"] == "[2, 3]"
    assert summary["best_medium_op"] == "adaptive_lane_mixer"
    assert summary["best_hard_op"] == "dual_compression_blend"
    assert summary["winner"]["hard_op"] == "dual_compression_blend"


def test_build_multiscale_variant_can_enable_phase2_curriculum_and_merge():
    graph = build_multiscale_variant(
        model_dim=64,
        span_widths=(2, 3, 4),
        medium_op="conv1d_seq",
        hard_op="mixed_recursion_gate",
        route_temperature=0.85,
        min_keep_fraction=0.125,
        confidence_threshold=0.55,
        enable_curriculum=True,
        use_calibrated_merge=True,
    )
    ops = [node.op_name for node in graph.nodes.values()]
    assert ops.count("calibrated_branch_merge") == 4
    hybrid_gate = next(
        node for node in graph.nodes.values() if node.op_name == "hybrid_token_gate"
    )
    assert hybrid_gate.config["curriculum_enabled"] is True


def test_build_multiscale_variant_accepts_curriculum_overrides():
    graph = build_multiscale_variant(
        model_dim=64,
        span_widths=(2, 3, 4),
        medium_op="conv_only",
        hard_op="mixed_recursion_gate",
        route_temperature=0.85,
        min_keep_fraction=0.125,
        confidence_threshold=0.55,
        enable_curriculum=True,
        use_calibrated_merge=True,
        gate_curriculum_overrides={
            "threshold_start": 0.4,
            "gate_temperature_start": 1.25,
        },
        router_curriculum_overrides={"confidence_threshold_start": 0.38},
        hard_curriculum_overrides={"active_depth_mid": 1},
        merge_curriculum_overrides={"routed_hard": {"min_secondary_share_start": 0.02}},
    )
    hybrid_gate = next(
        node for node in graph.nodes.values() if node.op_name == "hybrid_token_gate"
    )
    first_router = next(
        node for node in graph.nodes.values() if node.op_name == "hybrid_sparse_router"
    )
    merge_nodes = [
        node
        for node in graph.nodes.values()
        if node.op_name == "calibrated_branch_merge"
    ]
    hard_node = next(
        node for node in graph.nodes.values() if node.op_name == "mixed_recursion_gate"
    )
    assert hybrid_gate.config["threshold_start"] == 0.4
    assert hybrid_gate.config["gate_temperature_start"] == 1.25
    assert first_router.config["confidence_threshold_start"] == 0.38
    assert hard_node.config["active_depth_mid"] == 1
    assert any(
        node.config.get("min_secondary_share_start") == 0.02 for node in merge_nodes
    )


def test_functional_micro_train_loop_invokes_step_callback():
    model = nn.Sequential(nn.Embedding(32, 8), nn.Linear(8, 32))
    batches = [torch.randint(0, 32, (2, 6), dtype=torch.long)]
    params, buffers = clone_module_state(model)
    seen: list[tuple[int, int]] = []

    def _callback(step: int, total: int) -> None:
        seen.append((step, total))

    loss = functional_micro_train_loop(
        model,
        params,
        buffers,
        batches,
        vocab_size=32,
        n_steps=3,
        lr=1e-3,
        step_callback=_callback,
    )

    assert torch.isfinite(torch.tensor(loss))
    assert seen == [(0, 3), (1, 3), (2, 3)]

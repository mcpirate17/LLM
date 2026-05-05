from __future__ import annotations

import json

from research.scientist.llm.decision import (
    NextExperimentDecisionPlanner,
    NextExperimentPlannerConfig,
)
from research.scientist.runner.dashboard_hypothesis import _DashboardHypothesisMixin


def test_meta_profile_strategy_brief_loads_queue_and_builds_config_bias(tmp_path):
    queue = {
        "profile_refresh_queue": [
            {
                "op_name": "token_class_proj",
                "recommended_scaffold_family": "gpt2_replace",
                "action": "run_scaffold_profile",
            },
            {
                "op_name": "hybrid_sparse_router",
                "recommended_scaffold_family": "",
                "action": "add_scaffold_family_or_component_profile_harness",
            },
        ],
        "compression_safety_queue": [
            {
                "template_name": "token_merge_block",
                "selected_motif": "adjacent_token_merge",
                "nano_bind_rate": 0.4,
                "mean_frequency_risk": 0.75,
                "recommended_variant": (
                    "add_or_preserve_positional_or_content_mixer_after_compression"
                ),
            }
        ],
    }
    ml = {
        "summary": {
            "n_graphs": 100,
            "n_features": 65,
            "target_nano_bind_failure_rate": 0.03,
            "target_routing_improved_rate": 0.05,
        },
        "recommendations": [
            {
                "target": "target_nano_bind_failure",
                "feature": "external_tag_compression_count",
                "evidence": "importance=0.1",
                "recommendation": "validate compression with NanoBind",
            }
        ],
    }
    (tmp_path / "meta_experiment_queue_20260504.json").write_text(json.dumps(queue))
    (tmp_path / "meta_profile_ml_analysis_20260504.json").write_text(json.dumps(ml))

    brief = _DashboardHypothesisMixin._load_meta_profile_strategy_brief(tmp_path)

    assert brief["active"] is True
    assert brief["recommended_next_mode"] == "synthesis"
    assert brief["top_profile_refresh_ops"] == ["token_class_proj"]
    assert brief["needs_scaffold_harness_ops"] == ["hybrid_sparse_router"]
    assert brief["config_bias"]["op_weights"]["token_class_proj"] == 1.6
    assert brief["config_bias"]["op_weights"]["hybrid_sparse_router"] == 0.55
    assert brief["top_compression_safety_items"][0]["selected_motif"] == (
        "adjacent_token_merge"
    )


def test_planner_fallback_applies_meta_profile_strategy_bias():
    planner = NextExperimentDecisionPlanner(
        NextExperimentPlannerConfig(enabled=False, max_n_programs=200)
    )
    summary = {
        "recent_experiment_id": "exp1",
        "stage1_survivors": 0,
        "best_loss_ratio": None,
        "best_novelty": None,
        "meta_profile_strategy": {
            "active": True,
            "recommended_next_mode": "synthesis",
            "strategy_bias": "profile_refresh_guided_routing_compression_synthesis",
            "rationale": "Bias toward scaffoldable routing compression candidates.",
            "config_bias": {
                "n_programs": 80,
                "max_ops": 14,
                "op_weights": {
                    "token_class_proj": 1.6,
                    "hybrid_sparse_router": 0.55,
                },
                "category_weights": {"functional": 1.35},
            },
            "guardrails": {"no_hard_gates": True},
            "top_profile_refresh_ops": ["token_class_proj"],
        },
    }

    plan = planner.propose_plan(
        summary,
        fallback_plan={"mode": "novelty", "confidence": 0.41, "config": {}},
    )

    assert plan["mode"] == "synthesis"
    assert plan["confidence"] >= 0.62
    assert plan["config"]["n_programs"] == 80
    assert plan["config"]["op_weights"]["token_class_proj"] == 1.6
    assert plan["guardrails"]["meta_profile_strategy"]["no_hard_gates"] is True
    assert plan["meta_profile_strategy_used"] is True

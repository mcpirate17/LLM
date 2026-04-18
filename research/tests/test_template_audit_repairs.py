import random

from research.synthesis.graph import ComputationGraph
from research.synthesis.grammar import GrammarConfig, _validate_graph
from research.synthesis._template_helpers import get_slot_rule_summary
from research.synthesis.component_registry import fe_type_to_op_name
from research.synthesis.primitives import canonicalize_op_name
from research.scientist.runner._types import RunConfig
from research.scientist.runner.execution_screening import _make_stage1_screening_config
from research.scientist.runner.execution_training import _candidate_perf_budget_verdict
from research.synthesis.templates import (
    DEFAULT_TEMPLATE_WEIGHTS,
    RETIRED_TEMPLATE_NAMES,
    TEMPLATES,
    pick_template,
)
from research.synthesis.validator import validate_graph


def _finalize_template_graph(template_name: str, seed: int = 42) -> ComputationGraph:
    graph = ComputationGraph(model_dim=64)
    input_id = graph.add_input()
    out = TEMPLATES[template_name](graph, input_id, random.Random(seed), None)
    if graph.nodes[out].output_shape.dim != graph.model_dim:
        out = graph.add_op("linear_proj", [out], config={"out_dim": graph.model_dim})
    if graph.nodes[out].op_name not in ("rmsnorm", "layernorm"):
        out = graph.add_op("rmsnorm", [out])
    graph.set_output(out)
    graph.prune_unreachable_nodes()
    return graph


def test_template_exploration_excludes_retired_zero_weight_templates():
    rng = random.Random(123)
    seen = set()
    for _ in range(512):
        name, _fn, explored = pick_template(
            rng,
            weights=DEFAULT_TEMPLATE_WEIGHTS,
            exploration_budget=1.0,
        )
        assert explored is True
        seen.add(name)
    assert not (seen & RETIRED_TEMPLATE_NAMES)


def test_retrieval_v2_templates_build_screening_valid_graphs():
    for template_name in (
        "conv_residual_retrieval_v2",
        "state_space_retrieval_v2",
        "latent_attn_retrieval_v2",
    ):
        graph = _finalize_template_graph(template_name)
        result = validate_graph(graph, max_ops=24, max_depth=18)
        assert result.valid, f"{template_name} invalid: {result.errors}"


def test_hardened_templates_build_valid_graphs_across_seeds():
    for template_name in (
        "parallel_split",
        "moe",
        "induction_matmul_block",
        "multiscale_difficulty_router",
        "multiscale_rich_lane_router",
        "intelligent_multilane_router",
        "topk_retrieval",
        "mamba_reference",
        "dual_axis_block",
    ):
        for seed in (7, 42, 123):
            graph = _finalize_template_graph(template_name, seed=seed)
            result = validate_graph(graph, max_ops=24, max_depth=18)
            assert result.valid, f"{template_name}@{seed} invalid: {result.errors}"


def test_template_rule_violations_are_fatal_during_generation_validation():
    graph = ComputationGraph(model_dim=64)
    inp = graph.add_input()
    mid = graph.add_op("linear_proj", [inp], config={"out_dim": 64})
    graph.set_output(mid)

    try:
        _validate_graph(
            graph, GrammarConfig(model_dim=64, routing_mandatory=False)
        )
    except ValueError as exc:
        assert "Template rule violations" in str(exc)
        assert graph.metadata["template_rule_warnings"]
    else:
        raise AssertionError("template-invalid graph should be rejected")


def test_stage1_screening_config_enables_text_and_binding_probes():
    cfg = _make_stage1_screening_config(RunConfig())
    assert cfg.skip_screening_wikitext is False
    assert cfg.skip_binding_probes is False
    assert cfg.binding_probe_train_batch_size >= 1
    assert cfg.binding_probe_eval_batch_size >= 1


def test_graph_and_component_registry_canonicalize_alias_names():
    graph = ComputationGraph(model_dim=64)
    inp = graph.add_input()
    nid = graph.add_op("route_topk", [inp], config={"k": 4})
    assert graph.nodes[nid].op_name == canonicalize_op_name("route_topk")
    assert fe_type_to_op_name("routing/route_topk") == canonicalize_op_name(
        "route_topk"
    )


def test_candidate_perf_budget_verdict_uses_available_metrics_only():
    verdict = _candidate_perf_budget_verdict(
        {
            "trace_avg_ms": {
                "compile": 500.0,
                "forward_pass": 20.0,
            }
        }
    )
    assert verdict is not None
    assert verdict["partial"] is True
    failed = {check["metric"] for check in verdict["checks"] if not check["passed"]}
    assert "trace_avg_ms.compile" in failed


def test_slot_rule_summary_targets_live_declared_slots_only():
    legacy_no_telemetry = {"depth_token_mask_block"}
    built = {}
    missing = set()
    for row in get_slot_rule_summary():
        template_name = str(row["template_name"])
        assert template_name in TEMPLATES
        graph = built.get(template_name)
        if graph is None:
            graph = _finalize_template_graph(template_name)
            built[template_name] = graph
        slot_keys = {
            str(entry.get("slot_key_canonical"))
            for entry in graph.metadata.get("template_slot_usage") or []
        }
        if str(row["slot_key"]) not in slot_keys:
            missing.add(template_name)
    assert missing == legacy_no_telemetry

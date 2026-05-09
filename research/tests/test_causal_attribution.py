import json

import pytest

from research.scientist.causal_attribution import (
    causal_generation_adjustments,
    find_historical_ablation_observations,
    select_causal_ablation_candidates,
    summarize_ablation_effect,
)
from research.scientist.notebook import LabNotebook
from research.synthesis.graph import ComputationGraph
from research.synthesis.serializer import graph_to_json

pytestmark = pytest.mark.unit


def _full_s1_metrics(loss_ratio: float = 0.5) -> dict:
    """Synthetic S1-passing metrics that satisfy the universal completeness
    guardrail in notebook/program_writes.py. The user rule is 'never enter
    missing data', so test fixtures must mirror real S1 survivor rows."""
    return {
        "loss_ratio": loss_ratio,
        "wikitext_perplexity": 200.0,
        "hellaswag_acc": 0.25,
        "blimp_overall_accuracy": 0.5,
        "induction_screening_auc": 0.1,
        "binding_screening_auc": 0.1,
        "binding_screening_composite": 0.05,
        "ar_legacy_auc": 0.05,
    }


def _graph_with_slot_usage() -> ComputationGraph:
    graph = ComputationGraph(model_dim=32)
    x = graph.add_input()
    norm = graph.add_op("rmsnorm", [x])
    sparse = graph.add_op("sparse_threshold", [norm])
    out = graph.add_op("add", [norm, sparse])
    graph.set_output(out)
    graph.metadata["templates_used"] = ["conditional_compute"]
    graph.metadata["template_slot_usage"] = [
        {
            "template_name": "conditional_compute",
            "template_instance": 0,
            "slot_index": 1,
            "slot_key": "conditional_compute[0].slot1",
            "selected_motif": "sparse_block",
            "selected_motif_class": "sparse_core",
            "wildcard": False,
        }
    ]
    return graph


def test_select_causal_ablation_candidates_prefers_slot_signals(tmp_path):
    nb = LabNotebook(str(tmp_path / "causal.db"))
    try:
        exp_id = nb.start_experiment("synthesis", {}, "causal candidate test")
        graph = _graph_with_slot_usage()
        result_id = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint=graph.fingerprint(),
            graph_json=graph_to_json(graph),
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            bypass_quality_gate=True,
            **_full_s1_metrics(loss_ratio=0.42),
        )
        candidates = select_causal_ablation_candidates(
            nb,
            experiment_id=exp_id,
            max_survivors=1,
            max_signals_per_survivor=2,
        )
        assert candidates
        assert candidates[0].parent_result_id == result_id
        assert candidates[0].rule_type == "slot_motif"
        assert "sparse_block" in candidates[0].rule_key
        assert "ops:" in candidates[0].hypothesis
    finally:
        nb.close()


def test_causal_evidence_summary_and_generation_adjustments(tmp_path):
    nb = LabNotebook(str(tmp_path / "causal_adjust.db"))
    try:
        parent_exp = nb.start_experiment("synthesis", {}, "parent")
        graph = _graph_with_slot_usage()
        parent_result = nb.record_program_result(
            experiment_id=parent_exp,
            graph_fingerprint=graph.fingerprint(),
            graph_json=graph_to_json(graph),
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            bypass_quality_gate=True,
            **_full_s1_metrics(loss_ratio=0.40),
        )
        candidate = select_causal_ablation_candidates(
            nb,
            experiment_id=parent_exp,
            max_survivors=1,
            max_signals_per_survivor=1,
        )[0]

        ab_exp = nb.start_experiment("ablation", {}, "ablate sparse")
        ab_graph = _graph_with_slot_usage()
        nb.record_program_result(
            experiment_id=ab_exp,
            graph_fingerprint=ab_graph.fingerprint() + "-ab",
            graph_json=graph_to_json(ab_graph),
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            bypass_quality_gate=True,
            intentional_rerun_reason="test_ablation",
            **_full_s1_metrics(loss_ratio=0.55),
        )
        evidence = summarize_ablation_effect(
            nb,
            candidate=candidate,
            ablation_experiment_ids=[ab_exp],
            ablation_outcome="not_supported",
        )
        assert evidence["parent_result_id"] == parent_result
        assert evidence["outcome"] == "supported"
        assert evidence["effect_size"] == pytest.approx(0.15)
        evidence_id = nb.record_causal_rule_evidence(evidence)
        rows = nb.get_causal_rule_evidence(result_id=parent_result)
        assert rows[0]["evidence_id"] == evidence_id
        assert json.loads(rows[0]["rule_context"])["selected_motif"] == "sparse_block"

        adjustments = causal_generation_adjustments(nb, min_confidence=0.1)
        slot_key = "conditional_compute[0].slot1"
        assert adjustments["slot_motif_multipliers"][slot_key]["sparse_block"] > 1.0
    finally:
        nb.close()


def test_historical_ablation_observations_are_recorded_with_provenance(tmp_path):
    nb = LabNotebook(str(tmp_path / "causal_history.db"))
    try:
        parent_exp = nb.start_experiment("synthesis", {}, "parent")
        graph = _graph_with_slot_usage()
        parent_result = nb.record_program_result(
            experiment_id=parent_exp,
            graph_fingerprint=graph.fingerprint(),
            graph_json=graph_to_json(graph),
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            bypass_quality_gate=True,
            **_full_s1_metrics(loss_ratio=0.40),
        )
        candidate = select_causal_ablation_candidates(
            nb,
            experiment_id=parent_exp,
            max_survivors=1,
            max_signals_per_survivor=1,
        )[0]

        historical_exp = nb.start_experiment("synthesis", {}, "historical child")
        historical_result = nb.record_program_result(
            experiment_id=historical_exp,
            graph_fingerprint=graph.fingerprint(),
            graph_json=graph_to_json(graph),
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            bypass_quality_gate=True,
            intentional_rerun_reason="historical_counterfactual_fixture",
            **_full_s1_metrics(loss_ratio=0.49),
        )
        observations = find_historical_ablation_observations(
            nb,
            candidate=candidate,
            graphs=[graph],
        )
        assert len(observations) == 1
        assert observations[0]["child_result_id"] == historical_result
        assert observations[0]["source"] == "historical"

        evidence = summarize_ablation_effect(
            nb,
            candidate=candidate,
            ablation_experiment_ids=[],
            ablation_outcome="historical_reuse",
            child_observations=observations,
        )
        assert evidence["parent_result_id"] == parent_result
        assert evidence["outcome"] == "supported"
        evidence_id = nb.record_causal_rule_evidence(evidence)
        inserted = nb.record_causal_ablation_child_observations(
            evidence_id,
            observations,
        )
        assert inserted == 1
        child_rows = nb.get_causal_ablation_child_observations(evidence_id=evidence_id)
        assert child_rows[0]["child_result_id"] == historical_result
        assert child_rows[0]["provenance"]["source"] == "historical"
    finally:
        nb.close()


def test_causal_generation_adjustments_accumulates_repeated_weak_evidence(tmp_path):
    nb = LabNotebook(str(tmp_path / "causal_accumulated.db"))
    try:
        parent_exp = nb.start_experiment("synthesis", {}, "parent")
        graph = _graph_with_slot_usage()
        parent_result = nb.record_program_result(
            experiment_id=parent_exp,
            graph_fingerprint=graph.fingerprint(),
            graph_json=graph_to_json(graph),
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            bypass_quality_gate=True,
            **_full_s1_metrics(loss_ratio=0.40),
        )
        for idx in range(5):
            nb.record_causal_rule_evidence(
                {
                    "evidence_id": f"weak-{idx}",
                    "parent_experiment_id": parent_exp,
                    "parent_result_id": parent_result,
                    "parent_fingerprint": graph.fingerprint(),
                    "ablation_experiment_id": f"ab-{idx}",
                    "rule_type": "op_pair",
                    "rule_key": "route_lanes->linear_proj",
                    "rule_context": "{}",
                    "original_loss_ratio": 0.40,
                    "ablation_best_loss_ratio": 0.43,
                    "effect_size": 0.03,
                    "original_stage1_passed": 1,
                    "ablation_stage1_pass_count": 1,
                    "ablation_total": 1,
                    "outcome": "supported",
                    "confidence": 0.08,
                    "evidence_json": "{}",
                }
            )

        adjustments = causal_generation_adjustments(nb, min_confidence=0.35)
        assert adjustments["op_weights"]["route_lanes"] > 1.0
    finally:
        nb.close()

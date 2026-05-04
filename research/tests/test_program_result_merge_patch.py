from __future__ import annotations

import json
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from research.scientist.notebook import LabNotebook
from research.scientist.leaderboard_scoring import build_score_kwargs, compute_composite
from research.scientist.llm.context_experiment import build_experiment_context
from research.scientist.persona import Aria
from research.scientist.runner._helpers import program_result_kwargs_from_s1
from research.scientist.runner._helpers_benchmark import (
    finalize_validation_results_summary,
    promote_validation_candidate,
)
from research.scientist.runner.execution_screening import _record_screening_failure


class _FakeNode:
    def __init__(self, op_name: str, *, is_input: bool = False) -> None:
        self.op_name = op_name
        self.is_input = is_input
        self.is_output = False


class _FakeGraph:
    def __init__(self, fp: str) -> None:
        self._fp = fp
        self.nodes = {
            "0": _FakeNode("linear_proj"),
            "1": _FakeNode("gelu"),
        }

    def fingerprint(self) -> str:
        return self._fp

    def n_ops(self) -> int:
        return len(self.nodes)

    def has_gradient_path(self) -> bool:
        return True

    def to_dict(self) -> dict:
        return {"nodes": {"0": {"op_name": "linear_proj"}, "1": {"op_name": "gelu"}}}


def _validation_ev_res(**overrides):
    base = dict(
        is_breakthrough=False,
        flop_gated=False,
        ood_result=None,
        sensitivity_result=None,
        quant_int8_retention=None,
        quant_quality_per_byte=None,
        long_context_score=None,
        long_ctx_scaling_score=None,
        long_ctx_assoc_score=None,
        long_ctx_passkey_score=None,
        long_ctx_multi_hop_score=None,
        long_ctx_retrieval_aggregate=None,
        long_ctx_combined_score=None,
        induction_v2_investigation_auc=None,
        induction_v2_investigation_max_gap_acc=None,
        induction_v2_investigation_protocol_version=None,
        binding_v2_investigation_auc=None,
        binding_v2_investigation_max_distance_acc=None,
        binding_v2_investigation_protocol_version=None,
        permutation_composition_score=None,
        permutation_composition_train_chain_acc=None,
        permutation_composition_extrapolation_acc=None,
        permutation_composition_n_items=None,
        permutation_composition_train_chain_len=None,
        permutation_composition_eval_chain_len=None,
        permutation_composition_train_steps=None,
        permutation_composition_chance=None,
        permutation_composition_elapsed_ms=None,
        permutation_composition_status=None,
        permutation_composition_metric_version=None,
        noise_score=0.08,
        scaling_param_efficiency=None,
        scaling_d512_param_efficiency=None,
        scaling_flop_efficiency=None,
        scaling_gate_passed_val=1,
        scaling_best_family=None,
        scaling_confidence=None,
        activation_sparsity_score=0.21,
        dead_neuron_ratio=0.03,
        routing_collapse_score=0.02,
        wikitext_perplexity=7.2,
        wikitext_score=0.63,
        tinystories_perplexity=6.9,
        tinystories_score=0.61,
        cross_task_score=0.57,
        efficiency_wall_score=0.55,
        max_viable_seq_len=256,
        scaling_regime="stable",
        scaling_result=None,
        long_context_details=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _stage1_kwargs(
    *,
    loss_ratio: float = 0.42,
    novelty_score: float = 0.66,
    model_source: str = "graph_synthesis",
    **extra,
) -> dict:
    return program_result_kwargs_from_s1(
        {
            "passed": True,
            "final_loss": 4.5,
            "loss_ratio": loss_ratio,
            "wikitext_perplexity": 150.0,
            "wikitext_score": 0.55,
            "screening_wikitext_metric_version": "unit_test_wikitext_v1",
            "hellaswag_acc": 0.31,
            "hellaswag_status": "ran",
            "blimp_overall_accuracy": 0.55,
            "blimp_status": "ran",
            "induction_auc": 0.21,
            "binding_auc": 0.18,
            "binding_composite": 0.12,
            "ar_auc": 0.06,
        },
        model_source=model_source,
        extra={
            "stage1_passed": True,
            "novelty_score": novelty_score,
            "data_mode": "random",
            "tokenizer_mode": "byte",
            "vocab_size": 256,
            **extra,
        },
    )


def _mark_promotable(nb: LabNotebook, result_id: str) -> None:
    nb.conn.execute(
        """
        UPDATE program_results
        SET trust_label = ?, comparability_label = ?, data_provenance_json = ?
        WHERE result_id = ?
        """,
        (
            "candidate_grade",
            "candidate_comparable",
            json.dumps({"eligible_for_promotion": True, "provenance_complete": True}),
            result_id,
        ),
    )
    nb.conn.commit()


def test_merge_program_result_patch_clears_failure_when_stage1_recovers():
    with tempfile.TemporaryDirectory() as tmpdir:
        nb = LabNotebook(f"{tmpdir}/merge.db")
        exp_id = nb.start_experiment("synthesis", {}, "merge patch")
        rid = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_merge_success",
            graph_json="{}",
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=False,
            error_type="shape_mismatch",
            error_message="bad tensor shape",
            stage_at_death="stage1",
            loss_ratio=0.61,
        )
        nb.flush_writes()

        nb.merge_program_result_patch(
            result_id=rid,
            clear_failure_if_stage1=True,
            **_stage1_kwargs(loss_ratio=0.41, final_loss=1.8),
        )
        nb.flush_writes()

        row = nb.get_program_detail(rid)
        assert row is not None
        assert int(row["stage1_passed"] or 0) == 1
        assert row["loss_ratio"] == 0.41
        assert row["error_type"] is None
        assert row["error_message"] is None
        assert row["stage_at_death"] is None
        nb.close()


def test_sync_behavioral_fingerprint_result_updates_top_level_fields():
    with tempfile.TemporaryDirectory() as tmpdir:
        nb = LabNotebook(f"{tmpdir}/fingerprint_sync.db")
        exp_id = nb.start_experiment("synthesis", {}, "fingerprint sync")
        rid = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_sync_top_level",
            graph_json="{}",
            stage0_passed=True,
            stage05_passed=True,
            **_stage1_kwargs(
                novelty_score=0.28,
                novelty_confidence=0.35,
                cka_source="deferred",
                novelty_valid_for_promotion=0,
                novelty_validity_reason="cka_deferred_post_investigation",
                fp_jacobian_spectral_norm=43610.9,
                fingerprint_json=json.dumps({"novelty_score": 0.28}),
            ),
        )
        nb.flush_writes()
        nb.upsert_leaderboard(
            result_id=rid,
            model_source="graph_synthesis",
            screening_loss_ratio=0.42,
            screening_novelty=0.28,
            screening_passed=True,
            tier="screening",
            novelty_confidence=0.35,
            fp_jacobian_spectral_norm=43610.9,
        )
        nb.flush_writes()

        fp_payload = {
            "novelty_score": 0.794569122294585,
            "quality": "partial",
            "analyses_succeeded": 3,
            "cka_source": "none",
            "cka_artifact_version": None,
            "cka_probe_protocol_hash": None,
            "cka_reference_quality": "none",
            "novelty_valid_for_promotion": False,
            "novelty_validity_reason": "no_reference_available",
            "novelty_reference_version": "nv1:none",
            "jacobian_spectral_norm": 5518.1201171875,
            "jacobian_effective_rank": 4.2,
            "sensitivity_uniformity": 0.6,
            "interaction_locality": 0.1,
            "interaction_sparsity": 0.2,
            "interaction_symmetry": 0.3,
            "interaction_hierarchy": 0.4,
            "intrinsic_dim": 5.0,
            "isotropy": 0.7,
            "rank_ratio": 0.8,
            "cka_vs_transformer": 0.0,
            "cka_vs_ssm": 0.0,
            "cka_vs_conv": 0.0,
            "hierarchy_fitness": 0.15,
            "gromov_delta": 0.25,
            "fingerprint_completed_post_investigation": True,
        }

        changed = nb.sync_behavioral_fingerprint_result(
            result_id=rid,
            fp_payload=fp_payload,
        )
        nb.flush_writes()

        assert changed is True
        row = nb.get_program_detail(rid)
        assert row is not None
        assert row["novelty_score"] == fp_payload["novelty_score"]
        assert row["novelty_confidence"] == pytest.approx(0.7)
        assert row["cka_source"] == "none"
        assert row["novelty_validity_reason"] == "no_reference_available"
        assert row["fp_jacobian_spectral_norm"] == fp_payload["jacobian_spectral_norm"]

        entry = nb.conn.execute(
            "SELECT screening_novelty, fp_jacobian_spectral_norm "
            "FROM leaderboard WHERE result_id = ?",
            (rid,),
        ).fetchone()
        assert entry is not None
        assert entry["screening_novelty"] == fp_payload["novelty_score"]
        assert (
            entry["fp_jacobian_spectral_norm"] == fp_payload["jacobian_spectral_norm"]
        )
        nb.close()


def test_record_program_result_canonicalizes_fingerprint_json_from_explicit_novelty():
    with tempfile.TemporaryDirectory() as tmpdir:
        nb = LabNotebook(f"{tmpdir}/fingerprint_insert_sync.db")
        exp_id = nb.start_experiment("synthesis", {}, "fingerprint insert sync")
        rid = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_insert_sync",
            graph_json="{}",
            **_stage1_kwargs(
                novelty_score=0.62,
                novelty_confidence=0.71,
                cka_source="deferred",
                novelty_validity_reason="cka_deferred_post_investigation",
                fp_jacobian_spectral_norm=12.5,
                fingerprint_json=json.dumps(
                    {
                        "novelty_score": 0.0,
                        "cka_source": "none",
                        "novelty_validity_reason": "missing_reference",
                        "jacobian_spectral_norm": 0.0,
                        "quality": "partial",
                        "analyses_succeeded": 3,
                    }
                ),
            ),
        )
        nb.flush_writes()

        row = nb.get_program_detail(rid)
        assert row is not None
        fp_payload = json.loads(row["fingerprint_json"])
        assert row["novelty_score"] == 0.62
        assert fp_payload["novelty_score"] == 0.62
        assert fp_payload["cka_source"] == "deferred"
        assert (
            fp_payload["novelty_validity_reason"] == "cka_deferred_post_investigation"
        )
        assert fp_payload["jacobian_spectral_norm"] == 12.5
        nb.close()


def test_merge_program_result_patch_keeps_fingerprint_json_synced_with_novelty_updates():
    with tempfile.TemporaryDirectory() as tmpdir:
        nb = LabNotebook(f"{tmpdir}/fingerprint_merge_sync.db")
        exp_id = nb.start_experiment("validation", {}, "fingerprint merge sync")
        rid = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_merge_sync",
            graph_json="{}",
            **_stage1_kwargs(
                novelty_score=0.4,
                novelty_confidence=0.8,
                fingerprint_json=json.dumps(
                    {
                        "novelty_score": 0.4,
                        "quality": "partial",
                        "analyses_succeeded": 4,
                        "cka_source": "deferred",
                        "novelty_validity_reason": "cka_deferred_post_investigation",
                    }
                ),
            ),
        )
        nb.flush_writes()

        changed = nb.merge_program_result_patch(
            result_id=rid,
            novelty_score=0.5,
            novelty_confidence=0.9,
        )
        nb.flush_writes()

        assert changed is True
        row = nb.get_program_detail(rid)
        assert row is not None
        fp_payload = json.loads(row["fingerprint_json"])
        assert row["novelty_score"] == 0.5
        assert row["novelty_confidence"] == 0.9
        assert fp_payload["novelty_score"] == 0.5
        nb.close()


def test_merge_program_result_patch_persists_permutation_composition_fields():
    with tempfile.TemporaryDirectory() as tmpdir:
        nb = LabNotebook(f"{tmpdir}/permutation_merge.db")
        exp_id = nb.start_experiment("validation", {}, "permutation merge")
        rid = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_permutation_merge",
            graph_json="{}",
            **_stage1_kwargs(),
        )
        nb.flush_writes()

        changed = nb.merge_program_result_patch(
            result_id=rid,
            permutation_composition_score=0.42,
            permutation_composition_train_chain_acc=0.55,
            permutation_composition_extrapolation_acc=0.31,
            permutation_composition_n_items=8,
            permutation_composition_train_chain_len=2,
            permutation_composition_eval_chain_len=4,
            permutation_composition_train_steps=40,
            permutation_composition_chance=0.125,
            permutation_composition_elapsed_ms=123.4,
            permutation_composition_status="ok",
            permutation_composition_metric_version="permutation_composition_v1",
        )
        nb.flush_writes()

        assert changed is True
        row = nb.get_program_detail(rid)
        assert row is not None
        assert row["permutation_composition_score"] == 0.42
        assert row["permutation_composition_train_chain_acc"] == 0.55
        assert row["permutation_composition_extrapolation_acc"] == 0.31
        assert row["permutation_composition_n_items"] == 8
        assert row["permutation_composition_train_chain_len"] == 2
        assert row["permutation_composition_eval_chain_len"] == 4
        assert row["permutation_composition_train_steps"] == 40
        assert row["permutation_composition_chance"] == 0.125
        assert row["permutation_composition_elapsed_ms"] == 123.4
        assert row["permutation_composition_status"] == "ok"
        assert (
            row["permutation_composition_metric_version"]
            == "permutation_composition_v1"
        )
        nb.close()


def test_record_screening_failure_merges_into_source_row_without_duplicate():
    with tempfile.TemporaryDirectory() as tmpdir:
        nb = LabNotebook(f"{tmpdir}/replay.db")
        exp_id = nb.start_experiment("forced_exploration", {}, "source row")
        rid = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_replay_merge",
            graph_json="{}",
            stage0_passed=False,
            stage05_passed=False,
            stage1_passed=False,
            error_type="compile_error",
        )
        nb.flush_writes()

        with patch(
            "research.scientist.runner.execution_screening.infer_graph_failure_provenance",
            return_value={
                "failure_op": "linear_proj",
                "failure_details_json": '{"root_cause_code":"rapid_screening_error"}',
            },
        ):
            _record_screening_failure(
                nb=nb,
                exp_id=exp_id,
                source_result_id=rid,
                graph=_FakeGraph("fp_replay_merge"),
                stage0_passed=True,
                stage05_passed=True,
                error_type="rapid_screening_error",
                error_message="killed by replay",
                stage_at_death="rapid_screening",
                stability_score=0.44,
                extra_metrics={
                    "rapid_screening_passed": 0,
                    "wikitext_score": 0.52,
                },
            )
        nb.flush_writes()

        count = nb.conn.execute(
            "SELECT COUNT(*) AS n FROM program_results WHERE graph_fingerprint = ?",
            ("fp_replay_merge",),
        ).fetchone()
        row = nb.get_program_detail(rid)
        assert count["n"] == 1
        assert row is not None
        assert row["error_type"] == "rapid_screening_error"
        assert row["stage_at_death"] == "rapid_screening"
        assert row["result_cohort"] == "backfill"
        assert row["trust_label"] == "backfill_observation"
        assert row["comparability_label"] == "reconstructed_init_variant"
        nb.close()


def test_promote_validation_candidate_updates_source_row_without_duplicate():
    with tempfile.TemporaryDirectory() as tmpdir:
        nb = LabNotebook(f"{tmpdir}/validation.db")
        exp_id = nb.start_experiment("synthesis", {}, "validation source")
        rid = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_validation_source",
            graph_json="{}",
            stage0_passed=True,
            stage05_passed=True,
            **_stage1_kwargs(
                loss_ratio=0.42,
                novelty_score=0.66,
                novelty_confidence=0.71,
                fp_jacobian_spectral_norm=1.3,
            ),
        )
        nb.flush_writes()
        nb.upsert_leaderboard(
            result_id=rid,
            model_source="graph_synthesis",
            screening_loss_ratio=0.42,
            screening_novelty=0.66,
            screening_passed=True,
            tier="screening",
        )
        _mark_promotable(nb, rid)

        metrics = SimpleNamespace(
            val_loss_ratio=0.31,
            val_baseline_ratio=0.88,
            val_normalized_ratio=0.84,
            val_param_efficiency=0.11,
            multi_seed_std=0.03,
            robustness_score=0.79,
            is_unstable=False,
            passed_seeds=[1, 2, 3],
            init_sensitivity_std=0.05,
        )
        promote_validation_candidate(
            nb=nb,
            source_result_id=rid,
            source={
                "novelty_score": 0.66,
                "novelty_confidence": 0.71,
                "fp_jacobian_spectral_norm": 1.3,
            },
            tier="validation",
            metrics=metrics,
            ev_res=_validation_ev_res(),
        )
        nb.flush_writes()

        count = nb.conn.execute(
            "SELECT COUNT(*) AS n FROM program_results WHERE graph_fingerprint = ?",
            ("fp_validation_source",),
        ).fetchone()
        row = nb.conn.execute(
            """
            SELECT validation_loss_ratio, wikitext_perplexity
            FROM program_results
            WHERE result_id = ?
            """,
            (rid,),
        ).fetchone()
        lb = nb.get_leaderboard_entry(rid)
        assert count["n"] == 1
        assert row is not None
        assert row["validation_loss_ratio"] == 0.31
        assert row["wikitext_perplexity"] == 7.2
        assert lb is not None
        assert lb["tier"] == "validation"
        assert lb["validation_baseline_ratio"] == 0.88
        assert lb["validation_multi_seed_std"] == 0.03
        assert int(lb["validation_passed"] or 0) == 1
        nb.close()


def test_validation_experiment_programs_include_saved_source_views():
    with tempfile.TemporaryDirectory() as tmpdir:
        nb = LabNotebook(f"{tmpdir}/validation_views.db")
        source_exp = nb.start_experiment("synthesis", {}, "validation source")
        rid = nb.record_program_result(
            experiment_id=source_exp,
            graph_fingerprint="fp_validation_view",
            graph_json='{"nodes": {}}',
            stage0_passed=True,
            stage05_passed=True,
            **_stage1_kwargs(loss_ratio=0.42, novelty_score=0.66),
        )
        nb.flush_writes()

        val_exp = nb.start_experiment("validation", {}, "validation run")
        results = {
            "total": 1,
            "stage0_passed": 1,
            "stage05_passed": 1,
            "stage1_passed": 1,
            "best_loss_ratio": 0.31,
            "best_novelty_score": 0.66,
            "survivors": [],
            "validation_results": [
                {
                    "result_id": rid,
                    "source_experiment_id": source_exp,
                    "graph_fingerprint": "fp_validation_view",
                    "novelty_score": 0.66,
                    "novelty_confidence": 0.71,
                    "val_loss_ratio": 0.31,
                    "val_baseline_ratio": 0.88,
                    "val_normalized_ratio": 0.84,
                    "multi_seed_std": 0.03,
                    "robustness_score": 0.79,
                    "is_unstable": False,
                    "seeds_passed": 5,
                    "total_seeds": 5,
                    "is_breakthrough": True,
                    "tier": "breakthrough",
                }
            ],
        }
        finalize_validation_results_summary(results)
        nb.complete_experiment(val_exp, results=results)

        programs = nb.get_program_results(val_exp)
        assert len(programs) == 1
        row = programs[0]
        assert row["result_id"] == rid
        assert row["experiment_id"] == val_exp
        assert row["source_experiment_id"] == source_exp
        assert row["validation_experiment_id"] == val_exp
        assert row["is_validation_result_view"] is True
        assert row["tier"] == "breakthrough"
        assert row["validation_loss_ratio"] == 0.31
        assert row["validation_is_breakthrough"] is True
        nb.close()


def test_validation_summary_context_uses_structured_breakthrough_counts():
    results = {
        "total": 1,
        "stage0_passed": 1,
        "stage05_passed": 1,
        "stage1_passed": 1,
        "best_loss_ratio": 0.31,
        "best_novelty_score": 0.66,
        "survivors": [],
        "validation_results": [
            {
                "result_id": "abc123-validation",
                "novelty_score": 0.66,
                "novelty_confidence": 0.71,
                "val_loss_ratio": 0.31,
                "val_baseline_ratio": 0.88,
                "multi_seed_std": 0.03,
                "robustness_score": 0.79,
                "seeds_passed": 5,
                "total_seeds": 5,
                "is_breakthrough": True,
            }
        ],
    }
    finalize_validation_results_summary(results)

    context = build_experiment_context(results)
    assert "1 breakthrough" in context
    assert "1 with novelty_score > 0.5" in context

    summary = Aria().experiment_summary(results, context=context)
    assert "Breakthrough candidates:    1" in summary
    assert "Novel validated candidates: 1" in summary


def test_promote_validation_candidate_novelty_cap_keeps_fingerprint_json_synced():
    with tempfile.TemporaryDirectory() as tmpdir:
        nb = LabNotebook(f"{tmpdir}/validation_cap_sync.db")
        exp_id = nb.start_experiment("synthesis", {}, "validation cap source")
        rid = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_validation_cap",
            graph_json="{}",
            stage0_passed=True,
            stage05_passed=True,
            **_stage1_kwargs(
                loss_ratio=0.42,
                novelty_score=0.66,
                novelty_confidence=0.71,
                cka_source="deferred",
                novelty_validity_reason="cka_deferred_post_investigation",
                fingerprint_json=json.dumps(
                    {
                        "novelty_score": 0.66,
                        "quality": "partial",
                        "analyses_succeeded": 3,
                        "cka_source": "deferred",
                        "novelty_validity_reason": "cka_deferred_post_investigation",
                    }
                ),
            ),
        )
        nb.flush_writes()
        nb.upsert_leaderboard(
            result_id=rid,
            model_source="graph_synthesis",
            screening_loss_ratio=0.42,
            screening_novelty=0.66,
            screening_passed=True,
            tier="screening",
        )
        _mark_promotable(nb, rid)

        metrics = SimpleNamespace(
            val_loss_ratio=0.31,
            val_baseline_ratio=0.88,
            val_normalized_ratio=0.84,
            val_param_efficiency=0.11,
            multi_seed_std=0.03,
            robustness_score=0.79,
            is_unstable=False,
            passed_seeds=[1, 2, 3],
            init_sensitivity_std=0.05,
        )
        promote_validation_candidate(
            nb=nb,
            source_result_id=rid,
            source=nb.get_program_detail(rid),
            tier="validation",
            metrics=metrics,
            ev_res=_validation_ev_res(),
            novelty_cap=0.5,
        )
        nb.flush_writes()

        row = nb.get_program_detail(rid)
        assert row is not None
        fp_payload = json.loads(row["fingerprint_json"])
        assert row["novelty_score"] == pytest.approx(0.33)
        assert row["novelty_confidence"] == pytest.approx(0.355)
        assert fp_payload["novelty_score"] == pytest.approx(0.33)
        nb.close()


def test_promote_validation_candidate_uses_fingerprint_canonical_row():
    with tempfile.TemporaryDirectory() as tmpdir:
        nb = LabNotebook(f"{tmpdir}/validation_fingerprint.db")

        screen_exp = nb.start_experiment("synthesis", {}, "screening source")
        canonical_rid = nb.record_program_result(
            experiment_id=screen_exp,
            graph_fingerprint="fp_validation_fingerprint",
            graph_json="{}",
            stage0_passed=True,
            stage05_passed=True,
            **_stage1_kwargs(
                loss_ratio=0.41,
                novelty_score=0.63,
                novelty_confidence=0.72,
                fp_jacobian_spectral_norm=1.4,
            ),
        )
        _mark_promotable(nb, canonical_rid)
        nb.upsert_leaderboard(
            result_id=canonical_rid,
            model_source="graph_synthesis",
            screening_loss_ratio=0.41,
            screening_novelty=0.63,
            screening_passed=True,
            tier="screening",
        )

        inv_exp = nb.start_experiment("investigation", {}, "investigation child")
        child_rid = nb.record_program_result(
            experiment_id=inv_exp,
            graph_fingerprint="fp_validation_fingerprint",
            graph_json="{}",
            stage0_passed=True,
            stage05_passed=True,
            **_stage1_kwargs(
                loss_ratio=0.33,
                novelty_score=0.63,
                novelty_confidence=0.72,
                fp_jacobian_spectral_norm=1.4,
            ),
        )
        _mark_promotable(nb, child_rid)
        nb.flush_writes()

        metrics = SimpleNamespace(
            val_loss_ratio=0.29,
            val_baseline_ratio=0.83,
            val_normalized_ratio=0.8,
            val_param_efficiency=0.12,
            multi_seed_std=0.02,
            robustness_score=0.91,
            is_unstable=False,
            passed_seeds=[1, 2, 3, 4, 5],
            init_sensitivity_std=0.04,
        )
        promote_validation_candidate(
            nb=nb,
            source_result_id=child_rid,
            source=nb.get_program_detail(child_rid),
            tier="validation",
            metrics=metrics,
            ev_res=_validation_ev_res(
                scaling_result={"score": 0.77},
                long_context_details={"multi_hop": {"score": 0.69}},
            ),
        )
        nb.flush_writes()

        child_row = nb.conn.execute(
            """
            SELECT validation_loss_ratio, baseline_loss_ratio,
                   wikitext_perplexity, external_benchmarks_json
            FROM program_results
            WHERE result_id = ?
            """,
            (child_rid,),
        ).fetchone()
        canonical_lb = nb.get_leaderboard_entry(canonical_rid)
        canonical_row = nb.conn.execute(
            """
            SELECT external_benchmarks_json
            FROM program_results
            WHERE result_id = ?
            """,
            (canonical_rid,),
        ).fetchone()
        leaderboard_count = nb.conn.execute(
            """
            SELECT COUNT(*) AS n
            FROM leaderboard l
            JOIN program_results pr ON pr.result_id = l.result_id
            WHERE pr.graph_fingerprint = ?
            """,
            ("fp_validation_fingerprint",),
        ).fetchone()

        assert child_row is not None
        assert child_row["validation_loss_ratio"] == 0.29
        assert child_row["baseline_loss_ratio"] == 0.83
        assert child_row["wikitext_perplexity"] == 7.2
        assert canonical_lb is not None
        assert canonical_lb["tier"] == "validation"
        assert canonical_lb["result_id"] == canonical_rid
        assert canonical_lb["validation_loss_ratio"] == 0.29
        assert canonical_lb["validation_baseline_ratio"] == 0.83
        assert int(canonical_lb["validation_passed"] or 0) == 1
        assert leaderboard_count["n"] == 1
        assert json.loads(child_row["external_benchmarks_json"])["score"] == 0.77
        assert json.loads(canonical_row["external_benchmarks_json"])["score"] == 0.77
        nb.close()


def test_sync_fingerprint_leaderboard_preserves_child_validation_evidence():
    with tempfile.TemporaryDirectory() as tmpdir:
        nb = LabNotebook(f"{tmpdir}/fingerprint_sync.db")

        screen_exp = nb.start_experiment("synthesis", {}, "screening anchor")
        anchor_rid = nb.record_program_result(
            experiment_id=screen_exp,
            graph_fingerprint="fp_sync_validation",
            graph_json="{}",
            **_stage1_kwargs(loss_ratio=0.46, novelty_score=0.58),
        )
        nb.upsert_leaderboard(
            result_id=anchor_rid,
            model_source="graph_synthesis",
            screening_loss_ratio=0.46,
            screening_novelty=0.58,
            screening_passed=True,
            tier="screening",
        )

        val_exp = nb.start_experiment("validation", {}, "child validation")
        child_rid = nb.record_program_result(
            experiment_id=val_exp,
            graph_fingerprint="fp_sync_validation",
            graph_json="{}",
            **_stage1_kwargs(
                loss_ratio=0.32,
                validation_loss_ratio=0.27,
                baseline_loss_ratio=0.81,
                validation_multi_seed_std=0.019,
                validation_passed=True,
            ),
        )
        nb.flush_writes()

        nb._sync_fingerprint_leaderboard(child_rid)

        row = nb.get_leaderboard_entry(anchor_rid)
        assert row is not None
        assert row["tier"] == "validation"
        assert row["validation_loss_ratio"] == 0.27
        assert row["validation_baseline_ratio"] == 0.81
        expected = compute_composite(
            **build_score_kwargs(nb.conn, nb, anchor_rid, dict(row), False)
        )
        assert row["composite_score"] == expected
        nb.close()


def test_sync_fingerprint_leaderboard_preserves_investigation_tier_without_pass():
    with tempfile.TemporaryDirectory() as tmpdir:
        nb = LabNotebook(f"{tmpdir}/fingerprint_sync_investigation.db")

        exp_id = nb.start_experiment("investigation", {}, "investigation anchor")
        rid = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_sync_investigation",
            graph_json="{}",
            **_stage1_kwargs(
                loss_ratio=0.58,
                novelty_score=0.61,
                trust_label="test_fixture",
            ),
        )
        _mark_promotable(nb, rid)
        nb.upsert_leaderboard(
            result_id=rid,
            model_source="graph_synthesis",
            tier="investigation",
            screening_loss_ratio=0.58,
            screening_novelty=0.61,
            screening_passed=True,
            investigation_loss_ratio=0.58,
            investigation_robustness=1.0,
            investigation_passed=False,
        )
        nb.flush_writes()

        nb._sync_fingerprint_leaderboard(rid)

        row = nb.get_leaderboard_entry(rid)
        assert row is not None
        assert row["tier"] == "investigation"
        assert int(row["investigation_passed"] or 0) == 0
        nb.close()

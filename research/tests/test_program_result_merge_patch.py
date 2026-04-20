from __future__ import annotations

import json
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from research.scientist.notebook import LabNotebook
from research.scientist.runner._helpers_benchmark import promote_validation_candidate
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
            stage1_passed=True,
            loss_ratio=0.41,
            final_loss=1.8,
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
            model_source="graph_synthesis",
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            loss_ratio=0.42,
            novelty_score=0.66,
            novelty_confidence=0.71,
            fp_jacobian_spectral_norm=1.3,
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
        nb.conn.execute(
            """
            UPDATE program_results
            SET trust_label = ?, comparability_label = ?, data_provenance_json = ?
            WHERE result_id = ?
            """,
            (
                "candidate_grade",
                "candidate_comparable",
                json.dumps(
                    {"eligible_for_promotion": True, "provenance_complete": True}
                ),
                rid,
            ),
        )
        nb.conn.commit()

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

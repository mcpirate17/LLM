import sqlite3

from research.scientist.notebook import LabNotebook
from research.tools.backpopulate_screening_metrics import _target_post_fields
from research.tools.backfill_binding import (
    _query_candidates as query_binding_candidates,
)
from research.tools.backfill_templates import (
    _NON_ROUTING_TEMPLATES,
    _phase_settings,
    _scaffold_guided_priors,
    _start_backfill_experiment_with_retry,
    get_template_backfill_policy,
    plan_batch_size,
    resolve_weight_mode,
    should_stop_backfill_attempt,
)
from research.tools.run_s1_backpopulate import target_missing_clause


def test_phase_settings_differentiate_isolation_and_stack():
    isolation = _phase_settings("isolation")
    stack = _phase_settings("stack")
    assert isolation["composition_depth"] == 1
    assert stack["composition_depth"] == 2
    assert stack["n_layers"] > isolation["n_layers"]
    assert stack["stage1_steps"] > isolation["stage1_steps"]


def test_scaffold_guided_priors_build_weights_from_scaffold_stats(tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    nb = LabNotebook(db_path)
    try:
        nb.save_scaffold_profile_run(run_id="run1", config={"stage1_steps": 8})
        for idx in range(5):
            nb.save_scaffold_profile_result(
                run_id="run1",
                family="gpt2_attn",
                case_name=f"gpt2_attn:linear_attention:{idx}",
                status="ok",
                metrics={
                    "sandbox_passed": True,
                    "passed": True,
                    "loss_ratio": 0.4,
                    "validation_loss_ratio": 0.38,
                    "throughput_tok_s": 3500.0,
                },
                graph_json='{"metadata":{"scaffold_family":"gpt2_attn"}}',
                graph_fingerprint=f"fp{idx}",
                op_a="linear_attention",
            )
        op_weights, category_weights = _scaffold_guided_priors(str(db_path))
        assert op_weights["linear_attention"] > 1.0
        assert "mixing" in category_weights or "sequence" in category_weights
    finally:
        nb.close()


def test_non_routing_template_whitelist_covers_attention_ablations():
    assert "attn_softmax_normalized_matmul" in _NON_ROUTING_TEMPLATES
    assert "attn_softmax_normalized_matmul_v2" in _NON_ROUTING_TEMPLATES
    assert "attn_softmax_normalized_matmul_compact_ffn" in _NON_ROUTING_TEMPLATES
    assert "attn_softmax_normalized_matmul_fixed_tail_norm" in _NON_ROUTING_TEMPLATES
    assert "attn_linear_no_matmul_ffn" in _NON_ROUTING_TEMPLATES
    assert "attn_linear_no_matmul_ffn_v2" in _NON_ROUTING_TEMPLATES
    assert "attn_linear_no_matmul_ffn_dense_tail" in _NON_ROUTING_TEMPLATES
    assert "attn_linear_no_matmul_ffn_direct_recovery" in _NON_ROUTING_TEMPLATES
    assert "attn_softmax_router_sidecar" not in _NON_ROUTING_TEMPLATES


def test_template_backfill_policy_defaults_and_freeze_state():
    assert (
        get_template_backfill_policy("intelligent_multilane_router").mode == "coverage"
    )
    assert get_template_backfill_policy("recursive_depth_router").mode == "harvest"
    assert (
        get_template_backfill_policy("multiscale_rich_lane_router").mode == "coverage"
    )
    assert get_template_backfill_policy("unknown_template_name").mode == "coverage"


def test_plan_batch_size_applies_policy_caps_and_freeze():
    assert (
        plan_batch_size(
            template_name="intelligent_multilane_router",
            requested_batch_size=50,
            metric_deficit=74,
            s1_deficit=0,
        )
        == 24
    )
    assert (
        plan_batch_size(
            template_name="recursive_depth_router",
            requested_batch_size=5,
            metric_deficit=2,
            s1_deficit=0,
        )
        == 12
    )
    assert (
        plan_batch_size(
            template_name="multiscale_rich_lane_router",
            requested_batch_size=10,
            metric_deficit=100,
            s1_deficit=0,
        )
        == 12
    )


def test_resolve_weight_mode_uses_policy_preferences_only_in_auto():
    assert (
        resolve_weight_mode("recursive_depth_router", "random", "auto")
        == "scaffold_guided"
    )
    assert (
        resolve_weight_mode("attn_routing_block", "scaffold_guided", "auto")
        == "uniform"
    )
    assert resolve_weight_mode("recursive_depth_router", "random", "off") == "random"
    assert (
        resolve_weight_mode("intelligent_multilane_router", "uniform", "auto")
        == "uniform"
    )


def test_should_stop_backfill_attempt_detects_structural_and_zero_s1_regressions():
    stop, reason = should_stop_backfill_attempt(
        template_name="intelligent_multilane_router",
        batch_summary={
            "rows": 10,
            "s0": 6,
            "s1": 0,
            "error_counts": {"causality_violation": 5},
        },
    )
    assert stop is True
    assert reason == "structural_stop:causality_violation"

    stop, reason = should_stop_backfill_attempt(
        template_name="recursive_depth_router",
        batch_summary={
            "rows": 12,
            "s0": 10,
            "s1": 0,
            "error_counts": {"insufficient_learning": 12},
        },
    )
    assert stop is True
    assert reason == "zero_s1_conversion"


def test_start_backfill_experiment_with_retry_recovers_from_db_lock_once():
    class FakeNotebook:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class FakeRunner:
        def __init__(self):
            self.calls = 0

        def _make_notebook(self):
            return FakeNotebook()

        def _populate_refuted_cache(self, _nb):
            return None

        def _start_preregistered_experiment(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise sqlite3.OperationalError("database is locked")
            return "exp_ok"

    runner = FakeRunner()
    exp_id, nb = _start_backfill_experiment_with_retry(
        runner,
        experiment_type="backfill",
        config={"n_programs": 1},
        hypothesis="test",
        hypothesis_metadata={"source": "test"},
        created_by="pytest",
        max_attempts=2,
    )
    assert exp_id == "exp_ok"
    assert runner.calls == 2
    assert nb.closed is False


def test_binding_backfill_queries_missing_binding_not_missing_induction(tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    nb = LabNotebook(db_path)
    try:
        nb.conn.execute(
            """
            INSERT INTO program_results(
                result_id, timestamp, graph_json, graph_fingerprint, induction_auc, binding_auc, stage1_passed
            ) VALUES (?, datetime('now'), ?, ?, ?, ?, ?)
            """,
            ("r_missing_bind", "{}", "fp_missing_bind", 0.123, None, 1),
        )
        nb.conn.execute(
            """
            INSERT INTO program_results(
                result_id, timestamp, graph_json, graph_fingerprint, induction_auc, binding_auc, stage1_passed
            ) VALUES (?, datetime('now'), ?, ?, ?, ?, ?)
            """,
            ("r_has_bind", "{}", "fp_has_bind", None, 0.456, 1),
        )
        nb.conn.execute(
            """
            INSERT INTO program_results(
                result_id, timestamp, graph_json, graph_fingerprint, induction_auc, binding_auc, stage1_passed
            ) VALUES (?, datetime('now'), ?, ?, ?, ?, ?)
            """,
            ("r_no_s1", "{}", "fp_no_s1", 0.789, None, 0),
        )
        nb.conn.execute(
            """
            INSERT INTO leaderboard(
                entry_id, result_id, timestamp, tier, composite_score, is_reference, model_source
            ) VALUES (?, ?, datetime('now'), ?, ?, ?, ?)
            """,
            ("e_missing_bind", "r_missing_bind", "screening", 10.0, 0, "unit"),
        )
        nb.conn.execute(
            """
            INSERT INTO leaderboard(
                entry_id, result_id, timestamp, tier, composite_score, is_reference, model_source
            ) VALUES (?, ?, datetime('now'), ?, ?, ?, ?)
            """,
            ("e_has_bind", "r_has_bind", "screening", 9.0, 0, "unit"),
        )
        nb.conn.execute(
            """
            INSERT INTO leaderboard(
                entry_id, result_id, timestamp, tier, composite_score, is_reference, model_source
            ) VALUES (?, ?, datetime('now'), ?, ?, ?, ?)
            """,
            ("e_no_s1", "r_no_s1", "screening", 8.0, 0, "unit"),
        )
        nb.conn.commit()

        rows, _ = query_binding_candidates(
            nb, ["screening"], top=10, force=False, metrics=("binding",)
        )
        result_ids = {row["result_id"] for row in rows}
        assert "r_missing_bind" in result_ids
        assert "r_has_bind" not in result_ids
        assert "r_no_s1" not in result_ids
    finally:
        nb.close()


def test_post_train_target_binding_is_binding_only():
    assert _target_post_fields("binding") == ("binding_auc",)
    assert _target_post_fields("induction") == ("induction_auc",)
    assert _target_post_fields("hellaswag") == ("hellaswag_acc",)


def test_s1_backpopulate_binding_clause_only_checks_binding_auc():
    assert target_missing_clause("binding") == "pr.binding_auc IS NULL"
    assert target_missing_clause("induction") == "pr.induction_auc IS NULL"

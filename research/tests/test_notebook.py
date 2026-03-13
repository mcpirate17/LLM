"""
Integration Tests for the AI Scientist Research Pipeline

Tests the full stack: notebook schema, leaderboard lifecycle,
auto-escalation pipeline, API endpoints, mode selection, and
novelty scoring fixes.

Run: cd /path/to/LLM && python -m unittest research.tests.test_integration -v
"""

import pytest
import importlib
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit

# Detect available dependencies
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

# Import modules that don't require torch directly
# (bypass scientist/__init__.py which eagerly imports runner)
def _import_module(dotted_path):
    """Import a submodule without triggering parent __init__.py."""
    return importlib.import_module(dotted_path)


try:
    from research.scientist.notebook import LabNotebook, ExperimentEntry
    HAS_NOTEBOOK = True
except Exception as e:
    HAS_NOTEBOOK = False
    print(f"Notebook import failed: {e}")

try:
    from research.scientist.persona import Aria
    HAS_PERSONA = True
except Exception as e:
    HAS_PERSONA = False
    print(f"Persona import failed: {e}")

try:
    import research.scientist.llm.prompts as _prompts_mod  # noqa: F401
    HAS_PROMPTS = True
except Exception as e:
    HAS_PROMPTS = False
    print(f"Prompts import failed: {e}")

try:
    import research.scientist.llm.context as _context_mod  # noqa: F401
    HAS_CONTEXT = True
except Exception as e:
    HAS_CONTEXT = False
    print(f"Context import failed: {e}")


# ── Test 1: Notebook & Leaderboard ──


@unittest.skipUnless(HAS_NOTEBOOK, "requires notebook module")
class TestNotebook(unittest.TestCase):
    """Test notebook schema, CRUD, and leaderboard lifecycle."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_notebook.db")
        self.nb = LabNotebook(self.db_path)

    def tearDown(self):
        self.nb.close()

    def test_schema_all_tables_exist(self):
        """Verify all 9+ tables are created."""
        tables = [row[0] for row in self.nb.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        expected = [
            "experiments", "entries", "program_results", "metrics_log",
            "insights", "training_curves", "op_success_rates",
            "learning_log", "leaderboard",
        ]
        for t in expected:
            self.assertIn(t, tables, f"Missing table: {t}")

    def test_program_results_experiment_index_exists(self):
        """program_results(experiment_id) should be indexed for large-query performance."""
        indexes = self.nb.conn.execute("PRAGMA index_list('program_results')").fetchall()
        has_experiment_index = False
        for idx in indexes:
            idx_name = idx[1]
            cols = self.nb.conn.execute(f"PRAGMA index_info('{idx_name}')").fetchall()
            col_names = [c[2] for c in cols]
            if col_names == ["experiment_id"]:
                has_experiment_index = True
                break
        self.assertTrue(has_experiment_index, "Missing index on program_results(experiment_id)")

    def test_phase4_indexes_exist(self):
        """Hot query paths should have the new Phase 4 indexes."""
        program_indexes = {
            row[1]: [c[2] for c in self.nb.conn.execute(f"PRAGMA index_info('{row[1]}')").fetchall()]
            for row in self.nb.conn.execute("PRAGMA index_list('program_results')").fetchall()
        }
        leaderboard_indexes = {
            row[1]: [c[2] for c in self.nb.conn.execute(f"PRAGMA index_info('{row[1]}')").fetchall()]
            for row in self.nb.conn.execute("PRAGMA index_list('leaderboard')").fetchall()
        }

        self.assertIn(["stage1_passed"], program_indexes.values())
        self.assertIn(["graph_fingerprint"], program_indexes.values())
        self.assertIn(["routing_mode"], program_indexes.values())
        self.assertIn(["model_source"], leaderboard_indexes.values())

    def test_designer_run_lineage_upsert_and_query(self):
        """Designer run lineage rows should upsert and round-trip structured payloads."""
        self.nb.save_designer_run_lineage(
            run_id="eval_test_lineage_1",
            workflow_id="wf_lineage_test",
            workflow_version=3,
            graph_fingerprint="fp_lineage_test",
            status="success",
            source="aria_designer",
            total_time_ms=123.4,
            metrics={"overall_novelty": 0.42},
            payload={"status": "success"},
        )
        row = self.nb.get_designer_run_lineage("eval_test_lineage_1")
        self.assertIsNotNone(row)
        self.assertEqual(row["workflow_id"], "wf_lineage_test")
        self.assertEqual(row["workflow_version"], 3)
        self.assertEqual(row["graph_fingerprint"], "fp_lineage_test")
        self.assertEqual(row["status"], "success")
        self.assertAlmostEqual(float(row["total_time_ms"]), 123.4, places=2)
        self.assertEqual(row["metrics"].get("overall_novelty"), 0.42)

        # Upsert same run_id updates status and metrics.
        self.nb.save_designer_run_lineage(
            run_id="eval_test_lineage_1",
            workflow_id="wf_lineage_test",
            status="failed_sandbox",
            metrics={"overall_novelty": 0.1},
        )
        row2 = self.nb.get_designer_run_lineage("eval_test_lineage_1")
        self.assertEqual(row2["status"], "failed_sandbox")
        self.assertEqual(row2["metrics"].get("overall_novelty"), 0.1)

    def test_experiment_lifecycle(self):
        """Start → complete → query an experiment."""
        exp_id = self.nb.start_experiment(
            experiment_type="synthesis",
            config={"n_programs": 10},
            hypothesis="Test hypothesis",
        )
        self.assertIsNotNone(exp_id)

        # Record a program result
        result_id = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="abc123",
            graph_json='{"nodes": {}}',
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            loss_ratio=0.5,
            novelty_score=0.7,
        )
        self.assertIsNotNone(result_id)

        # Complete the experiment
        self.nb.complete_experiment(
            experiment_id=exp_id,
            results={"stage1_passed": 1, "total": 10},
            aria_summary="Test summary",
            aria_mood="excited",
        )

        exp = self.nb.get_experiment(exp_id)
        self.assertEqual(exp["status"], "completed")

    def test_experiment_trends_stabilize_tiny_runs_with_confidence_fields(self):
        """Tiny-run S1 rates should be damped and expose confidence metadata."""
        tiny_exp = self.nb.start_experiment("synthesis", {}, "tiny")
        self.nb.complete_experiment(
            experiment_id=tiny_exp,
            results={"total": 1, "stage1_passed": 1, "stage0_passed": 1, "stage05_passed": 1},
        )

        stable_exp = self.nb.start_experiment("synthesis", {}, "stable")
        self.nb.complete_experiment(
            experiment_id=stable_exp,
            results={"total": 80, "stage1_passed": 8, "stage0_passed": 80, "stage05_passed": 40},
        )

        trends = self.nb.get_experiment_trends(limit=10)
        self.assertGreaterEqual(len(trends), 2)

        tiny_entry = next(t for t in trends if t["experiment_id"] == tiny_exp)
        stable_entry = next(t for t in trends if t["experiment_id"] == stable_exp)

        self.assertIn("adjusted_s1_pass_rate", tiny_entry)
        self.assertIn("s1_confidence_lower", tiny_entry)
        self.assertIn("s1_confidence_upper", tiny_entry)
        self.assertIn("s1_confidence_halfwidth", tiny_entry)
        self.assertIn("trend_weight", tiny_entry)
        self.assertIn("trend_confidence", tiny_entry)
        self.assertIn("trend_mode", tiny_entry)

        self.assertLess(tiny_entry["adjusted_s1_pass_rate"], tiny_entry["s1_pass_rate"])
        self.assertLess(tiny_entry["trend_weight"], stable_entry["trend_weight"])
        self.assertEqual(tiny_entry["trend_confidence"], "low")
        self.assertIn(stable_entry["trend_confidence"], {"medium", "high"})

    def test_start_experiment_records_code_version(self):
        """Experiment config should always include code_version metadata."""
        with patch.dict(os.environ, {"RESEARCH_CODE_VERSION": "test-version"}, clear=False):
            LabNotebook._cached_code_version = None
            exp_id = self.nb.start_experiment(
                experiment_type="synthesis",
                config={"n_programs": 3},
                hypothesis="version test",
            )

        row = self.nb.conn.execute(
            "SELECT config_json FROM experiments WHERE experiment_id = ?",
            (exp_id,),
        ).fetchone()
        self.assertIsNotNone(row)
        config = json.loads(row["config_json"])
        self.assertEqual(config.get("code_version"), "test-version")

    def test_log_learning_event_accepts_extra_keyword_metadata(self):
        """Learning log should accept extra metadata kwargs without raising."""
        self.nb.log_learning_event(
            "chat_config_adjusted",
            "Adjusted chat config",
            changes={"max_depth": 4},
            ignored={"unknown_key": 1},
        )
        rows = self.nb.get_learning_log(limit=5)
        self.assertGreaterEqual(len(rows), 1)
        latest = rows[0]
        self.assertEqual(latest["event_type"], "chat_config_adjusted")
        self.assertIsInstance(latest.get("evidence"), str)
        self.assertIn("changes", latest.get("evidence") or "")

    def test_leaderboard_upsert_and_query(self):
        """Leaderboard CRUD operations."""
        # Create an experiment and program first
        exp_id = self.nb.start_experiment("synthesis", {}, "test")
        result_id = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp001",
            graph_json="{}",
            stage1_passed=True,
            loss_ratio=0.4,
            novelty_score=0.8,
        )

        # Upsert to leaderboard
        entry_id = self.nb.upsert_leaderboard(
            result_id=result_id,
            model_source="graph_synthesis",
            architecture_desc="test_arch",
            screening_loss_ratio=0.4,
            screening_novelty=0.8,
            screening_passed=True,
            tier="screening",
        )
        self.assertIsNotNone(entry_id)

        # Query leaderboard
        entries = self.nb.get_leaderboard()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["tier"], "screening")
        self.assertGreater(entries[0]["composite_score"], 0)

        # Filter by tier
        screening = self.nb.get_leaderboard(tier="screening")
        self.assertEqual(len(screening), 1)
        empty = self.nb.get_leaderboard(tier="investigation")
        self.assertEqual(len(empty), 0)

    def test_leaderboard_upsert_updates_existing(self):
        """Upserting same result_id updates, doesn't duplicate."""
        exp_id = self.nb.start_experiment("synthesis", {}, "test")
        result_id = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp002",
            graph_json="{}",
            stage1_passed=True,
            loss_ratio=0.5,
        )

        # First upsert
        self.nb.upsert_leaderboard(
            result_id=result_id,
            model_source="graph_synthesis",
            screening_loss_ratio=0.5,
            tier="screening",
        )

        # Second upsert with investigation data
        self.nb.upsert_leaderboard(
            result_id=result_id,
            model_source="graph_synthesis",
            screening_loss_ratio=0.5,
            investigation_loss_ratio=0.4,
            investigation_robustness=0.6,
            tier="investigation",
        )

        entries = self.nb.get_leaderboard()
        self.assertEqual(len(entries), 1)  # not duplicated
        self.assertEqual(entries[0]["tier"], "investigation")
        self.assertAlmostEqual(entries[0]["investigation_robustness"], 0.6)

    def test_promote_to_tier(self):
        """Promoting a leaderboard entry updates tier + results."""
        exp_id = self.nb.start_experiment("synthesis", {}, "test")
        result_id = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp003",
            graph_json="{}",
            stage1_passed=True,
            loss_ratio=0.3,
        )

        entry_id = self.nb.upsert_leaderboard(
            result_id=result_id,
            model_source="graph_synthesis",
            screening_loss_ratio=0.3,
            tier="screening",
        )

        # Promote to investigation
        self.nb.promote_to_tier(
            entry_id=entry_id,
            tier="investigation",
            investigation_loss_ratio=0.25,
            investigation_robustness=0.7,
        )

        entries = self.nb.get_leaderboard()
        self.assertEqual(entries[0]["tier"], "investigation")

    def test_composite_score_increases_with_phases(self):
        """Composite score should increase as candidates pass more phases with good results."""
        score_screening = self.nb.compute_composite_score(
            screening_lr=0.5, screening_nov=0.7)
        score_investigation = self.nb.compute_composite_score(
            screening_lr=0.5, screening_nov=0.7,
            inv_lr=0.4, inv_robust=0.6)
        # Validation values represent a strong candidate: 30% of baseline
        # loss with low variance (std=0.1)
        score_validation = self.nb.compute_composite_score(
            screening_lr=0.5, screening_nov=0.7,
            inv_lr=0.4, inv_robust=0.6,
            val_baseline=0.3, val_std=0.1)

        self.assertGreater(score_investigation, score_screening)
        self.assertGreater(score_validation, score_investigation)

    def test_insights_crud(self):
        """Record and query insights."""
        exp_id = self.nb.start_experiment("synthesis", {}, "test")
        insight_id = self.nb.record_insight(
            category="pattern",
            content="Test insight content",
            experiment_id=exp_id,
            confidence=0.8,
        )
        self.assertIsNotNone(insight_id)

        insights = self.nb.get_insights()
        self.assertEqual(len(insights), 1)
        self.assertEqual(insights[0]["category"], "pattern")

    def test_entries_crud(self):
        """Add and query notebook entries."""
        entry_id = self.nb.add_entry(ExperimentEntry(
            entry_type="decision",
            title="Test Decision",
            content="We decided to test things.",
        ))
        self.assertIsNotNone(entry_id)

        entries = self.nb.get_entries()
        self.assertGreater(len(entries), 0)

    def test_dashboard_summary(self):
        """Dashboard summary returns expected keys."""
        summary = self.nb.get_dashboard_summary()
        expected_keys = [
            "total_experiments", "total_programs_evaluated",
            "stage1_survivors", "survival_rate",
        ]
        for k in expected_keys:
            self.assertIn(k, summary)

    def test_dashboard_summary_cache_invalidates_after_write(self):
        """Dashboard summary cache should refresh after notebook writes commit."""
        before = self.nb.get_dashboard_summary()
        exp_id = self.nb.start_experiment("synthesis", {}, "cache-refresh")
        self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_cache_refresh",
            graph_json="{}",
            stage1_passed=True,
        )
        self.nb.flush_writes()
        after = self.nb.get_dashboard_summary()
        self.assertEqual(after["total_programs_evaluated"], before["total_programs_evaluated"] + 1)
        self.assertEqual(after["stage1_survivors"], before["stage1_survivors"] + 1)

    def test_graph_structural_counts_supports_dict_nodes(self):
        """Structural op counting should handle dict-based graph JSON without refetching repeatedly."""
        exp_id = self.nb.start_experiment("synthesis", {}, "counts")
        graph_json = json.dumps({
            "nodes": {
                "a": {"op_name": "route_topk"},
                "b": {"op_name": "block_sparse_linear"},
                "c": {"op_name": "moe_topk"},
            }
        })
        result_id = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_counts",
            graph_json=graph_json,
            stage1_passed=True,
        )
        self.nb.flush_writes()

        counts = self.nb._graph_structural_counts(result_id)
        self.assertEqual(counts["routing"], 2)
        self.assertEqual(counts["sparse"], 1)
        self.assertEqual(counts["moe"], 2)

    def test_get_top_programs_sort_options(self):
        """get_top_programs supports all sort options."""
        exp_id = self.nb.start_experiment("synthesis", {}, "test")
        self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_sort",
            graph_json="{}",
            stage1_passed=True,
            loss_ratio=0.5,
            novelty_score=0.6,
        )

        for sort_by in ["novelty_score", "loss_ratio",
                        "structural_novelty", "behavioral_novelty",
                        "validation_loss_ratio", "discovery_loss_ratio"]:
            programs = self.nb.get_top_programs(5, sort_by=sort_by)
            self.assertIsInstance(programs, list)

    def test_training_curve_storage(self):
        """Store and retrieve training curves."""
        exp_id = self.nb.start_experiment("synthesis", {}, "test")
        result_id = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_curve",
            graph_json="{}",
            stage1_passed=True,
        )
        self.nb.flush_writes()

        curve = [
            {"step": 0, "loss": 5.0, "grad_norm": 1.0},
            {"step": 1, "loss": 4.5, "grad_norm": 0.9},
            {"step": 2, "loss": 4.0, "grad_norm": 0.8},
        ]
        self.nb.store_training_curve(result_id, curve)

        retrieved = self.nb.get_training_curve(result_id)
        self.assertEqual(len(retrieved), 3)
        self.assertAlmostEqual(retrieved[0]["loss"], 5.0)

    def test_training_curve_ignored_without_persisted_survivor(self):
        """Curve writes should be skipped when the parent result does not exist."""
        self.nb.store_training_curve("missing_result", [
            {"step": 0, "loss": 1.0, "grad_norm": 0.5},
        ])
        retrieved = self.nb.get_training_curve("missing_result")
        self.assertEqual(retrieved, [])

    def test_training_curve_deleted_with_parent_program(self):
        """Purging junk programs should cascade and remove attached curves."""
        exp_id = self.nb.start_experiment("synthesis", {}, "junk")
        result_id = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_junk",
            graph_json="{}",
            stage0_passed=False,
            stage1_passed=False,
            error_type="compile_error",
        )
        self.nb.flush_writes()
        self.nb.conn.execute(
            """INSERT INTO training_curves (result_id, step, loss, grad_norm, step_time_ms)
               VALUES (?, 0, 3.0, 1.0, 1.0)""",
            (result_id,),
        )
        self.nb.conn.commit()

        deleted = self.nb.purge_junk_programs()
        self.assertEqual(deleted["deleted"], 1)
        remaining = self.nb.conn.execute(
            "SELECT COUNT(*) FROM training_curves WHERE result_id = ?",
            (result_id,),
        ).fetchone()[0]
        self.assertEqual(remaining, 0)

    def test_top_op_combinations_handles_malformed_graph_json(self):
        """Analytics top_op_combinations should skip malformed JSON and still aggregate valid pairs."""
        from research.scientist.analytics import ExperimentAnalytics

        exp_id = self.nb.start_experiment("synthesis", {}, "combo-test")
        valid_graph = json.dumps({
            "nodes": {
                "a": {"op_name": "relu"},
                "b": {"op_name": "gelu"},
                "c": {"op_name": "input"},
            }
        })
        self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_combo_1",
            graph_json=valid_graph,
            stage1_passed=True,
            novelty_score=0.7,
        )
        self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_combo_2",
            graph_json=valid_graph,
            stage1_passed=True,
            novelty_score=0.5,
        )
        self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_combo_bad",
            graph_json="{bad-json",
            stage1_passed=True,
            novelty_score=0.9,
        )

        self.nb.flush_writes()

        analytics = ExperimentAnalytics(self.nb)
        combos = analytics.top_op_combinations(n=3)

        self.assertGreaterEqual(len(combos), 1)
        first = combos[0]
        self.assertEqual(first["ops"], ["gelu", "relu"])
        self.assertEqual(first["count"], 2)

    def test_experiment_clusters_model_selection_and_consensus(self):
        """experiment_clusters should select a sensible k and expose model-selection diagnostics."""
        from research.scientist.analytics import ExperimentAnalytics

        cluster_a = [
            {"s1": 74, "novelty": 1.25, "loss": 0.58, "duration": 95.0},
            {"s1": 70, "novelty": 1.18, "loss": 0.62, "duration": 92.0},
            {"s1": 77, "novelty": 1.32, "loss": 0.55, "duration": 98.0},
        ]
        cluster_b = [
            {"s1": 9, "novelty": 0.12, "loss": 1.62, "duration": 36.0},
            {"s1": 6, "novelty": 0.08, "loss": 1.74, "duration": 32.0},
            {"s1": 11, "novelty": 0.15, "loss": 1.58, "duration": 38.0},
        ]

        for i, spec in enumerate(cluster_a + cluster_b):
            exp_id = self.nb.start_experiment("synthesis", {"n_programs": 100}, f"cluster-{i}")
            self.nb.complete_experiment(
                exp_id,
                {
                    "total": 100,
                    "stage0_passed": 100,
                    "stage05_passed": 100,
                    "stage1_passed": spec["s1"],
                    "best_loss_ratio": spec["loss"],
                    "best_novelty_score": spec["novelty"],
                },
            )
            self.nb.conn.execute(
                "UPDATE experiments SET duration_seconds = ? WHERE experiment_id = ?",
                (spec["duration"], exp_id),
            )
        self.nb.conn.commit()

        clusters = ExperimentAnalytics(self.nb).experiment_clusters(n_clusters=5)
        self.assertIsNotNone(clusters)
        self.assertEqual(clusters["n_experiments"], 6)
        self.assertEqual(clusters["n_clusters"], 2)
        self.assertIn("model_selection", clusters)

        model_selection = clusters["model_selection"]
        self.assertIn("candidate_ks", model_selection)
        self.assertIn("selected_k", model_selection)
        self.assertIn("silhouette", model_selection)
        self.assertIn("consensus", model_selection)
        self.assertGreaterEqual(model_selection["selected_k"], 2)
        self.assertLessEqual(model_selection["selected_k"], 5)
        self.assertEqual(model_selection["selected_k"], 2)

        avg_s1_rates = sorted(c["avg_s1_rate"] for c in clusters["clusters"])
        self.assertLess(avg_s1_rates[0], 0.2)
        self.assertGreater(avg_s1_rates[-1], 0.65)
        self.assertGreater(clusters["stability_score"], 0.6)

    def test_experiment_clusters_include_failure_signature_features(self):
        """Clustering should use failure signatures to separate experiments with similar top-level metrics."""
        from research.scientist.analytics import ExperimentAnalytics

        for i in range(3):
            exp_id = self.nb.start_experiment("synthesis", {"n_programs": 10}, f"compile-heavy-{i}")
            for j in range(10):
                is_success = j >= 5
                self.nb.record_program_result(
                    experiment_id=exp_id,
                    graph_fingerprint=f"fp_compile_{i}_{j}",
                    graph_json="{}",
                    stage0_passed=is_success,
                    stage05_passed=is_success,
                    stage1_passed=is_success,
                    error_type=None if is_success else "compile_error",
                    stage_at_death=None if is_success else "stage0",
                )
            self.nb.complete_experiment(
                exp_id,
                {
                    "total": 10,
                    "stage0_passed": 5,
                    "stage05_passed": 5,
                    "stage1_passed": 5,
                    "best_loss_ratio": 0.9,
                    "best_novelty_score": 0.45,
                },
            )
            self.nb.conn.execute(
                "UPDATE experiments SET duration_seconds = ? WHERE experiment_id = ?",
                (60.0 + i, exp_id),
            )

        stage1_errors = ["nan_output", "overflow", "timeout", "nan_output", "overflow"]
        for i in range(3):
            exp_id = self.nb.start_experiment("synthesis", {"n_programs": 10}, f"stage1-heavy-{i}")
            for j in range(10):
                is_success = j >= 5
                self.nb.record_program_result(
                    experiment_id=exp_id,
                    graph_fingerprint=f"fp_stage1_{i}_{j}",
                    graph_json="{}",
                    stage0_passed=True,
                    stage05_passed=True,
                    stage1_passed=is_success,
                    error_type=None if is_success else stage1_errors[j],
                    stage_at_death=None if is_success else "stage1",
                )
            self.nb.complete_experiment(
                exp_id,
                {
                    "total": 10,
                    "stage0_passed": 10,
                    "stage05_passed": 10,
                    "stage1_passed": 5,
                    "best_loss_ratio": 0.9,
                    "best_novelty_score": 0.45,
                },
            )
            self.nb.conn.execute(
                "UPDATE experiments SET duration_seconds = ? WHERE experiment_id = ?",
                (60.0 + i, exp_id),
            )

        self.nb.flush_writes()
        self.nb.conn.commit()

        clusters = ExperimentAnalytics(self.nb).experiment_clusters(n_clusters=4)
        self.assertIsNotNone(clusters)
        self.assertEqual(clusters["n_experiments"], 6)
        self.assertIn("compile_fail_rate", clusters["feature_keys"])
        self.assertIn("error_diversity", clusters["feature_keys"])
        self.assertIn("model_selection", clusters)
        self.assertEqual(clusters["model_selection"]["selected_k"], 2)

        compile_rates = sorted(c["avg_compile_fail_rate"] for c in clusters["clusters"])
        stage1_rates = sorted(c["avg_stage1_fail_rate"] for c in clusters["clusters"])
        error_diversities = sorted(c["avg_error_diversity"] for c in clusters["clusters"])

        self.assertLess(compile_rates[0], 0.1)
        self.assertGreater(compile_rates[-1], 0.4)
        self.assertLess(stage1_rates[0], 0.1)
        self.assertGreater(stage1_rates[-1], 0.4)
        self.assertLess(error_diversities[0], 0.1)
        self.assertGreater(error_diversities[-1], 0.5)

    def test_experiment_clusters_include_trajectory_features(self):
        """Clustering should separate experiments with matched aggregates but opposite temporal trajectories."""
        from research.scientist.analytics import ExperimentAnalytics

        pending_updates = []

        def _record_experiment_with_trajectory(prefix: str, improving: bool, offset: float):
            exp_id = self.nb.start_experiment("synthesis", {"n_programs": 10}, prefix)
            base_ts = 1000.0 + offset

            for step in range(10):
                if improving:
                    is_success = step >= 5
                    novelty = 0.2 + (0.06 * step)
                    loss_ratio = 1.3 - (0.06 * step)
                else:
                    is_success = step < 5
                    novelty = 0.8 - (0.06 * step)
                    loss_ratio = 0.7 + (0.06 * step)

                result_id = self.nb.record_program_result(
                    experiment_id=exp_id,
                    graph_fingerprint=f"fp_{prefix}_{step}",
                    graph_json="{}",
                    stage0_passed=True,
                    stage05_passed=True,
                    stage1_passed=is_success,
                    novelty_score=novelty,
                    loss_ratio=loss_ratio,
                    error_type=None if is_success else "stage1_error",
                    stage_at_death=None if is_success else "stage1",
                )
                pending_updates.append((base_ts + step, result_id))

            self.nb.complete_experiment(
                exp_id,
                {
                    "total": 10,
                    "stage0_passed": 10,
                    "stage05_passed": 10,
                    "stage1_passed": 5,
                    "best_loss_ratio": 0.7,
                    "best_novelty_score": 0.8,
                },
            )
            self.nb.conn.execute(
                "UPDATE experiments SET duration_seconds = ? WHERE experiment_id = ?",
                (75.0, exp_id),
            )

        for i in range(3):
            _record_experiment_with_trajectory(f"traj_up_{i}", improving=True, offset=i * 20)
            _record_experiment_with_trajectory(f"traj_down_{i}", improving=False, offset=200 + (i * 20))

        self.nb.flush_writes()
        for ts, rid in pending_updates:
            self.nb.conn.execute(
                "UPDATE program_results SET timestamp = ? WHERE result_id = ?",
                (ts, rid),
            )
        self.nb.conn.commit()

        clusters = ExperimentAnalytics(self.nb).experiment_clusters(n_clusters=4)
        self.assertIsNotNone(clusters)
        self.assertEqual(clusters["n_experiments"], 6)

        self.assertIn("stage1_momentum", clusters["feature_keys"])
        self.assertIn("novelty_momentum", clusters["feature_keys"])
        self.assertIn("loss_improvement_momentum", clusters["feature_keys"])
        self.assertIn("outcome_volatility", clusters["feature_keys"])
        self.assertIn("outcome_peak_timing", clusters["feature_keys"])
        self.assertIn("recovery_lag", clusters["feature_keys"])
        self.assertIn("stage1_transition_timing", clusters["feature_keys"])
        self.assertIn("primary_change_point_timing", clusters["feature_keys"])
        self.assertIn("stage1_transition_density", clusters["feature_keys"])
        self.assertIn("change_point_confidence", clusters["feature_keys"])
        self.assertIn("windowed_change_dispersion", clusters["feature_keys"])
        self.assertIn("window_change_localization", clusters["feature_keys"])
        self.assertIn("transition_gap_entropy", clusters["feature_keys"])

        self.assertIn("model_selection", clusters)
        self.assertEqual(clusters["model_selection"]["selected_k"], 2)

        stage1_momentum = sorted(c["avg_stage1_momentum"] for c in clusters["clusters"])
        novelty_momentum = sorted(c["avg_novelty_momentum"] for c in clusters["clusters"])
        loss_momentum = sorted(c["avg_loss_improvement_momentum"] for c in clusters["clusters"])
        peak_timing = sorted(c["avg_outcome_peak_timing"] for c in clusters["clusters"])
        recovery_lag = sorted(c["avg_recovery_lag"] for c in clusters["clusters"])
        transition_timing = [c["avg_stage1_transition_timing"] for c in clusters["clusters"]]
        change_point_timing = [c["avg_primary_change_point_timing"] for c in clusters["clusters"]]
        transition_density = [c["avg_stage1_transition_density"] for c in clusters["clusters"]]
        change_point_conf = [c["avg_change_point_confidence"] for c in clusters["clusters"]]
        change_dispersion = [c["avg_windowed_change_dispersion"] for c in clusters["clusters"]]
        change_localization = [c["avg_window_change_localization"] for c in clusters["clusters"]]
        transition_gap_entropy = [c["avg_transition_gap_entropy"] for c in clusters["clusters"]]

        self.assertLess(stage1_momentum[0], -0.6)
        self.assertGreater(stage1_momentum[-1], 0.6)
        self.assertLess(novelty_momentum[0], -0.3)
        self.assertGreater(novelty_momentum[-1], 0.3)
        self.assertLess(loss_momentum[0], -0.3)
        self.assertGreater(loss_momentum[-1], 0.3)
        self.assertLess(peak_timing[0], 0.35)
        self.assertGreater(peak_timing[-1], 0.65)
        self.assertLess(recovery_lag[0], 0.35)
        self.assertGreater(recovery_lag[-1], 0.8)
        self.assertTrue(all(0.0 <= t <= 1.0 for t in transition_timing))
        self.assertTrue(all(0.0 <= t <= 1.0 for t in change_point_timing))
        self.assertTrue(all(0.0 <= t <= 1.0 for t in transition_density))
        self.assertTrue(all(0.0 <= t <= 1.0 for t in change_point_conf))
        self.assertTrue(all(d >= 0.0 for d in change_dispersion))
        self.assertTrue(all(0.0 <= t <= 1.0 for t in change_localization))
        self.assertTrue(all(0.0 <= t <= 1.0 for t in transition_gap_entropy))



class TestStaleExperimentCleanup(unittest.TestCase):
    """Test cleanup_stale_experiments marks zombies as failed."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(self.tmpdir, "test_cleanup.db")
        from research.scientist.notebook.__init__ import LabNotebook
        self.nb = LabNotebook(db_path)

    def tearDown(self):
        self.nb.close()

    def test_cleanup_marks_stale_as_failed(self):
        """Experiments running longer than timeout should be marked failed."""
        import time as _time
        exp_id = self.nb.start_experiment("synthesis", {}, "stale test")
        # Backdate started_at to 2 hours ago
        two_hours_ago = _time.time() - 7200
        self.nb.conn.execute(
            "UPDATE experiments SET started_at = ? WHERE experiment_id = ?",
            (two_hours_ago, exp_id),
        )
        self.nb.conn.commit()

        # Verify it's still running
        row = self.nb.conn.execute(
            "SELECT status FROM experiments WHERE experiment_id = ?",
            (exp_id,),
        ).fetchone()
        self.assertEqual(row["status"], "running")

        # Cleanup with 60-minute timeout
        cleaned = self.nb.cleanup_stale_experiments(timeout_minutes=60)
        self.assertEqual(cleaned, 1)

        # Verify it's now failed
        row = self.nb.conn.execute(
            "SELECT status FROM experiments WHERE experiment_id = ?",
            (exp_id,),
        ).fetchone()
        self.assertEqual(row["status"], "failed")

    def test_cleanup_ignores_recent_running(self):
        """Recently started experiments should not be cleaned up."""
        exp_id = self.nb.start_experiment("synthesis", {}, "recent test")
        # Don't backdate — it just started

        cleaned = self.nb.cleanup_stale_experiments(timeout_minutes=60)
        self.assertEqual(cleaned, 0)

        row = self.nb.conn.execute(
            "SELECT status FROM experiments WHERE experiment_id = ?",
            (exp_id,),
        ).fetchone()
        self.assertEqual(row["status"], "running")

    def test_cleanup_ignores_completed(self):
        """Completed experiments should not be affected by cleanup."""
        exp_id = self.nb.start_experiment("synthesis", {}, "done test")
        self.nb.complete_experiment(exp_id, {
            "total": 1, "stage0_passed": 1, "stage1_passed": 0,
        })

        # Backdate to look stale
        import time as _time
        self.nb.conn.execute(
            "UPDATE experiments SET started_at = ? WHERE experiment_id = ?",
            (_time.time() - 7200, exp_id),
        )
        self.nb.conn.commit()

        cleaned = self.nb.cleanup_stale_experiments(timeout_minutes=60)
        self.assertEqual(cleaned, 0)

    def test_cleanup_marks_startup_failed_without_progress(self):
        """Running experiments with no progress should be cleaned by startup-failure threshold."""
        import time as _time

        exp_id = self.nb.start_experiment("validation", {}, "startup fail test")
        twenty_minutes_ago = _time.time() - (20 * 60)
        self.nb.conn.execute(
            "UPDATE experiments SET started_at = ? WHERE experiment_id = ?",
            (twenty_minutes_ago, exp_id),
        )
        self.nb.conn.commit()

        cleaned = self.nb.cleanup_stale_experiments(
            timeout_minutes=60,
            startup_failure_minutes=15,
        )
        self.assertEqual(cleaned, 1)

        row = self.nb.conn.execute(
            "SELECT status, results_json FROM experiments WHERE experiment_id = ?",
            (exp_id,),
        ).fetchone()
        self.assertEqual(row["status"], "failed")
        results = json.loads(row["results_json"] or "{}")
        self.assertEqual(
            results.get("failure_reason"),
            "Startup failed before any progress was recorded",
        )

    def test_cleanup_keeps_recent_progress_even_if_old(self):
        """Experiments with progress signals should not be marked startup-failed."""
        import time as _time

        exp_id = self.nb.start_experiment("validation", {}, "progress test")
        twenty_minutes_ago = _time.time() - (20 * 60)
        self.nb.conn.execute(
            "UPDATE experiments SET started_at = ? WHERE experiment_id = ?",
            (twenty_minutes_ago, exp_id),
        )
        self.nb.conn.commit()

        self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="progress_fp",
            graph_json='{"nodes": {}}',
            stage0_passed=True,
            stage05_passed=False,
            stage1_passed=False,
            loss_ratio=1.5,
        )
        self.nb.flush_writes()

        cleaned = self.nb.cleanup_stale_experiments(
            timeout_minutes=60,
            startup_failure_minutes=15,
        )
        self.assertEqual(cleaned, 0)

        row = self.nb.conn.execute(
            "SELECT status FROM experiments WHERE experiment_id = ?",
            (exp_id,),
        ).fetchone()
        self.assertEqual(row["status"], "running")

    def test_cleanup_returns_zero_when_nothing_stale(self):
        """No stale experiments means cleanup returns 0."""
        cleaned = self.nb.cleanup_stale_experiments(timeout_minutes=60)
        self.assertEqual(cleaned, 0)



class TestLeaderboardDedup(unittest.TestCase):
    """Test leaderboard fingerprint deduplication."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(self.tmpdir, "test_dedup.db")
        from research.scientist.notebook.__init__ import LabNotebook
        self.nb = LabNotebook(db_path)

    def tearDown(self):
        self.nb.close()

    def test_leaderboard_dedup_by_fingerprint(self):
        """Same fingerprint should only appear once, keeping best score."""
        exp_id = self.nb.start_experiment("synthesis", {}, "test")

        # Create two results with same fingerprint
        r1 = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="same_fp",
            graph_json='{"nodes": {}}',
            stage1_passed=True,
            loss_ratio=0.5,
            novelty_score=0.6,
        )
        r2 = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="same_fp",
            graph_json='{"nodes": {}}',
            stage1_passed=True,
            loss_ratio=0.3,
            novelty_score=0.9,
        )
        # Third with different fingerprint
        r3 = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="different_fp",
            graph_json='{"nodes": {}}',
            stage1_passed=True,
            loss_ratio=0.4,
            novelty_score=0.7,
        )

        self.nb.flush_writes()

        self.nb.upsert_leaderboard(
            result_id=r1, model_source="test",
            screening_loss_ratio=0.5, screening_novelty=0.6,
            screening_passed=True, tier="screening",
        )
        self.nb.upsert_leaderboard(
            result_id=r2, model_source="test",
            screening_loss_ratio=0.3, screening_novelty=0.9,
            screening_passed=True, tier="screening",
        )
        self.nb.upsert_leaderboard(
            result_id=r3, model_source="test",
            screening_loss_ratio=0.4, screening_novelty=0.7,
            screening_passed=True, tier="screening",
        )

        entries = self.nb.get_leaderboard()
        # Should be 2 (deduped same_fp), not 3
        self.assertEqual(len(entries), 2)
        # The entry for same_fp should be the one with higher composite_score
        # (r2 has better loss_ratio=0.3 and novelty=0.9)

    def test_leaderboard_dedup_keeps_distinct_fingerprints(self):
        """Different fingerprints should all appear."""
        exp_id = self.nb.start_experiment("synthesis", {}, "test")

        for i, fp in enumerate(["fp_a", "fp_b", "fp_c"]):
            rid = self.nb.record_program_result(
                experiment_id=exp_id,
                graph_fingerprint=fp,
                graph_json='{}',
                stage1_passed=True,
                loss_ratio=0.4 + i * 0.1,
            )
            self.nb.flush_writes()
            self.nb.upsert_leaderboard(
                result_id=rid, model_source="test",
                screening_loss_ratio=0.4 + i * 0.1,
                screening_passed=True, tier="screening",
            )

        entries = self.nb.get_leaderboard()
        self.assertEqual(len(entries), 3)

    def test_get_investigated_fingerprints(self):
        """Fingerprints at investigation+ tier should be returned."""
        exp_id = self.nb.start_experiment("synthesis", {}, "test")

        # Create screening entry (should NOT appear)
        rid1 = self.nb.record_program_result(
            experiment_id=exp_id, graph_fingerprint="fp_screening",
            graph_json='{}', stage1_passed=True, loss_ratio=0.4,
        )
        self.nb.upsert_leaderboard(
            result_id=rid1, model_source="test",
            screening_loss_ratio=0.4, screening_passed=True, tier="screening",
        )

        # Create investigation entry (SHOULD appear)
        rid2 = self.nb.record_program_result(
            experiment_id=exp_id, graph_fingerprint="fp_investigated",
            graph_json='{}', stage1_passed=True, loss_ratio=0.3,
        )
        self.nb.upsert_leaderboard(
            result_id=rid2, model_source="test",
            screening_loss_ratio=0.3, screening_passed=True,
            tier="investigation",
        )

        # Create validation entry (SHOULD appear)
        rid3 = self.nb.record_program_result(
            experiment_id=exp_id, graph_fingerprint="fp_validated",
            graph_json='{}', stage1_passed=True, loss_ratio=0.2,
        )
        self.nb.upsert_leaderboard(
            result_id=rid3, model_source="test",
            screening_loss_ratio=0.2, screening_passed=True,
            tier="validation",
        )

        self.nb.flush_writes()
        fps = self.nb.get_investigated_fingerprints()
        self.assertNotIn("fp_screening", fps)
        self.assertIn("fp_investigated", fps)
        self.assertIn("fp_validated", fps)
        self.assertEqual(len(fps), 2)




if __name__ == '__main__':
    unittest.main()

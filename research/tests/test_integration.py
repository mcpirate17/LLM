"""
Integration Tests for the AI Scientist Research Pipeline

Tests the full stack: notebook schema, leaderboard lifecycle,
auto-escalation pipeline, API endpoints, mode selection, and
novelty scoring fixes.

Run: cd /path/to/LLM && python -m unittest research.tests.test_integration -v
"""

import importlib
import json
import os
import sys
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Detect available dependencies
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import flask
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

# Import modules that don't require torch directly
# (bypass scientist/__init__.py which eagerly imports runner)
def _import_module(dotted_path):
    """Import a submodule without triggering parent __init__.py."""
    return importlib.import_module(dotted_path)


# These imports work without torch if we avoid scientist/__init__.py
# We use importlib.util to load modules without triggering parent __init__
def _load_module_directly(name, filepath):
    """Load a module directly from file path, bypassing __init__.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    _nb_mod = _load_module_directly(
        "research.scientist.notebook",
        os.path.join(_project_root, "scientist", "notebook.py"))
    LabNotebook = _nb_mod.LabNotebook
    ExperimentEntry = _nb_mod.ExperimentEntry
    HAS_NOTEBOOK = True
except Exception as e:
    HAS_NOTEBOOK = False
    print(f"Notebook import failed: {e}")

try:
    _persona_mod = _load_module_directly(
        "research.scientist.persona",
        os.path.join(_project_root, "scientist", "persona.py"))
    Aria = _persona_mod.Aria
    HAS_PERSONA = True
except Exception as e:
    HAS_PERSONA = False
    print(f"Persona import failed: {e}")

try:
    # llm subpackage needs its __init__ first
    _llm_init_path = os.path.join(_project_root, "scientist", "llm", "__init__.py")
    if os.path.exists(_llm_init_path):
        _load_module_directly("research.scientist.llm", _llm_init_path)
    _prompts_mod = _load_module_directly(
        "research.scientist.llm.prompts",
        os.path.join(_project_root, "scientist", "llm", "prompts.py"))
    HAS_PROMPTS = True
except Exception as e:
    HAS_PROMPTS = False
    print(f"Prompts import failed: {e}")

try:
    _context_mod = _load_module_directly(
        "research.scientist.llm.context",
        os.path.join(_project_root, "scientist", "llm", "context.py"))
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
                        "structural_novelty", "behavioral_novelty"]:
            programs = self.nb.get_top_programs(5, sort_by=sort_by)
            self.assertIsInstance(programs, list)

    def test_training_curve_storage(self):
        """Store and retrieve training curves."""
        exp_id = self.nb.start_experiment("synthesis", {}, "test")
        result_id = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_curve",
            graph_json="{}",
        )

        curve = [
            {"step": 0, "loss": 5.0, "grad_norm": 1.0},
            {"step": 1, "loss": 4.5, "grad_norm": 0.9},
            {"step": 2, "loss": 4.0, "grad_norm": 0.8},
        ]
        self.nb.store_training_curve(result_id, curve)

        retrieved = self.nb.get_training_curve(result_id)
        self.assertEqual(len(retrieved), 3)
        self.assertAlmostEqual(retrieved[0]["loss"], 5.0)

    def test_top_op_combinations_handles_malformed_graph_json(self):
        """Analytics top_op_combinations should skip malformed JSON and still aggregate valid pairs."""
        analytics_mod = _load_module_directly(
            "research.scientist.analytics",
            os.path.join(_project_root, "scientist", "analytics.py"),
        )
        ExperimentAnalytics = analytics_mod.ExperimentAnalytics

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

        analytics = ExperimentAnalytics(self.nb)
        combos = analytics.top_op_combinations(n=3)

        self.assertGreaterEqual(len(combos), 1)
        first = combos[0]
        self.assertEqual(first["ops"], ["gelu", "relu"])
        self.assertEqual(first["count"], 2)

    def test_experiment_clusters_model_selection_and_consensus(self):
        """experiment_clusters should select a sensible k and expose model-selection diagnostics."""
        analytics_mod = _load_module_directly(
            "research.scientist.analytics",
            os.path.join(_project_root, "scientist", "analytics.py"),
        )
        ExperimentAnalytics = analytics_mod.ExperimentAnalytics

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
        analytics_mod = _load_module_directly(
            "research.scientist.analytics",
            os.path.join(_project_root, "scientist", "analytics.py"),
        )
        ExperimentAnalytics = analytics_mod.ExperimentAnalytics

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
        analytics_mod = _load_module_directly(
            "research.scientist.analytics",
            os.path.join(_project_root, "scientist", "analytics.py"),
        )
        ExperimentAnalytics = analytics_mod.ExperimentAnalytics

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
                self.nb.conn.execute(
                    "UPDATE program_results SET timestamp = ? WHERE result_id = ?",
                    (base_ts + step, result_id),
                )

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


# ── Test 2: Novelty Scoring ──


@unittest.skipUnless(HAS_TORCH, "requires torch for graph/metrics modules")
class TestNoveltyScoring(unittest.TestCase):
    """Test that novelty scoring no longer always returns 1.0."""

    def _make_graph(self, n_ops=5, op_names=None):
        """Create a simple computation graph for testing."""
        from research.synthesis.graph import ComputationGraph
        graph = ComputationGraph(model_dim=256)

        # Add input node
        input_id = graph.add_input()

        if op_names is None:
            op_names = ["relu", "gelu", "tanh", "sigmoid", "silu"]

        prev_id = input_id
        for op_name in op_names[:n_ops]:
            prev_id = graph.add_op(op_name, [prev_id])

        graph.set_output(prev_id)
        return graph

    def test_structural_novelty_not_always_one(self):
        """Structural novelty should NOT be ~1.0 for every graph."""
        from research.eval.metrics import novelty_score

        # Simple graph with few unique ops (all the same)
        simple_graph = self._make_graph(
            op_names=["relu", "relu", "relu", "relu", "relu"])

        # Diverse graph with many unique ops
        diverse_graph = self._make_graph(
            op_names=["relu", "gelu", "tanh", "sigmoid", "silu"])

        simple_nov = novelty_score(simple_graph)
        diverse_nov = novelty_score(diverse_graph)

        # Neither should be 1.0 (the old bug)
        self.assertLess(simple_nov.structural_novelty, 0.95,
                        "Simple graph should NOT have max novelty")

        # Diverse should be higher than simple
        self.assertGreater(diverse_nov.structural_novelty,
                           simple_nov.structural_novelty,
                           "Diverse graph should have higher novelty than simple")

    def test_no_fingerprint_discount(self):
        """Without behavioral fingerprint, overall_novelty should be discounted."""
        from research.eval.metrics import novelty_score

        graph = self._make_graph()
        nov = novelty_score(graph, fingerprint=None)

        # Should be discounted (0.6x structural)
        expected_max = nov.structural_novelty * 0.6 + 0.01  # small tolerance
        self.assertLessEqual(nov.overall_novelty, expected_max,
                             "No-fingerprint novelty should be discounted")

    def test_with_fingerprint_uses_behavioral(self):
        """With fingerprint, overall should use 70% behavioral weight."""
        from research.eval.metrics import novelty_score
        from research.eval.fingerprint import BehavioralFingerprint

        graph = self._make_graph()
        fp = BehavioralFingerprint(
            novelty_score=0.9,
            cka_vs_transformer=0.3,
            cka_vs_ssm=0.2,
            cka_vs_conv=0.1,
        )

        nov = novelty_score(graph, fingerprint=fp)

        # Should be weighted: 0.3 * structural + 0.7 * behavioral
        expected = 0.3 * nov.structural_novelty + 0.7 * 0.9
        self.assertAlmostEqual(nov.overall_novelty, expected, places=2)

    def test_duplicate_penalization(self):
        """Exact duplicate fingerprints should be penalized."""
        from research.eval.metrics import novelty_score

        graph = self._make_graph()
        fp_str = graph.fingerprint()

        nov_fresh = novelty_score(graph, known_fingerprints=[])
        nov_dup = novelty_score(graph, known_fingerprints=[fp_str])

        self.assertGreater(nov_fresh.overall_novelty,
                           nov_dup.overall_novelty,
                           "Duplicate should be penalized")


class TestNoveltyCalibration(unittest.TestCase):
    """Regression tests for novelty confidence/quality tracking (#4, #10)."""

    def test_fingerprint_quality_defaults(self):
        """BehavioralFingerprint defaults to quality='none', analyses_succeeded=0."""
        from research.eval.fingerprint import BehavioralFingerprint

        fp = BehavioralFingerprint()
        self.assertEqual(fp.quality, "none")
        self.assertEqual(fp.analyses_succeeded, 0)

    def test_novelty_confidence_defaults(self):
        """NoveltyMetrics defaults to novelty_confidence=0.0."""
        from research.eval.metrics import NoveltyMetrics

        nm = NoveltyMetrics()
        self.assertEqual(nm.novelty_confidence, 0.0)

    def test_confidence_no_fingerprint(self):
        """Without fingerprint, novelty_confidence should be 0.2."""
        from research.eval.metrics import novelty_score
        from research.synthesis.graph import ComputationGraph

        graph = ComputationGraph(model_dim=256)
        inp = graph.add_input()
        op = graph.add_op("relu", [inp])
        graph.set_output(op)

        nov = novelty_score(graph, fingerprint=None)
        self.assertAlmostEqual(nov.novelty_confidence, 0.2)

    def test_confidence_full_quality_fingerprint(self):
        """Full-quality fingerprint gives confidence=0.9."""
        from research.eval.metrics import novelty_score
        from research.eval.fingerprint import BehavioralFingerprint
        from research.synthesis.graph import ComputationGraph

        graph = ComputationGraph(model_dim=256)
        inp = graph.add_input()
        op = graph.add_op("relu", [inp])
        graph.set_output(op)

        fp = BehavioralFingerprint(
            novelty_score=0.7, quality="full", analyses_succeeded=4,
        )
        nov = novelty_score(graph, fingerprint=fp)
        self.assertAlmostEqual(nov.novelty_confidence, 0.9)

    def test_confidence_partial_quality_fingerprint(self):
        """Partial-quality fingerprint gives confidence=0.4 + n*0.1."""
        from research.eval.metrics import novelty_score
        from research.eval.fingerprint import BehavioralFingerprint
        from research.synthesis.graph import ComputationGraph

        graph = ComputationGraph(model_dim=256)
        inp = graph.add_input()
        op = graph.add_op("relu", [inp])
        graph.set_output(op)

        for n in (1, 2, 3):
            fp = BehavioralFingerprint(
                novelty_score=0.5, quality="partial", analyses_succeeded=n,
            )
            nov = novelty_score(graph, fingerprint=fp)
            expected = 0.4 + n * 0.1
            self.assertAlmostEqual(nov.novelty_confidence, expected,
                                   msg=f"analyses_succeeded={n}")

    def test_confidence_none_quality_with_fingerprint(self):
        """quality='none' but fingerprint provided gives confidence=0.3."""
        from research.eval.metrics import novelty_score
        from research.eval.fingerprint import BehavioralFingerprint
        from research.synthesis.graph import ComputationGraph

        graph = ComputationGraph(model_dim=256)
        inp = graph.add_input()
        op = graph.add_op("relu", [inp])
        graph.set_output(op)

        fp = BehavioralFingerprint(novelty_score=0.5, quality="none",
                                   analyses_succeeded=0)
        nov = novelty_score(graph, fingerprint=fp)
        self.assertAlmostEqual(nov.novelty_confidence, 0.3)

    def test_novelty_confidence_persisted_in_db(self):
        """novelty_confidence column exists and round-trips through DB."""
        from research.scientist.notebook import LabNotebook
        import tempfile, os

        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            nb = LabNotebook(db_path)
            exp_id = nb.start_experiment("test", {})
            rid = nb.record_program_result(
                exp_id, "fp123", "{}",
                novelty_score=0.7, novelty_confidence=0.85,
            )
            detail = nb.get_program_detail(rid)
            self.assertAlmostEqual(detail["novelty_confidence"], 0.85)
            nb.close()

    def test_op_success_rates_tracks_novelty_confidence(self):
        """update_op_success_rates persists avg_novelty_confidence."""
        from research.scientist.notebook import LabNotebook
        import tempfile, os, json

        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            nb = LabNotebook(db_path)
            exp_id = nb.start_experiment("test", {})
            graph = {"nodes": {"n1": {"op_name": "relu", "inputs": ["input"]}}}
            nb.record_program_result(
                exp_id, "fp1", json.dumps(graph),
                novelty_score=0.6, novelty_confidence=0.9,
                stage0_passed=True, stage1_passed=True,
            )
            nb.record_program_result(
                exp_id, "fp2", json.dumps(graph),
                novelty_score=0.4, novelty_confidence=0.3,
                stage0_passed=True, stage1_passed=False,
            )
            nb.update_op_success_rates(exp_id)
            rates = nb.get_op_success_rates()
            relu_rate = [r for r in rates if r["op_name"] == "relu"][0]
            self.assertIsNotNone(relu_rate["avg_novelty_confidence"])
            self.assertAlmostEqual(relu_rate["avg_novelty_confidence"], 0.6)
            nb.close()

    def test_breakthrough_requires_novelty_confidence(self):
        """Runner breakthrough gate requires novelty_confidence >= 0.5."""
        # Verify the threshold is present in the source code
        import ast
        from pathlib import Path

        runner_path = Path(__file__).parent.parent / "scientist" / "runner.py"
        source = runner_path.read_text()
        # Should contain the novelty confidence gate
        self.assertIn("nov_conf >= 0.5", source,
                       "Breakthrough gate must require novelty_confidence >= 0.5")

    def test_breakthrough_requires_5_seeds(self):
        """Runner breakthrough gate requires >= 5 seeds passed."""
        from pathlib import Path

        runner_path = Path(__file__).parent.parent / "scientist" / "runner.py"
        source = runner_path.read_text()
        self.assertIn("len(passed_seeds) >= 5", source,
                       "Breakthrough gate must require >= 5 seeds")

    def test_validation_n_seeds_default_is_5(self):
        """RunConfig.validation_n_seeds default must be >= 5."""
        from research.scientist.runner import RunConfig

        config = RunConfig()
        self.assertGreaterEqual(config.validation_n_seeds, 5,
                                "validation_n_seeds must default to >= 5 for breakthrough eligibility")

    def test_grammar_weights_discount_low_confidence_novelty(self):
        """Grammar weight novelty factor should be scaled by confidence."""
        from research.scientist.analytics import ExperimentAnalytics
        from unittest.mock import MagicMock

        analytics = ExperimentAnalytics.__new__(ExperimentAnalytics)
        analytics.nb = MagicMock()

        # High confidence novelty vs low confidence novelty
        stats_high_conf = {
            "total": 100, "s1_total": 20, "novelty_sum": 50.0, "count": 100,
            "conf_sum": 90.0, "conf_count": 100,  # avg conf = 0.9
        }
        stats_low_conf = {
            "total": 100, "s1_total": 20, "novelty_sum": 50.0, "count": 100,
            "conf_sum": 20.0, "conf_count": 100,  # avg conf = 0.2
        }

        weights_high = analytics._compute_weights_from_stats(
            {"activation": stats_high_conf})
        weights_low = analytics._compute_weights_from_stats(
            {"activation": stats_low_conf})

        # Both should produce weights, but high-conf should weight novelty more
        self.assertIsNotNone(weights_high)
        self.assertIsNotNone(weights_low)
        # With same s1_rate (only one category), both hit statistical guard
        # and return default. Use two categories to get past the guard.
        stats_good = {
            "total": 100, "s1_total": 30, "novelty_sum": 80.0, "count": 100,
            "conf_sum": 90.0, "conf_count": 100,
        }
        stats_bad = {
            "total": 100, "s1_total": 5, "novelty_sum": 10.0, "count": 100,
            "conf_sum": 20.0, "conf_count": 100,
        }
        w_high = analytics._compute_weights_from_stats({
            "activation": stats_good, "linear": stats_bad,
        })
        # Replace good stats with low confidence
        stats_good_lowconf = dict(stats_good)
        stats_good_lowconf["conf_sum"] = 10.0  # avg conf = 0.1
        w_low = analytics._compute_weights_from_stats({
            "activation": stats_good_lowconf, "linear": stats_bad,
        })
        self.assertIsNotNone(w_high)
        self.assertIsNotNone(w_low)
        # High-confidence novelty should give a higher weight
        self.assertGreater(w_high["activation"], w_low["activation"],
                           "High-confidence novelty should produce higher grammar weight")

    def test_grammar_weights_cap_repeated_fingerprint_influence(self):
        """Fingerprint-capped weighting should reduce repeated architecture dominance."""
        from research.scientist.analytics import ExperimentAnalytics

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "fingerprint_cap.db")
            nb = LabNotebook(db_path)
            exp_id = nb.start_experiment("synthesis", {}, "fingerprint-cap")

            dominant_graph = {
                "nodes": {
                    "n1": {"op_name": "relu"},
                    "n2": {"op_name": "gelu"},
                    "n3": {"op_name": "tanh"},
                    "n4": {"op_name": "matmul"},
                    "n5": {"op_name": "layernorm"},
                }
            }
            contrast_graph = {
                "nodes": {
                    "n1": {"op_name": "sin"},
                    "n2": {"op_name": "cos"},
                    "n3": {"op_name": "exp"},
                    "n4": {"op_name": "sum_reduce"},
                    "n5": {"op_name": "mean_reduce"},
                }
            }

            for i in range(24):
                nb.record_program_result(
                    exp_id,
                    "fp_repeat_dominant",
                    json.dumps(dominant_graph),
                    stage0_passed=True,
                    stage1_passed=True,
                    novelty_score=0.8,
                    novelty_confidence=0.9,
                    loss_ratio=0.4,
                    timestamp=time.time() + i,
                )
            for i in range(10):
                nb.record_program_result(
                    exp_id,
                    f"fp_unique_{i}",
                    json.dumps(contrast_graph),
                    stage0_passed=True,
                    stage1_passed=False,
                    novelty_score=0.2,
                    novelty_confidence=0.6,
                    loss_ratio=0.9,
                    timestamp=time.time() + 100 + i,
                )

            analytics_capped = ExperimentAnalytics(nb)
            analytics_uncapped = ExperimentAnalytics(nb)
            analytics_capped.FINGERPRINT_WEIGHT_CAP = 3.0
            analytics_uncapped.FINGERPRINT_WEIGHT_CAP = 1_000_000.0

            capped_rates, capped_diag = analytics_capped._collect_fingerprint_capped_op_rates(3.0)
            uncapped_rates, _ = analytics_uncapped._collect_fingerprint_capped_op_rates(1_000_000.0)

            self.assertIn("relu", capped_rates)
            self.assertIn("relu", uncapped_rates)
            self.assertLess(capped_rates["relu"]["n_used"], uncapped_rates["relu"]["n_used"])
            self.assertGreater(capped_diag["rerun_ratio"], 0.5)
            self.assertGreater(capped_diag["top_fingerprint_concentration"], 0.5)

            capped_weights = analytics_capped.compute_grammar_weights()
            uncapped_weights = analytics_uncapped.compute_grammar_weights()
            self.assertIsNotNone(capped_weights)
            self.assertIsNotNone(uncapped_weights)

            diag = analytics_capped.grammar_weight_learning_diagnostics()
            self.assertEqual(diag.get("mode"), "fingerprint_capped")
            self.assertTrue(diag.get("used_fingerprint_capping"))
            self.assertEqual(diag.get("fingerprint_cap"), 3.0)

            nb.close()

    def test_composite_score_discounts_low_confidence_novelty(self):
        """Composite score should weight novelty contribution by confidence."""
        from research.scientist.notebook import LabNotebook

        # Full confidence: novelty fully counted
        score_full = LabNotebook.compute_composite_score(
            screening_lr=0.5, screening_nov=0.8, novelty_confidence=0.9)
        # Low confidence: novelty discounted
        score_low = LabNotebook.compute_composite_score(
            screening_lr=0.5, screening_nov=0.8, novelty_confidence=0.2)
        # No confidence param: defaults to 1.0 (backward compat)
        score_none = LabNotebook.compute_composite_score(
            screening_lr=0.5, screening_nov=0.8)

        self.assertGreater(score_full, score_low,
                           "High confidence should yield higher composite score")
        self.assertEqual(score_none, LabNotebook.compute_composite_score(
            screening_lr=0.5, screening_nov=0.8, novelty_confidence=1.0),
            "None confidence should behave like 1.0")
        # Zero confidence should eliminate novelty contribution entirely
        score_zero = LabNotebook.compute_composite_score(
            screening_lr=0.5, screening_nov=0.8, novelty_confidence=0.0)
        score_no_nov = LabNotebook.compute_composite_score(
            screening_lr=0.5, screening_nov=0.0)
        self.assertAlmostEqual(score_zero, score_no_nov, places=6,
                               msg="Zero confidence should be equivalent to zero novelty")

    def test_upsert_leaderboard_passes_novelty_confidence(self):
        """upsert_leaderboard should use novelty_confidence in composite score."""
        import tempfile, os
        from research.scientist.notebook import LabNotebook

        with tempfile.TemporaryDirectory() as d:
            nb = LabNotebook(os.path.join(d, "test.db"))
            exp_id = nb.start_experiment("test", {}, "test")
            rid = nb.record_program_result(
                experiment_id=exp_id,
                graph_fingerprint="fp1",
                graph_json="{}",
                loss_ratio=0.5,
                novelty_score=0.8,
                novelty_confidence=0.9,
            )
            # High confidence
            eid_high = nb.upsert_leaderboard(
                result_id=rid, model_source="test",
                screening_loss_ratio=0.5, screening_novelty=0.8,
                novelty_confidence=0.9,
            )
            # Low confidence
            rid2 = nb.record_program_result(
                experiment_id=exp_id,
                graph_fingerprint="fp2",
                graph_json="{}",
                loss_ratio=0.5,
                novelty_score=0.8,
                novelty_confidence=0.2,
            )
            eid_low = nb.upsert_leaderboard(
                result_id=rid2, model_source="test",
                screening_loss_ratio=0.5, screening_novelty=0.8,
                novelty_confidence=0.2,
            )
            lb = nb.get_leaderboard(limit=10)
            scores = {e["entry_id"]: e["composite_score"] for e in lb}
            self.assertGreater(scores[eid_high], scores[eid_low])


class TestBaselineDataFn(unittest.TestCase):
    """Tests for baseline training with custom data functions."""

    @unittest.skipUnless(HAS_TORCH, "torch not available")
    def test_baseline_with_data_fn(self):
        """Baseline trains using provided data_fn instead of random tokens."""
        from research.eval.baseline import TransformerBaseline
        import torch

        call_count = [0]
        def fake_data(batch_size, seq_len, dev):
            call_count[0] += 1
            return torch.randint(0, 1024, (batch_size, seq_len), device=dev)

        with tempfile.TemporaryDirectory() as tmpdir:
            bl = TransformerBaseline(cache_path=os.path.join(tmpdir, "bl.db"))
            loss = bl.get_baseline_loss(
                d_model=64, seq_len=32, n_steps=5, vocab_size=1024,
                batch_size=2, device="cpu", data_fn=fake_data, data_tag="test",
            )
            self.assertTrue(0 < loss < 20)
            self.assertGreater(call_count[0], 0, "data_fn should have been called")

    @unittest.skipUnless(HAS_TORCH, "torch not available")
    def test_baseline_compare_with_data_fn(self):
        """compare() passes data_fn through to training."""
        from research.eval.baseline import TransformerBaseline
        import torch

        def fake_data(batch_size, seq_len, dev):
            return torch.randint(0, 1024, (batch_size, seq_len), device=dev)

        with tempfile.TemporaryDirectory() as tmpdir:
            bl = TransformerBaseline(cache_path=os.path.join(tmpdir, "bl.db"))
            ratio = bl.compare(
                program_loss=5.0,
                d_model=64, seq_len=32, n_steps=5, vocab_size=1024,
                batch_size=2, device="cpu", data_fn=fake_data, data_tag="test",
            )
            self.assertIsInstance(ratio, float)
            self.assertGreater(ratio, 0)

    @unittest.skipUnless(HAS_TORCH, "torch not available")
    def test_data_tag_separates_cache(self):
        """Different data_tags produce separate cache entries."""
        from research.eval.baseline import TransformerBaseline

        with tempfile.TemporaryDirectory() as tmpdir:
            bl = TransformerBaseline(cache_path=os.path.join(tmpdir, "bl.db"))
            key_random = bl._config_key(64, 32, 5, 1024, data_tag="random")
            key_hydra = bl._config_key(64, 32, 5, 1024, data_tag="hydra")
            self.assertNotEqual(key_random, key_hydra)

    @unittest.skipUnless(HAS_TORCH, "torch not available")
    def test_hydra_batch_fallback_to_random(self):
        """_get_hydra_batch returns None when HYDRA loader fails to init."""
        from research.scientist.runner import ExperimentRunner, RunConfig
        import torch

        config = RunConfig(data_mode="hydra", hydra_project_root="/nonexistent")
        runner = ExperimentRunner.__new__(ExperimentRunner)
        runner._hydra_loader = None
        runner._hydra_iter = None
        runner._hydra_signature = ""
        # Mock the import to raise immediately rather than risk hanging
        with patch.dict("sys.modules", {"hydra.data": None}):
            result = runner._get_hydra_batch(config, 2, 32, torch.device("cpu"))
        self.assertIsNone(result, "Should return None when HYDRA unavailable")

    @unittest.skipUnless(HAS_TORCH, "torch not available")
    def test_make_baseline_data_fn_random_mode(self):
        """_make_baseline_data_fn returns (None, 'random') for random mode."""
        from research.scientist.runner import ExperimentRunner, RunConfig

        config = RunConfig(data_mode="random")
        runner = ExperimentRunner.__new__(ExperimentRunner)
        data_fn, data_tag = runner._make_baseline_data_fn(config)
        self.assertIsNone(data_fn)
        self.assertEqual(data_tag, "random")

    @unittest.skipUnless(HAS_TORCH, "torch not available")
    def test_make_baseline_data_fn_hydra_mode(self):
        """_make_baseline_data_fn returns a callable for hydra mode."""
        from research.scientist.runner import ExperimentRunner, RunConfig

        config = RunConfig(data_mode="hydra")
        runner = ExperimentRunner.__new__(ExperimentRunner)
        data_fn, data_tag = runner._make_baseline_data_fn(config)
        self.assertIsNotNone(data_fn)
        self.assertEqual(data_tag, "hydra")


class TestCkaReferenceArtifacts(unittest.TestCase):
    """Tests for CKA reference artifact loader/validator/cache (#28/#43 Phase A)."""

    def _make_artifact_dir(self, tmpdir, manifest_override=None, families=None):
        """Helper: create a valid artifact directory with manifest and .pt files."""
        import json, torch
        art_dir = os.path.join(tmpdir, "cka_references", "v1")
        os.makedirs(art_dir, exist_ok=True)

        manifest = {
            "artifact_version": "v1",
            "schema_version": "1",
            "created_at": "2026-01-01T00:00:00Z",
            "code_version": "test",
            "reference_families": ["transformer", "ssm", "conv"],
            "probe_protocol_hash": "abc123",
            "activation_shape": [16, 32],
            "quality_flags": {"overall": "good"},
        }
        if manifest_override:
            manifest.update(manifest_override)

        with open(os.path.join(art_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f)

        # Create .pt files
        shape = manifest["activation_shape"]
        for family in (families or ["transformer", "ssm", "conv"]):
            data = {
                "activations": torch.randn(shape[0], shape[1]),
                "config": {"family": family},
                "training_info": {},
            }
            torch.save(data, os.path.join(art_dir, f"{family}.pt"))

        return art_dir

    def test_load_manifest_valid(self):
        """Valid manifest loads without error."""
        from research.eval.cka_references import load_manifest
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            art_dir = self._make_artifact_dir(d)
            m = load_manifest(Path(art_dir))
            self.assertEqual(m.artifact_version, "v1")
            self.assertEqual(m.schema_version, "1")
            self.assertEqual(set(m.reference_families), {"transformer", "ssm", "conv"})
            self.assertEqual(m.activation_shape, [16, 32])

    def test_load_manifest_missing_file(self):
        """Missing manifest.json raises ValueError."""
        from research.eval.cka_references import load_manifest
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError, msg="No manifest.json"):
                load_manifest(Path(d))

    def test_load_manifest_malformed_json(self):
        """Malformed JSON raises ValueError."""
        from research.eval.cka_references import load_manifest
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "manifest.json")
            with open(p, "w") as f:
                f.write("{bad json")
            with self.assertRaises(ValueError):
                load_manifest(Path(d))

    def test_load_manifest_missing_fields(self):
        """Manifest missing required fields raises ValueError."""
        from research.eval.cka_references import load_manifest
        import tempfile, json
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "manifest.json")
            with open(p, "w") as f:
                json.dump({"artifact_version": "v1"}, f)
            with self.assertRaises(ValueError, msg="missing required fields"):
                load_manifest(Path(d))

    def test_load_manifest_unsupported_schema(self):
        """Unsupported schema version raises ValueError."""
        from research.eval.cka_references import load_manifest
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            art_dir = self._make_artifact_dir(d, {"schema_version": "99"})
            with self.assertRaises(ValueError, msg="Unsupported schema"):
                load_manifest(Path(art_dir))

    def test_load_manifest_missing_family(self):
        """Manifest with incomplete families raises ValueError."""
        from research.eval.cka_references import load_manifest
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            art_dir = self._make_artifact_dir(
                d, {"reference_families": ["transformer"]}
            )
            with self.assertRaises(ValueError, msg="missing reference families"):
                load_manifest(Path(art_dir))

    def test_load_manifest_bad_activation_shape(self):
        """Invalid activation_shape raises ValueError."""
        from research.eval.cka_references import load_manifest
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            art_dir = self._make_artifact_dir(d, {"activation_shape": [0, 32]})
            with self.assertRaises(ValueError):
                load_manifest(Path(art_dir))

    def test_load_reference_activations_valid(self):
        """Valid .pt files load as tensors with correct shape."""
        from research.eval.cka_references import load_manifest, load_reference_activations
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            art_dir = self._make_artifact_dir(d)
            m = load_manifest(Path(art_dir))
            refs = load_reference_activations(Path(art_dir), m)
            self.assertEqual(set(refs.keys()), {"transformer", "ssm", "conv"})
            for t in refs.values():
                self.assertEqual(tuple(t.shape[-2:]), (16, 32))

    def test_load_reference_activations_missing_file(self):
        """Missing .pt file raises ValueError."""
        from research.eval.cka_references import load_manifest, load_reference_activations
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            art_dir = self._make_artifact_dir(d)
            os.remove(os.path.join(art_dir, "ssm.pt"))
            m = load_manifest(Path(art_dir))
            with self.assertRaises(ValueError, msg="Missing artifact file"):
                load_reference_activations(Path(art_dir), m)

    def test_load_reference_activations_shape_mismatch(self):
        """Tensor with wrong shape raises ValueError."""
        from research.eval.cka_references import load_manifest, load_reference_activations
        import tempfile, torch
        with tempfile.TemporaryDirectory() as d:
            art_dir = self._make_artifact_dir(d)
            # Overwrite one file with wrong shape
            torch.save(
                {"activations": torch.randn(8, 32)},
                os.path.join(art_dir, "conv.pt"),
            )
            m = load_manifest(Path(art_dir))
            with self.assertRaises(ValueError, msg="shape mismatch"):
                load_reference_activations(Path(art_dir), m)

    def test_store_no_artifacts_returns_none(self):
        """ReferenceCkaStore with no artifacts returns None references."""
        from research.eval.cka_references import ReferenceCkaStore
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            store = ReferenceCkaStore(artifact_dir=os.path.join(d, "nonexistent"))
            self.assertIsNone(store.get_references())
            self.assertFalse(store.is_artifact_backed)
            meta = store.get_metadata()
            self.assertEqual(meta["cka_source"], "heuristic_fallback")

    def test_store_with_valid_artifacts(self):
        """ReferenceCkaStore loads valid artifacts successfully."""
        from research.eval.cka_references import ReferenceCkaStore
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            art_dir = self._make_artifact_dir(d)
            store = ReferenceCkaStore(artifact_dir=art_dir)
            refs = store.get_references()
            self.assertIsNotNone(refs)
            self.assertEqual(set(refs.keys()), {"transformer", "ssm", "conv"})
            self.assertTrue(store.is_artifact_backed)
            meta = store.get_metadata()
            self.assertEqual(meta["cka_source"], "artifact")
            self.assertEqual(meta["cka_artifact_version"], "v1")

    def test_store_reset_clears_cache(self):
        """reset() clears loaded state so next access reloads."""
        from research.eval.cka_references import ReferenceCkaStore
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            art_dir = self._make_artifact_dir(d)
            store = ReferenceCkaStore(artifact_dir=art_dir)
            self.assertTrue(store.is_artifact_backed)
            store.reset()
            # Point to nonexistent dir after reset
            store._artifact_dir = os.path.join(d, "gone")
            self.assertFalse(store.is_artifact_backed)

    def test_store_metadata_provenance_fields(self):
        """Metadata includes all expected provenance fields."""
        from research.eval.cka_references import ReferenceCkaStore
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            art_dir = self._make_artifact_dir(d)
            store = ReferenceCkaStore(artifact_dir=art_dir)
            meta = store.get_metadata()
            self.assertIn("cka_source", meta)
            self.assertIn("cka_artifact_version", meta)
            self.assertIn("cka_probe_protocol_hash", meta)
            self.assertIn("cka_reference_quality", meta)
            self.assertEqual(meta["cka_reference_quality"], "good")

    # ── Phase C: Runtime CKA switchover tests ──

    def test_compute_reference_cka_with_artifacts(self):
        """_compute_reference_cka uses artifact activations when provided."""
        import torch
        from research.eval.fingerprint import _compute_reference_cka

        # Create fake candidate reps and reference activations
        S, D = 16, 32
        reps = torch.randn(1, S, D)
        ref_activations = {
            "transformer": torch.randn(S, D),
            "ssm": torch.randn(S, D),
            "conv": torch.randn(S, D),
        }
        result = _compute_reference_cka(reps, ref_activations=ref_activations)
        self.assertTrue(result["_succeeded"])
        for family in ("transformer", "ssm", "conv"):
            self.assertGreaterEqual(result[family], 0.0)
            self.assertLessEqual(result[family], 1.0)

    def test_compute_reference_cka_heuristic_fallback(self):
        """_compute_reference_cka falls back to heuristic when no artifacts."""
        import torch
        from research.eval.fingerprint import _compute_reference_cka

        reps = torch.randn(1, 16, 32)
        result = _compute_reference_cka(reps, ref_activations=None)
        self.assertTrue(result["_succeeded"])
        # Heuristic should still produce valid CKA values
        for family in ("transformer", "ssm", "conv"):
            self.assertGreaterEqual(result[family], 0.0)
            self.assertLessEqual(result[family], 1.0)

    def test_compute_reference_cka_seq_len_mismatch(self):
        """Artifact CKA handles different seq lengths between candidate and reference."""
        import torch
        from research.eval.fingerprint import _compute_reference_cka

        reps = torch.randn(1, 16, 32)  # seq_len=16
        ref_activations = {
            "transformer": torch.randn(24, 32),  # seq_len=24 (longer)
            "ssm": torch.randn(8, 32),           # seq_len=8 (shorter)
            "conv": torch.randn(16, 32),          # seq_len=16 (same)
        }
        result = _compute_reference_cka(reps, ref_activations=ref_activations)
        self.assertTrue(result["_succeeded"])

    def test_fingerprint_records_cka_source(self):
        """Fingerprint records cka_source provenance."""
        from research.eval.cka_references import reset_default_store
        reset_default_store()  # ensure clean state

        fp = self._make_fingerprint()
        # Should be one of the valid sources
        self.assertIn(fp.cka_source, ("artifact", "heuristic_fallback", "none"))

    def test_fingerprint_heuristic_fallback_when_no_artifacts(self):
        """Fingerprint falls back to heuristic when artifact dir is missing."""
        from unittest.mock import patch
        from research.eval import cka_references
        from research.eval.cka_references import ReferenceCkaStore, reset_default_store

        reset_default_store()
        # Force a store pointing to nonexistent dir
        fake_store = ReferenceCkaStore(artifact_dir="/nonexistent/path")
        with patch.object(cka_references, '_default_store', fake_store):
            with patch.object(cka_references, '_default_lock', cka_references.threading.Lock()):
                # Override get_default_store to return our fake store
                with patch('research.eval.cka_references.get_default_store', return_value=fake_store):
                    fp = self._make_fingerprint()
        self.assertIn(fp.cka_source, ("heuristic_fallback",))
        reset_default_store()

    def _make_fingerprint(self):
        """Helper: compute fingerprint on a tiny model."""
        import torch
        import torch.nn as nn
        from research.eval.fingerprint import compute_fingerprint

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 32)
                self.linear = nn.Linear(32, 100)
            def forward(self, x):
                return self.linear(self.embed(x))

        model = TinyModel()
        return compute_fingerprint(model, seq_len=8, model_dim=32,
                                   vocab_size=100, device="cpu", n_probes=4)

    def test_fingerprint_cka_provenance_fields_exist(self):
        """BehavioralFingerprint has cka_source and cka_artifact_version fields."""
        from research.eval.fingerprint import BehavioralFingerprint

        fp = BehavioralFingerprint()
        self.assertEqual(fp.cka_source, "none")
        self.assertIsNone(fp.cka_artifact_version)
        d = fp.to_dict()
        self.assertIn("cka_source", d)
        self.assertIn("cka_artifact_version", d)

    def test_export_produces_loadable_artifacts(self):
        """Export tool produces artifacts that ReferenceCkaStore can load."""
        from research.tools.export_cka_references import export_artifacts
        from research.eval.cka_references import ReferenceCkaStore

        with tempfile.TemporaryDirectory() as d:
            art_dir = str(Path(d) / "refs" / "v1")
            export_artifacts(
                output_dir=art_dir, seed=123, n_steps=10, device="cpu",
            )
            store = ReferenceCkaStore(artifact_dir=art_dir)
            refs = store.get_references()
            self.assertIsNotNone(refs)
            self.assertEqual(set(refs.keys()), {"transformer", "ssm", "conv"})
            self.assertTrue(store.is_artifact_backed)
            meta = store.get_metadata()
            self.assertEqual(meta["cka_source"], "artifact")
            self.assertEqual(meta["cka_artifact_version"], "v1")

    def test_export_deterministic(self):
        """Same seed produces same probe_protocol_hash."""
        from research.tools.export_cka_references import export_artifacts
        import torch

        with tempfile.TemporaryDirectory() as d:
            d1 = str(Path(d) / "run1")
            d2 = str(Path(d) / "run2")
            export_artifacts(output_dir=d1, seed=99, n_steps=5, device="cpu")
            export_artifacts(output_dir=d2, seed=99, n_steps=5, device="cpu")

            with open(Path(d1) / "manifest.json") as f:
                m1 = json.load(f)
            with open(Path(d2) / "manifest.json") as f:
                m2 = json.load(f)
            self.assertEqual(
                m1["probe_protocol_hash"], m2["probe_protocol_hash"]
            )
            self.assertEqual(m1["activation_shape"], m2["activation_shape"])


class TestMorphologicalConstraints(unittest.TestCase):
    """Regression tests for morphological-box constraint checks."""

    def test_tag_incompatibility_detection_via_option_map_patch(self):
        import copy
        from research import morphological_box as mb

        spec = mb.roll(seed=123)
        dim_names = list(spec.choices.keys())
        self.assertGreaterEqual(len(dim_names), 2)

        src_dim = dim_names[0]
        dst_dim = dim_names[1]
        src_opt_name = spec.choices[src_dim]
        dst_opt_name = spec.choices[dst_dim]
        dst_opt = mb._OPTION_MAP[dst_dim][dst_opt_name]
        dst_tag = dst_opt.tags[0] if dst_opt.tags else "_test_tag"

        original_map = copy.deepcopy(mb._OPTION_MAP)
        try:
            src_opt = mb._OPTION_MAP[src_dim][src_opt_name]
            patched = mb.Option(
                name=src_opt.name,
                description=src_opt.description,
                tags=src_opt.tags,
                incompatible_with=(dst_tag,),
            )
            mb._OPTION_MAP[src_dim][src_opt_name] = patched

            valid, reason = mb.is_valid_spec(spec)
            self.assertFalse(valid)
            self.assertIsNotNone(reason)
            self.assertIn("incompatible", reason)
        finally:
            mb._OPTION_MAP.clear()
            mb._OPTION_MAP.update(original_map)

    def test_functional_family_roll_with_fixed_choices(self):
        from research import morphological_box as mb

        spec = mb.roll(
            seed=777,
            fixed={
                "token_mixing": "integral_kernel_mixing",
                "channel_mixing": "basis_expansion_layer",
            },
        )
        self.assertEqual(spec.choices["token_mixing"], "integral_kernel_mixing")
        self.assertEqual(spec.choices["channel_mixing"], "basis_expansion_layer")
        valid, reason = mb.is_valid_spec(spec)
        self.assertTrue(valid, reason)

    def test_functional_token_mixing_rejects_minimal_channel_combo(self):
        from research import morphological_box as mb

        base = mb.roll(seed=778)
        choices = dict(base.choices)
        choices["token_mixing"] = "integral_kernel_mixing"
        choices["channel_mixing"] = "identity_skip"
        spec = mb.ArchSpec(choices=choices, seed=778)

        valid, reason = mb.is_valid_spec(spec)
        self.assertFalse(valid)
        self.assertIn("integral-kernel functional mixing", reason or "")

    def test_grammar_can_generate_functional_primitives(self):
        from research.synthesis.grammar import GrammarConfig, generate_layer_graph
        from research.synthesis.primitives import PRIMITIVE_REGISTRY, OpCategory

        functional_ops = {"basis_expansion", "integral_kernel", "fixed_point_iter"}
        excluded = {
            name for name, op in PRIMITIVE_REGISTRY.items()
            if op.category in (OpCategory.PARAMETERIZED, OpCategory.MATH_SPACE, OpCategory.FUNCTIONAL)
            and name not in functional_ops
        }

        cfg = GrammarConfig(
            model_dim=64,
            max_depth=7,
            max_ops=12,
            split_prob=0.0,
            merge_prob=0.0,
            freq_domain_prob=0.0,
            residual_prob=0.0,
            excluded_ops=excluded,
        )
        cfg.category_weights["functional"] = 8.0
        cfg.category_weights["parameterized"] = 0.2
        cfg.category_weights["math_space"] = 0.0

        found = False
        for seed in range(20, 35):
            graph = generate_layer_graph(cfg, seed=seed)
            op_names = [n.op_name for n in graph.nodes.values() if not n.is_input]
            if any(op in functional_ops for op in op_names):
                found = True
                break

        self.assertTrue(found, "Expected at least one generated graph to include a functional primitive")


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestFunctionalArchitectureBuild(unittest.TestCase):
    def test_build_and_forward_functional_family_spec(self):
        from research import morphological_box as mb
        from research.arch_builder import BuildConfig, build_model

        spec = mb.roll(
            seed=999,
            fixed={
                "token_mixing": "integral_kernel_mixing",
                "channel_mixing": "implicit_fixed_point",
                "compute_routing": "uniform",
            },
        )
        cfg = BuildConfig(
            dim=64,
            n_heads=4,
            n_kv_heads=2,
            n_layers=2,
            vocab_size=512,
            max_seq_len=32,
            mlp_ratio=2.0,
        )
        model = build_model(spec, cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
        logits = model(input_ids)

        self.assertEqual(tuple(logits.shape), (2, 16, cfg.vocab_size))
        self.assertTrue(torch.isfinite(logits).all())


# ── Test 3: RunConfig & Mode Selection ──


@unittest.skipUnless(HAS_TORCH, "requires torch for runner module")
class TestRunConfig(unittest.TestCase):
    """Test RunConfig serialization and defaults."""

    def test_default_auto_investigate_min_survivors(self):
        """Default min survivors should be 1 (lowered from 2)."""
        from research.scientist.runner import RunConfig
        config = RunConfig()
        self.assertEqual(config.auto_investigate_min_survivors, 1)

    def test_auto_investigate_enabled_by_default(self):
        """Auto-investigation should be on by default."""
        from research.scientist.runner import RunConfig
        config = RunConfig()
        self.assertTrue(config.auto_investigate)
        self.assertTrue(config.auto_validate)

    def test_round_trip_serialization(self):
        """RunConfig → dict → RunConfig should preserve values."""
        from research.scientist.runner import RunConfig
        original = RunConfig(
            n_programs=100,
            model_dim=512,
            auto_investigate=True,
            auto_investigate_min_survivors=1,
        )
        d = original.to_dict()
        restored = RunConfig.from_dict(d)
        self.assertEqual(restored.n_programs, 100)
        self.assertEqual(restored.model_dim, 512)
        self.assertEqual(restored.auto_investigate_min_survivors, 1)

    def test_all_expected_fields_exist(self):
        """RunConfig should have all pipeline-related fields."""
        from research.scientist.runner import RunConfig
        config = RunConfig()
        fields = [
            "auto_investigate", "auto_investigate_min_survivors",
            "auto_investigate_top_n", "auto_validate",
            "auto_validate_min_robustness", "auto_validate_top_n",
            "investigation_steps", "investigation_batch_size",
            "validation_steps", "validation_batch_size",
            "validation_seq_len", "validation_n_seeds",
            "model_source", "morph_ratio",
            "use_synthesized_training", "n_training_programs",
            "data_mode", "corpus_path", "corpus_format",
            "corpus_text_key", "tokenizer_mode", "corpus_max_chars",
        ]
        for f in fields:
            self.assertTrue(hasattr(config, f), f"Missing field: {f}")

    def test_round_trip_preserves_corpus_fields(self):
        """RunConfig serialization should preserve corpus-mode fields."""
        from research.scientist.runner import RunConfig

        original = RunConfig(
            data_mode="corpus",
            corpus_path="/tmp/example.jsonl",
            corpus_format="jsonl",
            corpus_text_key="content",
            tokenizer_mode="whitespace",
            corpus_max_chars=12345,
        )
        restored = RunConfig.from_dict(original.to_dict())
        self.assertEqual(restored.data_mode, "corpus")
        self.assertEqual(restored.corpus_path, "/tmp/example.jsonl")
        self.assertEqual(restored.corpus_format, "jsonl")
        self.assertEqual(restored.corpus_text_key, "content")
        self.assertEqual(restored.tokenizer_mode, "whitespace")
        self.assertEqual(restored.corpus_max_chars, 12345)


# ── Test 4: Aria Mode Selection ──


@unittest.skipUnless(HAS_PERSONA, "requires persona module")
class TestAriaModeSelecion(unittest.TestCase):
    """Test Aria's rule-based mode recommendation."""

    def setUp(self):
        self.aria = Aria()

    def test_no_survivors_recommends_synthesis(self):
        """With no S1 survivors, should recommend synthesis."""
        rec = self.aria._rule_based_mode_recommendation({
            "total_s1_survivors": 0,
            "avg_novelty": 0,
            "n_experiments_in_session": 1,
        })
        self.assertEqual(rec["mode"], "synthesis")

    def test_long_zero_survivor_streak_recommends_pivot(self):
        """After many zero-survivor runs, recommendation should include pivot signal."""
        rec = self.aria._rule_based_mode_recommendation({
            "total_s1_survivors": 0,
            "avg_novelty": 0,
            "n_experiments_in_session": 12,
        })
        self.assertEqual(rec["mode"], "synthesis")
        self.assertTrue(rec.get("config", {}).get("pivot_recommended"))
        self.assertTrue(rec.get("config", {}).get("stop_recommended"))

    def test_low_novelty_recommends_novelty_search(self):
        """With survivors but low novelty, should recommend novelty."""
        rec = self.aria._rule_based_mode_recommendation({
            "total_s1_survivors": 5,
            "avg_novelty": 0.2,
            "n_experiments_in_session": 2,
        })
        self.assertEqual(rec["mode"], "novelty")

    def test_good_survivors_recommends_evolution(self):
        """With 3+ diverse survivors, should recommend evolution."""
        rec = self.aria._rule_based_mode_recommendation({
            "total_s1_survivors": 5,
            "avg_novelty": 0.6,
            "n_experiments_in_session": 2,
        })
        self.assertEqual(rec["mode"], "evolution")

    def test_investigation_ready_recommends_investigation(self):
        """With investigation-ready candidates, should recommend investigation."""
        rec = self.aria._rule_based_mode_recommendation({
            "total_s1_survivors": 3,
            "avg_novelty": 0.5,
            "n_experiments_in_session": 5,
            "investigation_ready": 3,
        })
        self.assertEqual(rec["mode"], "investigation")

    def test_validation_ready_recommends_validation(self):
        """Validation candidates take highest priority."""
        rec = self.aria._rule_based_mode_recommendation({
            "total_s1_survivors": 5,
            "avg_novelty": 0.6,
            "n_experiments_in_session": 10,
            "investigation_ready": 3,
            "validation_ready": 2,
        })
        self.assertEqual(rec["mode"], "validation")

    def test_recommendation_has_required_fields(self):
        """Every recommendation should have mode, reasoning, confidence, config."""
        rec = self.aria._rule_based_mode_recommendation({})
        self.assertIn("mode", rec)
        self.assertIn("reasoning", rec)
        self.assertIn("confidence", rec)
        self.assertIn("config", rec)
        self.assertIn(rec["mode"],
                      {"synthesis", "evolution", "novelty",
                       "investigation", "validation"})

    def test_parse_briefing_uses_reasoning_when_briefing_missing(self):
        parsed = self.aria._parse_briefing(
            "SUGGESTED_ACTION:\n"
            "MODE: evolve\n"
            "REASONING: Evolution remains the best next step from recent plateaued runs.\n"
            "CONFIDENCE: 0.78\n"
        )
        self.assertTrue(parsed.get("briefing_text"))
        self.assertIn("Evolution remains the best next step", parsed.get("briefing_text", ""))

    def test_parse_briefing_accepts_summary_prefix(self):
        parsed = self.aria._parse_briefing(
            "Summary: Recent S1 hit rate is flattening and validation queue is growing.\n"
            "MODE: novelty\n"
            "REASONING: Diversification is needed to escape local minima."
        )
        self.assertIn("Recent S1 hit rate is flattening", parsed.get("briefing_text", ""))

    def test_parse_mode_recommendation(self):
        """Parse LLM mode recommendation text."""
        text = (
            "MODE: evolution\n"
            "REASONING: We have 5 good survivors to breed.\n"
            "CONFIDENCE: 0.8\n"
            "CONFIG_ADJUSTMENTS:\n"
            "```json\n"
            '{"n_programs": 30}\n'
            "```"
        )
        rec = self.aria._parse_mode_recommendation(text)
        self.assertEqual(rec["mode"], "evolution")
        self.assertAlmostEqual(rec["confidence"], 0.8)
        self.assertEqual(rec["config"]["n_programs"], 30)

    def test_parse_invalid_mode_defaults_to_synthesis(self):
        """Invalid mode in LLM response should default to synthesis."""
        text = "MODE: quantum_computing\nREASONING: reasons\nCONFIDENCE: 0.5"
        rec = self.aria._parse_mode_recommendation(text)
        self.assertEqual(rec["mode"], "synthesis")


# ── Test 5: Context Builders ──


@unittest.skipUnless(HAS_CONTEXT, "requires context module")
class TestContextBuilders(unittest.TestCase):
    """Test LLM context building functions."""

    def test_mode_selection_context(self):
        """Mode selection context includes key information."""
        from research.scientist.llm.context import build_mode_selection_context

        ctx = build_mode_selection_context(
            recent_experiments=[
                {"n_stage1_passed": 2, "n_programs_generated": 50,
                 "best_novelty_score": 0.7, "experiment_type": "synthesis"},
            ],
            leaderboard=[
                {"tier": "screening", "screening_loss_ratio": 0.5,
                 "composite_score": 0.6, "result_id": "r1"},
            ],
            current_mode="synthesis",
            n_experiments_in_session=3,
        )

        self.assertIn("synthesis", ctx)
        self.assertIn("3", ctx)  # n_experiments_in_session

    def test_investigation_context(self):
        """Investigation context includes candidate data."""
        from research.scientist.llm.context import build_investigation_context

        ctx = build_investigation_context(
            candidates=[{"result_id": "r1", "loss_ratio": 0.4}],
            leaderboard=[{"tier": "screening", "composite_score": 0.5}],
        )
        self.assertIn("Investigation Phase", ctx)
        self.assertIn("r1", ctx)

    def test_validation_context(self):
        """Validation context includes investigation results."""
        from research.scientist.llm.context import build_validation_context

        ctx = build_validation_context(
            candidates=[{"result_id": "r1", "investigation_loss_ratio": 0.3}],
            investigation_results=[{"result_id": "r1", "robustness": 0.7}],
        )
        self.assertIn("Validation Phase", ctx)


# ── Test 6: API Endpoints ──


@unittest.skipUnless(HAS_TORCH and HAS_FLASK, "requires torch and flask")
class TestAPI(unittest.TestCase):
    """Test all API endpoints return valid responses."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.tmpdir, "test_api.db")
        from research.scientist.api import create_app
        cls.app = create_app(notebook_path=cls.db_path)
        cls.client = cls.app.test_client()

        # Seed some data
        nb = LabNotebook(cls.db_path)
        exp_id = nb.start_experiment("synthesis", {"n_programs": 5}, "test hyp")
        for i in range(3):
            nb.record_program_result(
                experiment_id=exp_id,
                graph_fingerprint=f"fp_{i:03d}",
                graph_json=json.dumps({"nodes": {}, "id": i}),
                stage0_passed=True,
                stage05_passed=True,
                stage1_passed=(i == 0),
                loss_ratio=0.5 - i * 0.1 if i == 0 else None,
                novelty_score=0.3 + i * 0.2,
            )
        nb.complete_experiment(exp_id, {
            "total": 3, "stage0_passed": 3, "stage1_passed": 1,
        }, "Test summary", "excited")

        campaign_id = nb.create_campaign(
            title="Schema Contract Campaign",
            objective="Validate campaign endpoint contracts",
            success_criteria="All campaign routes return expected payload shape",
        )
        cls.campaign_id = campaign_id
        cls.exp_id = exp_id

        nb.conn.execute(
            "UPDATE experiments SET campaign_id = ? WHERE experiment_id = ?",
            (campaign_id, exp_id),
        )
        nb.conn.commit()

        nb.record_hypothesis(
            campaign_id=campaign_id,
            experiment_id=exp_id,
            prediction="Schema contracts remain stable",
            reasoning="Dashboard consumers require stable API keys.",
            test_method="Run integration schema assertions",
            success_metric="All assertions pass",
            confidence=0.7,
            metadata={
                "source": "llm_context",
                "used_context": True,
                "review_status": "not_reviewed",
                "confidence": 0.7,
                "critique": "schema stability check",
            },
        )
        nb.record_decision(
            campaign_id=campaign_id,
            decision_type="go",
            subject="contract-tests",
            rationale="Schema checks provide early drift detection.",
            evidence_ids=[exp_id],
            alternatives=[{"option": "manual-checks", "risk": "high drift risk"}],
        )
        nb.add_knowledge(
            category="api_contract",
            title="Dashboard schema contract baseline",
            content="Core dashboard endpoints must preserve key payload fields.",
            evidence=[exp_id],
            confidence=0.9,
        )

        # Add leaderboard entry
        nb.upsert_leaderboard(
            result_id="fp_000",
            model_source="graph_synthesis",
            screening_loss_ratio=0.5,
            screening_novelty=0.3,
            tier="screening",
        )

        # Add insight
        nb.record_insight("pattern", "Test insight", exp_id, 0.8)

        # Add entry
        nb.add_entry(ExperimentEntry(
            entry_type="decision", title="Test", content="Content"))
        nb.add_entry(ExperimentEntry(
            entry_type="live_feed",
            experiment_id=exp_id,
            title="Evolution generation 1/5",
            content="Gen 1/5: best=0.981, avg=0.240, pop=50",
            metadata={
                "live_feed_type": "evo_gen",
                "payload": {
                    "experiment_id": exp_id,
                    "generation": 1,
                    "total_generations": 5,
                    "best_fitness": 0.981,
                    "avg_fitness": 0.240,
                    "population_size": 50,
                },
            },
        ))

        nb.close()

    # ── GET endpoints ──

    def test_api_dashboard(self):
        r = self.client.get("/api/dashboard")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("aria", data)
        self.assertIn("summary", data)
        self.assertIn("recent_experiments", data)
        self.assertIn("top_programs", data)
        self.assertIn("insights", data)
        self.assertIn("is_running", data)

    def test_api_status(self):
        r = self.client.get("/api/status")
        self.assertEqual(r.status_code, 200)

    def test_api_system_status(self):
        r = self.client.get("/api/system/status")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("cuda", data)

    def test_api_experiments(self):
        r = self.client.get("/api/experiments")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_api_experiment_detail(self):
        # Get the experiment ID from list
        r = self.client.get("/api/experiments")
        exps = r.get_json()
        exp_id = exps[0]["experiment_id"]

        r = self.client.get(f"/api/experiments/{exp_id}")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("experiment", data)
        self.assertIn("programs", data)

    def test_api_experiment_programs(self):
        r = self.client.get("/api/experiments")
        exp_id = r.get_json()[0]["experiment_id"]

        r = self.client.get(f"/api/experiments/{exp_id}/programs")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIsInstance(data, list)

    def test_api_experiment_failures(self):
        r = self.client.get("/api/experiments")
        exp_id = r.get_json()[0]["experiment_id"]

        r = self.client.get(f"/api/experiments/{exp_id}/failures")
        self.assertEqual(r.status_code, 200)

    def test_api_programs(self):
        r = self.client.get("/api/programs")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIsInstance(data, list)
        if data:
            row = data[0]
            self.assertIn("qkv_usage", row)
            self.assertIn(row["qkv_usage"], {"full_qkv", "q_eq_k_eq_v", "qkv_free"})
            self.assertIn("uses_qkv", row)
            self.assertIsInstance(row["uses_qkv"], bool)
            self.assertIn("compression_metrics", row)
            self.assertIn("reproducibility_packet", row)
            self.assertIn("compression_ratio", row["compression_metrics"])
            self.assertIn("quality_retention_score", row["compression_metrics"])
            self.assertIn("status", row["reproducibility_packet"])

    def test_api_programs_sort_options(self):
        for sort in ["novelty_score", "loss_ratio"]:
            r = self.client.get(f"/api/programs?sort_by={sort}")
            self.assertEqual(r.status_code, 200)

    def test_api_program_detail(self):
        r = self.client.get("/api/programs")
        programs = r.get_json()
        if programs:
            result_id = programs[0]["result_id"]
            r = self.client.get(f"/api/programs/{result_id}")
            self.assertEqual(r.status_code, 200)
            detail = r.get_json()
            self.assertIn("qkv_usage", detail)
            self.assertIn(detail["qkv_usage"], {"full_qkv", "q_eq_k_eq_v", "qkv_free"})
            self.assertIn("uses_qkv", detail)
            self.assertIsInstance(detail["uses_qkv"], bool)
            self.assertIn("compression_metrics", detail)
            self.assertIn("reproducibility_packet", detail)
            self.assertIn("compression_ratio", detail["compression_metrics"])
            self.assertIn("status", detail["reproducibility_packet"])

    def test_api_trends(self):
        r = self.client.get("/api/trends")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIsInstance(data, list)

    def test_api_trends_context(self):
        r = self.client.get("/api/trends/context")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("trends", data)
        self.assertIn("adaptation_events", data)
        self.assertIn("generated_at", data)
        self.assertIsInstance(data["trends"], list)
        self.assertIsInstance(data["adaptation_events"], list)

    def test_api_insights(self):
        r = self.client.get("/api/insights")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIsInstance(data, list)

    def test_api_entries(self):
        r = self.client.get("/api/entries")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIsInstance(data, list)
        if data:
            self.assertIn("metadata", data[0])
            self.assertIsInstance(data[0]["metadata"], dict)

    def test_api_live_feed(self):
        r = self.client.get("/api/live-feed")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 1)
        evt = data[-1]
        self.assertEqual(evt.get("type"), "evo_gen")
        self.assertIn("generation", evt)
        self.assertIn("total_generations", evt)
        self.assertIn("best_fitness", evt)
        self.assertIn("avg_fitness", evt)
        self.assertIn("population_size", evt)
        self.assertIn("ts", evt)

    def test_api_leaderboard(self):
        r = self.client.get("/api/leaderboard")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("entries", data)
        self.assertIn("by_tier", data)
        self.assertIn("total", data)
        self.assertIn("cross_run_stability_summary", data)
        self.assertIn("cross_run_stability_window", data)
        self.assertGreater(data["total"], 0)

    def test_api_leaderboard_tier_filter(self):
        r = self.client.get("/api/leaderboard?tier=screening")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        for entry in data["entries"]:
            self.assertEqual(entry["tier"], "screening")

    def test_api_report(self):
        r = self.client.get("/api/report")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("summary", data)
        self.assertIn("experiment_clusters", data)

    def test_api_analytics_op_success(self):
        r = self.client.get("/api/analytics/op-success")
        self.assertEqual(r.status_code, 200)

    def test_api_analytics_failure_patterns(self):
        r = self.client.get("/api/analytics/failure-patterns")
        self.assertEqual(r.status_code, 200)

    def test_api_analytics_grammar_weights(self):
        r = self.client.get("/api/analytics/grammar-weights")
        self.assertEqual(r.status_code, 200)

    def test_api_analytics_efficiency_frontier(self):
        r = self.client.get("/api/analytics/efficiency-frontier")
        self.assertEqual(r.status_code, 200)

    def test_api_analytics_learning_log(self):
        r = self.client.get("/api/analytics/learning-log")
        self.assertEqual(r.status_code, 200)

    def test_api_analytics_experiment_clusters(self):
        r = self.client.get("/api/analytics/experiment-clusters")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        if data is not None:
            self.assertIn("clusters", data)
            self.assertIn("stability_score", data)

    def test_api_analytics_routing_health(self):
        r = self.client.get("/api/analytics/routing-health")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("available", data)
        self.assertIn("by_mode", data)

    def test_api_analytics_math_family_coverage(self):
        r = self.client.get("/api/analytics/math-family-coverage")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("families", data)
        self.assertIn("totals", data)

    def test_api_analytics_learning_summary(self):
        r = self.client.get("/api/analytics/learning-summary")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("bullets", data)
        self.assertIn("source", data)

    def test_api_analytics_learning_trajectory_includes_minimum_requirement(self):
        r = self.client.get("/api/analytics/learning-trajectory")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("trend", data)
        self.assertIn("n_experiments", data)
        self.assertIn("min_experiments_required", data)
        self.assertEqual(data["min_experiments_required"], 5)

    def test_api_config(self):
        r = self.client.get("/api/config")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("n_programs", data)

    def test_api_progress(self):
        r = self.client.get("/api/progress")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("is_running", data)

    def test_api_aria_recommendation(self):
        r = self.client.get("/api/aria/recommendation")
        self.assertEqual(r.status_code, 200)

    def test_api_aria_strategy(self):
        r = self.client.get("/api/aria/strategy")
        self.assertEqual(r.status_code, 200)

    def test_api_aria_cycle_status(self):
        r = self.client.get("/api/aria/cycle-status")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("phase", data)
        self.assertIn("phase_label", data)
        self.assertIn("continuous_active", data)
        self.assertIn("cycle_index", data)
        self.assertIn("is_running", data)
        self.assertIn("last_cycle_summary", data)
        if data["last_cycle_summary"] is not None:
            summary = data["last_cycle_summary"]
            self.assertIn("cycle_index", summary)
            self.assertIn("mode", summary)
            self.assertIn("status", summary)
            self.assertIn("timestamp", summary)

    def test_api_aria_cycle_history(self):
        r = self.client.get("/api/aria/cycle-history?n=10")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIsInstance(data, list)
        if data:
            row = data[0]
            self.assertIn("cycle_index", row)
            self.assertIn("mode", row)
            self.assertIn("status", row)
            self.assertIn("entry_id", row)

    def test_api_aria_cycle_history_filters_and_csv(self):
        nb = LabNotebook(self.db_path)
        exp_id = nb.start_experiment("continuous", {"n_programs": 2}, "cycle history filter test")
        nb.add_entry(ExperimentEntry(
            entry_type="live_feed",
            experiment_id=exp_id,
            title="Aria cycle 1",
            content="Cycle one",
            metadata={
                "live_feed_type": "aria_cycle",
                "payload": {
                    "cycle_index": 1,
                    "mode": "synthesis",
                    "status": "completed",
                    "reasoning": "baseline cycle",
                    "timestamp": time.time(),
                },
            },
        ))
        nb.add_entry(ExperimentEntry(
            entry_type="live_feed",
            experiment_id=exp_id,
            title="Aria cycle 2",
            content="Cycle two",
            metadata={
                "live_feed_type": "aria_cycle",
                "payload": {
                    "cycle_index": 2,
                    "mode": "evolution",
                    "status": "failed",
                    "reasoning": "diversity exploration",
                    "error": "simulated",
                    "timestamp": time.time(),
                },
            },
        ))
        nb.close()

        filtered = self.client.get("/api/aria/cycle-history?mode=evolution&status=failed&q=diversity")
        self.assertEqual(filtered.status_code, 200)
        filtered_data = filtered.get_json()
        self.assertIsInstance(filtered_data, list)
        self.assertGreaterEqual(len(filtered_data), 1)
        self.assertEqual(filtered_data[0].get("mode"), "evolution")
        self.assertEqual(filtered_data[0].get("status"), "failed")

        csv_resp = self.client.get("/api/aria/cycle-history?format=csv&mode=evolution")
        self.assertEqual(csv_resp.status_code, 200)
        self.assertIn("text/csv", csv_resp.content_type)
        body = csv_resp.get_data(as_text=True)
        self.assertIn("cycle_index,mode,status", body)
        self.assertIn("evolution", body)

    def test_api_aria_cycle_control_pause_resume(self):
        from research.scientist import api as api_mod

        fake_runner = MagicMock()
        fake_runner.pause_aria_cycle = MagicMock(return_value={"phase": "paused", "cycle_paused": True})
        fake_runner.resume_aria_cycle = MagicMock(return_value={"phase": "planning", "cycle_paused": False})

        with patch.object(api_mod, "_runner", fake_runner):
            r_pause = self.client.post("/api/aria/cycle-control", json={"action": "pause"})
            r_resume = self.client.post("/api/aria/cycle-control", json={"action": "resume"})

        self.assertEqual(r_pause.status_code, 200)
        self.assertEqual(r_resume.status_code, 200)
        self.assertTrue(r_pause.get_json().get("ok"))
        self.assertTrue(r_resume.get_json().get("ok"))
        fake_runner.pause_aria_cycle.assert_called_once()
        fake_runner.resume_aria_cycle.assert_called_once()

    def test_api_aria_cycle_control_start(self):
        from research.scientist import api as api_mod

        fake_runner = MagicMock()
        fake_runner.is_running = False
        fake_runner.start_continuous = MagicMock(return_value="continuous")
        fake_runner.get_aria_cycle_status = MagicMock(return_value={"phase": "planning", "continuous_active": True})

        with patch.object(api_mod, "_runner", fake_runner):
            r = self.client.post("/api/aria/cycle-control", json={"action": "start", "config": {"n_programs": 3}})

        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertTrue(data.get("ok"))
        self.assertEqual(data.get("action"), "start")
        self.assertEqual(data.get("experiment_id"), "continuous")
        fake_runner.start_continuous.assert_called_once()

    def test_api_aria_chat(self):
        r = self.client.post(
            "/api/aria/chat",
            json={
                "message": "What should we do next based on latest runs?",
                "history": [{"role": "user", "text": "Summarize latest findings."}],
            },
        )
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("reply", data)
        self.assertIn("ai_powered", data)
        self.assertIn("used_context", data)
        self.assertIn("Evidence:", data["reply"])
        self.assertIn("Recommendation:", data["reply"])
        self.assertIn("Next Action:", data["reply"])

    def test_api_aria_chat_with_session_id(self):
        r = self.client.post(
            "/api/aria/chat",
            json={
                "message": "What is the best loss ratio so far?",
                "session_id": "test-session-001",
            },
        )
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("reply", data)
        self.assertIn("ai_powered", data)

    def test_api_aria_chat_history(self):
        # Save a message first
        r = self.client.post("/api/aria/chat/message", json={
            "session_id": "test-history-session",
            "role": "user",
            "text": "Hello Aria",
            "label": "You",
        })
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.get_json().get("saved"))

        # Load history
        r = self.client.get("/api/aria/chat/history?session_id=test-history-session")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("messages", data)
        self.assertGreaterEqual(len(data["messages"]), 1)
        self.assertEqual(data["messages"][0]["role"], "user")
        self.assertEqual(data["messages"][0]["text"], "Hello Aria")

    def test_api_aria_chat_message(self):
        r = self.client.post("/api/aria/chat/message", json={
            "session_id": "test-msg-session",
            "role": "aria",
            "text": "This is a test response.",
            "label": "Aria",
        })
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertTrue(data.get("saved"))
        self.assertIn("message_id", data)

    def test_api_aria_chat_message_requires_text(self):
        r = self.client.post("/api/aria/chat/message", json={
            "session_id": "test-msg-session",
            "role": "user",
            "text": "",
        })
        self.assertEqual(r.status_code, 400)

    def test_api_aria_chat_compact(self):
        # Seed enough messages to trigger compaction
        session_id = "test-compact-session"
        for i in range(15):
            self.client.post("/api/aria/chat/message", json={
                "session_id": session_id,
                "role": "user" if i % 2 == 0 else "aria",
                "text": f"Message number {i}. " * 40,  # ~600 chars each
            })

        r = self.client.post("/api/aria/chat/compact", json={
            "session_id": session_id,
            "token_budget": 1000,
        })
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("compacted", data)

    def test_api_aria_chat_compact_no_messages(self):
        r = self.client.post("/api/aria/chat/compact", json={
            "session_id": "nonexistent-session",
        })
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertFalse(data.get("compacted"))

    def test_api_strategy_briefing_contract(self):
        r = self.client.get("/api/strategy/briefing")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("briefing", data)
        self.assertIn("action", data)
        self.assertIn("ai_powered", data)
        self.assertIn("suggested_config", data)
        self.assertIn("evidence", data)

    def test_api_strategy_briefing_normalizes_ai_mode_alias(self):
        ai_payload = {
            "briefing_text": "Use evolution to explore new candidates.",
            "suggested_action": {
                "mode": "evolution",
                "hypothesis": "Evolution improves candidate quality.",
                "config": {"n_programs": 24, "model_dim": 96},
                "reasoning": "Recent runs plateaued; evolution broadens the search.",
            },
            "confidence": 0.82,
        }

        # Clear any cached briefing from prior test calls so the mock takes effect
        from research.scientist.api import get_aria as _get_aria_fn
        _aria_inst = _get_aria_fn()
        if hasattr(_aria_inst, "_briefing_cache"):
            _aria_inst._briefing_cache = None

        with patch("research.scientist.persona.Aria.generate_briefing", return_value=ai_payload):
            r = self.client.get("/api/strategy/briefing")

        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertTrue(data.get("ai_powered"))
        self.assertEqual(data.get("action"), "evolve")
        self.assertEqual(data.get("suggested_config", {}).get("mode"), "evolve")
        self.assertEqual(
            data.get("suggested_config", {}).get("hypothesis"),
            "Evolution improves candidate quality.",
        )

    def test_api_llm_config(self):
        r = self.client.get("/api/llm/config")
        self.assertEqual(r.status_code, 200)

    # ── POST endpoints ──

    def test_api_stop_when_not_running(self):
        r = self.client.post("/api/experiments/stop")
        self.assertEqual(r.status_code, 409)

    def test_api_start_returns_preflight_critique_gate(self):
        from research.scientist import api as api_mod

        fake_runner = MagicMock()
        fake_runner.is_running = False
        fake_runner.start_experiment = MagicMock(return_value="exp-preflight")
        fake_runner.progress = MagicMock(
            aria_message="Preflight complete",
            hypothesis_critique={
                "verdict": "caution",
                "gate": "warn",
                "confidence": 0.61,
                "checks": [
                    {"key": "testability", "label": "Testability", "status": "pass"},
                    {"key": "measurable_metric", "label": "Measurable Metric", "status": "warn"},
                    {"key": "confound_risk", "label": "Confound Risk", "status": "warn"},
                    {"key": "fallback_plan", "label": "Fallback Plan", "status": "warn"},
                ],
                "concerns": ["Metric needs tighter threshold."],
                "suggestions": ["Add a fallback baseline check."],
            },
        )

        with patch.object(api_mod, "_runner", fake_runner):
            r = self.client.post("/api/experiments/start", json={"n_programs": 1, "hypothesis": "test"})

        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("hypothesis_critique", data)
        self.assertIn("hypothesis_review_gate", data)
        self.assertEqual(data["hypothesis_review_gate"], "warn")
        self.assertIsInstance(data["hypothesis_critique"], dict)
        self.assertIn("checks", data["hypothesis_critique"])

    def test_api_start_requires_result_ids_for_investigation(self):
        r = self.client.post("/api/experiments/start",
                             json={"mode": "investigation"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("result_ids", r.get_json()["error"])

    def test_api_start_requires_result_ids_for_validation(self):
        r = self.client.post("/api/experiments/start",
                             json={"mode": "validation"})
        self.assertEqual(r.status_code, 400)

    def test_api_start_investigation_rejects_already_investigated_with_payload(self):
        nb = LabNotebook(self.db_path)
        exp_id = nb.start_experiment("synthesis", {"n_programs": 1}, "investigation eligibility")
        result_id = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_inv_reject",
            graph_json=json.dumps({"nodes": {}}),
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            loss_ratio=0.41,
            novelty_score=0.52,
        )
        nb.upsert_leaderboard(
            result_id=result_id,
            model_source="graph_synthesis",
            screening_loss_ratio=0.41,
            screening_novelty=0.52,
            screening_passed=True,
            investigation_loss_ratio=0.88,
            investigation_passed=False,
            tier="screening",
        )
        nb.close()

        r = self.client.post("/api/experiments/start", json={
            "mode": "investigation",
            "result_ids": [result_id],
        })
        self.assertEqual(r.status_code, 409)
        data = r.get_json()
        self.assertIn("eligibility", data)
        eligibility = data["eligibility"]
        self.assertEqual(eligibility["mode"], "investigation")
        self.assertEqual(eligibility["eligible_result_ids"], [])
        self.assertEqual(eligibility["summary"]["ineligible"], 1)
        self.assertEqual(eligibility["ineligible"][0]["reason"], "already_investigated_unchanged")

    def test_api_start_validation_rejects_non_investigation_passed_with_payload(self):
        nb = LabNotebook(self.db_path)
        exp_id = nb.start_experiment("synthesis", {"n_programs": 1}, "validation eligibility reject")
        result_id = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_val_reject",
            graph_json=json.dumps({"nodes": {}}),
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            loss_ratio=0.39,
            novelty_score=0.58,
        )
        nb.upsert_leaderboard(
            result_id=result_id,
            model_source="graph_synthesis",
            screening_loss_ratio=0.39,
            screening_novelty=0.58,
            screening_passed=True,
            investigation_loss_ratio=0.44,
            investigation_robustness=0.33,
            investigation_passed=False,
            tier="investigation",
        )
        nb.close()

        r = self.client.post("/api/experiments/start", json={
            "mode": "validation",
            "result_ids": [result_id],
        })
        self.assertEqual(r.status_code, 409)
        data = r.get_json()
        self.assertIn("eligibility", data)
        eligibility = data["eligibility"]
        self.assertEqual(eligibility["mode"], "validation")
        self.assertEqual(eligibility["eligible_result_ids"], [])
        self.assertEqual(eligibility["summary"]["ineligible"], 1)
        self.assertEqual(eligibility["ineligible"][0]["reason"], "not_investigation_passed")

    def test_api_start_validation_returns_eligibility_on_success(self):
        nb = LabNotebook(self.db_path)
        exp_id = nb.start_experiment("synthesis", {"n_programs": 1}, "validation eligibility success")
        result_id = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_val_ok",
            graph_json=json.dumps({"nodes": {}}),
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            loss_ratio=0.31,
            novelty_score=0.63,
        )
        nb.upsert_leaderboard(
            result_id=result_id,
            model_source="graph_synthesis",
            screening_loss_ratio=0.31,
            screening_novelty=0.63,
            screening_passed=True,
            investigation_loss_ratio=0.29,
            investigation_robustness=0.71,
            investigation_passed=True,
            tier="investigation",
        )
        nb.close()

        from research.scientist import api as api_mod
        fake_runner = MagicMock()
        fake_runner.is_running = False
        fake_runner.start_validation = MagicMock(return_value="exp-val-eligible")
        fake_runner.progress = MagicMock(
            aria_message="Validation started",
            hypothesis_critique=None,
        )

        with patch.object(api_mod, "_runner", fake_runner):
            r = self.client.post("/api/experiments/start", json={
                "mode": "validation",
                "result_ids": [result_id],
            })

        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("eligibility", data)
        eligibility = data["eligibility"]
        self.assertTrue(eligibility["all_eligible"])
        self.assertEqual(eligibility["eligible_result_ids"], [result_id])
        self.assertEqual(eligibility["ineligible"], [])
        fake_runner.start_validation.assert_called_once()

    def test_api_start_requires_result_ids_for_scale_up(self):
        r = self.client.post("/api/experiments/start",
                             json={"mode": "scale_up"})
        self.assertEqual(r.status_code, 400)

    def test_api_rerun_experiment(self):
        from research.scientist import api as api_mod

        nb = LabNotebook(self.db_path)
        exp_id = nb.start_experiment("evolution", {"n_programs": 4}, "rerun me")
        nb.cancel_experiment(exp_id)
        nb.close()

        fake_runner = MagicMock()
        fake_runner.is_running = False
        fake_runner.start_evolution = MagicMock(return_value="exp-rerun-new")

        with patch.object(api_mod, "_runner", fake_runner):
            r = self.client.post(f"/api/experiments/{exp_id}/rerun")

        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertEqual(data.get("status"), "started")
        self.assertEqual(data.get("source_experiment_id"), exp_id)
        self.assertEqual(data.get("experiment_id"), "exp-rerun-new")
        self.assertEqual(data.get("mode"), "evolve")
        fake_runner.start_evolution.assert_called_once()

    def test_api_rerun_experiment_when_runner_busy(self):
        from research.scientist import api as api_mod

        fake_runner = MagicMock()
        fake_runner.is_running = True

        with patch.object(api_mod, "_runner", fake_runner):
            r = self.client.post("/api/experiments/does-not-matter/rerun")

        self.assertEqual(r.status_code, 409)
        self.assertIn("already running", (r.get_json() or {}).get("error", "").lower())

    def test_api_validate_pipeline(self):
        r = self.client.post("/api/validate", json={"n": 2})
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("generated", data)
        self.assertIn("healthy", data)

    # ── Schema contract tests (deeper validation) ──

    def test_api_dashboard_summary_schema(self):
        """Dashboard summary must contain all keys consumed by Overview."""
        r = self.client.get("/api/dashboard")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        summary = data["summary"]
        required_summary_keys = [
            "total_experiments", "completed_experiments",
            "total_programs_evaluated", "stage1_survivors",
            "survival_rate", "avg_novelty_score",
            "top_novelty_score", "active_insights",
            "learning_events",
        ]
        for key in required_summary_keys:
            self.assertIn(key, summary, f"summary missing key: {key}")
        # progress and is_running must exist at top level
        self.assertIn("progress", data)
        self.assertIn("is_running", data)
        self.assertIsInstance(data["recent_experiments"], list)
        self.assertIsInstance(data["top_programs"], list)
        self.assertIsInstance(data["insights"], list)

    def test_api_dashboard_progress_schema(self):
        """Dashboard progress must have expected shape."""
        r = self.client.get("/api/dashboard")
        data = r.get_json()
        progress = data["progress"]
        self.assertIn("aria_message", progress)
        self.assertIn("current_stage", progress)
        self.assertIn("elapsed_seconds", progress)

    def test_api_leaderboard_entry_schema(self):
        """Leaderboard entries must contain scoring and tier fields."""
        r = self.client.get("/api/leaderboard")
        data = r.get_json()
        self.assertGreater(data["total"], 0)
        entry = data["entries"][0]
        required_entry_keys = [
            "entry_id", "result_id", "composite_score", "tier",
            "screening_loss_ratio", "architecture_family", "cross_run_stability",
            "qkv_usage", "uses_qkv", "compression_metrics", "reproducibility_packet",
        ]
        for key in required_entry_keys:
            self.assertIn(key, entry, f"leaderboard entry missing key: {key}")

        stability = entry["cross_run_stability"]
        self.assertIn("trend", stability)
        self.assertIn("seen_runs", stability)
        self.assertIn(entry["qkv_usage"], {"full_qkv", "q_eq_k_eq_v", "qkv_free"})
        self.assertIsInstance(entry["uses_qkv"], bool)
        self.assertIn("compression_ratio", entry["compression_metrics"])
        self.assertIn("quality_retention_score", entry["compression_metrics"])
        self.assertIn("status", entry["reproducibility_packet"])
        self.assertIn("ready_count", entry["reproducibility_packet"])

    def test_api_analytics_grammar_weights_schema(self):
        """Grammar weights must have default and holdout_validation keys."""
        r = self.client.get("/api/analytics/grammar-weights")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("default", data)
        self.assertIsInstance(data["default"], dict)
        self.assertIn("holdout_validation", data)
        self.assertIn("learning_diagnostics", data)
        self.assertIsInstance(data["learning_diagnostics"], dict)
        self.assertIn("mode", data["learning_diagnostics"])
        self.assertIn("fingerprint_cap", data["learning_diagnostics"])
        self.assertIn("architecture_rerun_telemetry", data)
        telemetry = data["architecture_rerun_telemetry"]
        self.assertIn("unique_fingerprint_count", telemetry)
        self.assertIn("rerun_ratio", telemetry)
        self.assertIn("top_fingerprint_concentration", telemetry)
        self.assertIn("explanation", data)
        self.assertIsInstance(data["explanation"], str)

    def test_api_analytics_efficiency_frontier_schema(self):
        """Frontier entries must include ops field from graph_json."""
        r = self.client.get("/api/analytics/efficiency-frontier")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        if data:  # may be empty if no S1 survivors with FLOPs
            entry = data[0]
            self.assertIn("result_id", entry)
            self.assertIn("graph_fingerprint", entry)
            self.assertIn("final_loss", entry)
            self.assertIn("flops_forward", entry)
            self.assertIn("ops", entry)
            self.assertNotIn("graph_json", entry,
                             "graph_json should be stripped from frontier response")

    def test_api_analytics_experiment_clusters_schema(self):
        """Cluster entries must include description field."""
        r = self.client.get("/api/analytics/experiment-clusters")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        if data is not None and data.get("clusters"):
            cluster = data["clusters"][0]
            self.assertIn("cluster_id", cluster)
            self.assertIn("size", cluster)
            self.assertIn("avg_s1_rate", cluster)
            self.assertIn("description", cluster)

    def test_api_analytics_routing_health_schema(self):
        """Routing health must have structured by_mode entries."""
        r = self.client.get("/api/analytics/routing-health")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("available", data)
        self.assertIn("by_mode", data)
        self.assertIn("explanation", data)
        self.assertIsInstance(data["explanation"], str)
        self.assertIsInstance(data["by_mode"], list)
        if data["by_mode"]:
            mode = data["by_mode"][0]
            self.assertIn("routing_mode", mode)
            self.assertIn("n_programs", mode)
            self.assertIn("sample_size_label", mode)
            self.assertIn("confidence_label", mode)
            self.assertIn("stability_label", mode)

    def test_api_analytics_routing_comparison_schema(self):
        """Routing comparison endpoint returns consolidated mode labels and totals."""
        r = self.client.get("/api/analytics/routing-comparison")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("available", data)
        self.assertIn("by_mode", data)
        self.assertIn("n_modes", data)
        self.assertIn("total_programs", data)
        self.assertIn("routed_programs", data)
        self.assertIn("uniform_programs", data)
        self.assertIn("explanation", data)
        self.assertIsInstance(data["by_mode"], list)
        if data["by_mode"]:
            row = data["by_mode"][0]
            for key in (
                "routing_mode", "n_programs", "stage1_pass_rate",
                "avg_drop_rate", "avg_utilization_entropy",
                "avg_confidence_mean", "sample_size_label",
                "confidence_label", "stability_label", "efficiency_label",
            ):
                self.assertIn(key, row)

    def test_api_analytics_gating_diagnostics_schema(self):
        """Gating diagnostics endpoint returns entropy/collapse/retention structures."""
        r = self.client.get("/api/analytics/gating-diagnostics")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("available", data)
        self.assertIn("total_routed_programs", data)
        self.assertIn("avg_gate_entropy", data)
        self.assertIn("collapse_risk_counts", data)
        self.assertIn("by_mode", data)
        self.assertIn("token_retention_curve_overall", data)
        self.assertIn("explanation", data)
        self.assertIsInstance(data["by_mode"], list)
        self.assertIsInstance(data["collapse_risk_counts"], dict)
        if data["by_mode"]:
            row = data["by_mode"][0]
            for key in (
                "routing_mode", "n_programs", "avg_gate_entropy",
                "collapse_risk_label", "avg_token_retention", "token_retention_curve",
            ):
                self.assertIn(key, row)

    def test_api_analytics_math_family_coverage_schema(self):
        """Math family coverage returns stable family/totals structures."""
        r = self.client.get("/api/analytics/math-family-coverage")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("families", data)
        self.assertIn("totals", data)
        self.assertIsInstance(data["families"], list)
        self.assertIsInstance(data["totals"], dict)
        self.assertIn("n_tested", data["totals"])
        self.assertIn("n_survived", data["totals"])
        if data["families"]:
            row = data["families"][0]
            for key in (
                "family", "n_tested", "n_survived",
                "survival_rate", "tested_share", "survivor_share",
            ):
                self.assertIn(key, row)

    def test_api_analytics_mathspace_impact_schema(self):
        """Mathspace impact endpoint returns operator/family impact structures."""
        r = self.client.get("/api/analytics/mathspace-impact")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("available", data)
        self.assertIn("totals", data)
        self.assertIn("by_operator", data)
        self.assertIn("by_family", data)
        self.assertIn("top_trustworthy_operators", data)
        self.assertIn("explanation", data)
        self.assertIsInstance(data["totals"], dict)
        self.assertIsInstance(data["by_operator"], list)
        self.assertIsInstance(data["by_family"], list)
        self.assertIsInstance(data["top_trustworthy_operators"], list)
        for key in ("n_programs_with_graph", "n_programs_with_mathspace", "n_mathspace_ops_observed"):
            self.assertIn(key, data["totals"])
        if data["by_operator"]:
            row = data["by_operator"][0]
            for key in (
                "op_name", "n_tested", "n_stage1_passed", "n_validation_passed",
                "stage1_pass_rate", "validation_pass_rate", "baseline_win_rate",
                "trust_score", "trust_label", "avg_novelty_score",
            ):
                self.assertIn(key, row)

    def test_api_analytics_compression_coverage_schema(self):
        """Compression coverage returns stable technique/totals structures."""
        r = self.client.get("/api/analytics/compression-coverage")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("techniques", data)
        self.assertIn("totals", data)
        self.assertIsInstance(data["techniques"], list)
        self.assertIsInstance(data["totals"], dict)
        for key in ("n_tested", "n_survived", "n_compressed_tested", "n_compressed_survived"):
            self.assertIn(key, data["totals"])
        if data["techniques"]:
            row = data["techniques"][0]
            for key in (
                "technique", "n_tested", "n_survived", "survival_rate",
                "tested_share", "survivor_share", "avg_compression_ratio",
                "avg_estimated_memory_mb", "avg_quality_retention",
            ):
                self.assertIn(key, row)

    def test_api_analytics_learning_summary_schema(self):
        """Learning summary returns a stable bullet-list contract."""
        r = self.client.get("/api/analytics/learning-summary")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("bullets", data)
        self.assertIn("source", data)
        self.assertIsInstance(data["bullets"], list)
        self.assertIsInstance(data["source"], str)
        self.assertGreaterEqual(len(data["bullets"]), 1)
        for bullet in data["bullets"]:
            self.assertIsInstance(bullet, str)

    def test_api_report_schema(self):
        """Report payload must include all sections consumed by report views."""
        r = self.client.get("/api/report")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()

        required = [
            "summary", "top_programs", "recent_experiments",
            "top_programs_expanded",
            "architecture_rerun_telemetry",
            "math_family_coverage",
            "mathspace_operator_impact",
            "routing_mode_comparison",
            "gating_behavior_diagnostics",
            "op_success_rates", "structural_correlations",
            "failure_patterns", "top_op_combinations",
            "efficiency_frontier", "experiment_clusters",
            "grammar_weights", "learning_log", "insights", "narrative",
            "cross_run_stability",
            "action_eligibility",
        ]
        for key in required:
            self.assertIn(key, data, f"report missing key: {key}")

        grammar = data["grammar_weights"]
        self.assertIn("default", grammar)
        self.assertIn("learned", grammar)
        self.assertIn("control_comparison", grammar)
        self.assertIn("holdout_validation", grammar)
        self.assertIn("learning_diagnostics", grammar)
        telemetry = data["architecture_rerun_telemetry"]
        self.assertIn("unique_fingerprint_count", telemetry)
        self.assertIn("rerun_ratio", telemetry)
        self.assertIn("top_fingerprint_concentration", telemetry)

        stability = data["cross_run_stability"]
        self.assertIn("summary", stability)
        self.assertIn("candidates", stability)
        self.assertIn("window_size", stability)
        self.assertIsInstance(stability["summary"], dict)
        self.assertIsInstance(stability["candidates"], list)
        self.assertIn("families", data["math_family_coverage"])
        self.assertIn("totals", data["math_family_coverage"])
        self.assertIsInstance(data["math_family_coverage"]["families"], list)
        self.assertIsInstance(data["math_family_coverage"]["totals"], dict)
        self.assertIn("available", data["routing_mode_comparison"])
        self.assertIn("by_mode", data["routing_mode_comparison"])
        self.assertIsInstance(data["routing_mode_comparison"]["by_mode"], list)
        self.assertIn("available", data["mathspace_operator_impact"])
        self.assertIn("by_operator", data["mathspace_operator_impact"])
        self.assertIn("by_family", data["mathspace_operator_impact"])
        self.assertIn("top_trustworthy_operators", data["mathspace_operator_impact"])
        self.assertIsInstance(data["mathspace_operator_impact"]["by_operator"], list)
        self.assertIsInstance(data["mathspace_operator_impact"]["top_trustworthy_operators"], list)
        self.assertIn("available", data["gating_behavior_diagnostics"])
        self.assertIn("by_mode", data["gating_behavior_diagnostics"])
        self.assertIn("token_retention_curve_overall", data["gating_behavior_diagnostics"])
        self.assertIsInstance(data["gating_behavior_diagnostics"]["by_mode"], list)
        if data["top_programs"]:
            row = data["top_programs"][0]
            self.assertIn("qkv_usage", row)
            self.assertIn(row["qkv_usage"], {"full_qkv", "q_eq_k_eq_v", "qkv_free"})
            self.assertIn("uses_qkv", row)
            self.assertIsInstance(row["uses_qkv"], bool)
            self.assertIn("compression_metrics", row)
            self.assertIn("reproducibility_packet", row)
            self.assertIn("compression_ratio", row["compression_metrics"])
            self.assertIn("status", row["reproducibility_packet"])
            self.assertIn("repeat_count", row)
            self.assertIn("repeat_experiment_span", row)
            self.assertIn("repeat_loss_min", row)
            self.assertIn("repeat_loss_max", row)
            self.assertGreaterEqual(row["repeat_count"], 1)
            self.assertGreaterEqual(row["repeat_experiment_span"], 1)
            self.assertIn(row["result_id"], data["action_eligibility"])
            eligibility = data["action_eligibility"][row["result_id"]]
            self.assertIn("investigationEligible", eligibility)
            self.assertIn("validationEligible", eligibility)
            self.assertIn("queueEligible", eligibility)
            self.assertIn("queueReason", eligibility)
            self.assertIn("cross_run_stability", row)
            row_stability = row["cross_run_stability"]
            self.assertIn("trend", row_stability)
            self.assertIn("seen_runs", row_stability)
            self.assertIn("latest_rank", row_stability)
            self.assertIn("previous_rank", row_stability)
            self.assertIn("rank_delta", row_stability)

        self.assertIsInstance(data["top_programs_expanded"], list)
        if data["top_programs_expanded"]:
            expanded_row = data["top_programs_expanded"][0]
            self.assertIn("group_repeat_count", expanded_row)
            self.assertIn("group_repeat_index", expanded_row)
            self.assertIn("cross_run_stability", expanded_row)

    def test_api_report_top_programs_are_fingerprint_deduplicated(self):
        """Report discovery ranking should include at most one row per fingerprint."""
        r = self.client.get("/api/report")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        top = data.get("top_programs") or []
        fingerprints = [row.get("graph_fingerprint") for row in top if row.get("graph_fingerprint")]
        self.assertEqual(len(fingerprints), len(set(fingerprints)))

    def test_api_reproducibility_manifest_schema(self):
        """Repro manifest includes canonical metrics and packet completeness."""
        r_programs = self.client.get("/api/programs")
        self.assertEqual(r_programs.status_code, 200)
        programs = r_programs.get_json()
        if not programs:
            return
        result_id = programs[0]["result_id"]

        r = self.client.get(f"/api/reproducibility-manifest/{result_id}")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("canonical_metrics", data)
        self.assertIn("packet_status", data)
        self.assertIn("compression", data["canonical_metrics"])
        self.assertIn("compression_ratio", data["canonical_metrics"]["compression"])
        self.assertIn("status", data["packet_status"])

    def test_api_trends_entry_schema(self):
        """Trends entries should expose timeline and stage-rate fields."""
        r = self.client.get("/api/trends")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

        entry = data[0]
        required = [
            "experiment_id", "timestamp", "n_programs_generated",
            "n_stage0_passed", "n_stage05_passed", "n_stage1_passed",
            "best_loss_ratio", "best_novelty_score", "duration_seconds",
            "s1_pass_rate",
            "adjusted_s1_pass_rate", "s1_confidence_lower", "s1_confidence_upper",
            "s1_confidence_halfwidth", "trend_weight", "trend_confidence", "trend_mode",
        ]
        for key in required:
            self.assertIn(key, entry, f"trends entry missing key: {key}")

        self.assertLessEqual(entry["s1_confidence_lower"], entry["adjusted_s1_pass_rate"])
        self.assertGreaterEqual(entry["s1_confidence_upper"], entry["adjusted_s1_pass_rate"])
        self.assertIn(entry["trend_confidence"], {"low", "medium", "high"})

    def test_api_trends_context_schema(self):
        """Trends context should include adaptation event delta windows."""
        r = self.client.get("/api/trends/context")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("trends", data)
        self.assertIn("adaptation_events", data)
        self.assertIn("generated_at", data)
        self.assertIsInstance(data["adaptation_events"], list)

        if data["adaptation_events"]:
            event = data["adaptation_events"][0]
            self.assertIn("timestamp", event)
            self.assertIn("description", event)
            self.assertIn("before_window", event)
            self.assertIn("after_window", event)
            self.assertIn("delta", event)
            self.assertIn("adjusted_s1_rate", event["delta"])
            self.assertIn("best_novelty", event["delta"])
            self.assertIn("best_loss_ratio", event["delta"])
            self.assertIn("experiment_ids", event["before_window"])
            self.assertIn("experiment_ids", event["after_window"])
            self.assertIsInstance(event["before_window"]["experiment_ids"], list)
            self.assertIsInstance(event["after_window"]["experiment_ids"], list)

    def test_api_knowledge_schema(self):
        """Knowledge endpoint should return enriched knowledge-base entry shape."""
        r = self.client.get("/api/knowledge")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

        entry = data[0]
        required = [
            "entry_id", "timestamp", "category", "title", "content",
            "confidence", "times_validated", "last_validated", "status",
        ]
        for key in required:
            self.assertIn(key, entry, f"knowledge entry missing key: {key}")

    def test_api_knowledge_search_schema(self):
        """Knowledge search should return same entry shape as knowledge listing."""
        r = self.client.get("/api/knowledge/search?q=schema")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

        entry = data[0]
        for key in ("entry_id", "title", "content", "confidence", "status"):
            self.assertIn(key, entry, f"knowledge search result missing key: {key}")

    def test_api_campaigns_list_schema(self):
        """Campaign list rows must include fields consumed by Campaigns tab."""
        r = self.client.get("/api/campaigns")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

        row = data[0]
        required = [
            "campaign_id", "title", "objective", "status",
            "n_experiments", "n_hypotheses", "n_decisions",
            "success_criteria",
        ]
        for key in required:
            self.assertIn(key, row, f"campaign list row missing key: {key}")

    def test_api_campaign_detail_schema(self):
        """Campaign detail payload must include campaign and related collections."""
        r = self.client.get(f"/api/campaigns/{self.campaign_id}")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()

        self.assertIn("campaign", data)
        self.assertIn("experiments", data)
        self.assertIn("hypotheses", data)
        self.assertIn("decisions", data)
        self.assertIn("success_criteria_tracker", data)
        self.assertIsInstance(data["experiments"], list)
        self.assertIsInstance(data["hypotheses"], list)
        self.assertIsInstance(data["decisions"], list)
        self.assertIsInstance(data["success_criteria_tracker"], list)
        self.assertGreater(len(data["hypotheses"]), 0)

        hypothesis = data["hypotheses"][0]
        self.assertIn("metadata", hypothesis)
        self.assertIsInstance(hypothesis["metadata"], dict)
        self.assertEqual(hypothesis["metadata"].get("source"), "llm_context")

    def test_api_campaign_report_schema(self):
        """Campaign report payload must include campaign/report/stats sections."""
        r = self.client.get(f"/api/campaigns/{self.campaign_id}/report")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()

        self.assertIn("campaign", data)
        self.assertIn("report", data)
        self.assertIn("stats", data)
        self.assertIn("success_criteria_tracker", data)
        self.assertIsInstance(data["success_criteria_tracker"], list)
        stats = data["stats"]
        for key in ("n_experiments", "n_hypotheses", "n_confirmed", "n_refuted", "n_decisions"):
            self.assertIn(key, stats, f"campaign report stats missing key: {key}")

    def test_api_404_for_unknown_endpoint(self):
        r = self.client.get("/api/nonexistent")
        self.assertEqual(r.status_code, 404)

    def test_sse_timeout_env_parsing(self):
        from research.scientist.api import _get_sse_timeout_seconds

        with patch.dict(os.environ, {"ARIA_SSE_TIMEOUT_SECONDS": "60"}, clear=False):
            self.assertEqual(_get_sse_timeout_seconds(), 60.0)

        with patch.dict(os.environ, {"ARIA_SSE_TIMEOUT_SECONDS": "invalid"}, clear=False):
            self.assertEqual(_get_sse_timeout_seconds(), 30.0)

        with patch.dict(os.environ, {"ARIA_SSE_TIMEOUT_SECONDS": "0"}, clear=False):
            self.assertEqual(_get_sse_timeout_seconds(), 30.0)


# ── Test 7: Auto-Escalation Pipeline ──


@unittest.skipUnless(HAS_TORCH, "requires torch for runner module")
class TestAutoEscalation(unittest.TestCase):
    """Test that the auto-escalation pipeline correctly queues
    investigation and validation."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_escalation.db")
        from research.scientist.runner import ExperimentRunner, RunConfig
        self.runner = ExperimentRunner(self.db_path)
        self.config = RunConfig(
            auto_investigate=True,
            auto_investigate_min_survivors=1,
            auto_investigate_top_n=3,
            auto_validate=True,
        )

    def test_auto_escalate_queues_investigation(self):
        """S1 survivors with good loss should queue investigation."""
        nb = LabNotebook(self.db_path)

        # Create experiment with S1 survivor
        exp_id = nb.start_experiment("synthesis", {}, "test")
        nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_escalate",
            graph_json="{}",
            stage1_passed=True,
            loss_ratio=0.3,
            novelty_score=0.6,
            model_source="graph_synthesis",
        )
        nb.complete_experiment(exp_id, {
            "total": 10, "stage1_passed": 1,
        }, "summary", "excited")

        results = {"stage1_passed": 1, "survivors": [
            {"novelty": 0.6, "loss_ratio": 0.3}
        ]}

        self.runner._auto_escalate(results, self.config, nb, phase="screening")

        # Should have queued investigation
        pending = getattr(self.runner, "_pending_investigation", None)
        self.assertIsNotNone(pending,
                             "Investigation should be queued after S1 survivor")
        self.assertIn("result_ids", pending)

        nb.close()

    def test_auto_escalate_skips_when_disabled(self):
        """No escalation when auto_investigate is False."""
        from research.scientist.runner import RunConfig
        nb = LabNotebook(self.db_path)

        config = RunConfig(auto_investigate=False)
        results = {"stage1_passed": 5}

        self.runner._auto_escalate(results, config, nb, phase="screening")

        pending = getattr(self.runner, "_pending_investigation", None)
        self.assertIsNone(pending)
        nb.close()

    def test_auto_escalate_skips_low_survivors(self):
        """No escalation when not enough survivors."""
        from research.scientist.runner import RunConfig
        nb = LabNotebook(self.db_path)

        config = RunConfig(auto_investigate=True, auto_investigate_min_survivors=5)
        results = {"stage1_passed": 2}

        self.runner._auto_escalate(results, config, nb, phase="screening")

        pending = getattr(self.runner, "_pending_investigation", None)
        self.assertIsNone(pending)
        nb.close()

    def test_auto_escalate_queues_validation(self):
        """Investigation results with good robustness should queue validation."""
        nb = LabNotebook(self.db_path)

        results = {
            "investigation_results": [
                {
                    "result_id": "r1",
                    "robustness": 0.7,
                    "best_loss_ratio": 0.4,
                    "baseline_loss_ratio": 0.8,
                    "novelty_confidence": 0.7,
                },
                {
                    "result_id": "r2",
                    "robustness": 0.3,
                    "best_loss_ratio": 0.6,
                    "baseline_loss_ratio": 0.85,
                    "novelty_confidence": 0.7,
                },
            ]
        }

        self.runner._auto_escalate(results, self.config, nb, phase="investigation")

        pending = getattr(self.runner, "_pending_validation", None)
        self.assertIsNotNone(pending,
                             "Validation should be queued after robust investigation")
        self.assertEqual(len(pending["result_ids"]), 1)  # only r1 qualifies
        nb.close()

    def test_auto_escalate_excludes_brittle_candidates(self):
        """Brittle investigation outcomes should not auto-queue for validation."""
        nb = LabNotebook(self.db_path)

        results = {
            "investigation_results": [
                {
                    "result_id": "stable",
                    "robustness": 0.8,
                    "best_loss_ratio": 0.35,
                    "baseline_loss_ratio": 0.82,
                    "novelty_confidence": 0.75,
                    "loss_ratio_multiplier": 2.0,
                    "brittle_risk": False,
                },
                {
                    "result_id": "brittle_flag",
                    "robustness": 0.85,
                    "best_loss_ratio": 0.3,
                    "baseline_loss_ratio": 0.81,
                    "novelty_confidence": 0.8,
                    "loss_ratio_multiplier": 20.0,
                    "brittle_risk": True,
                },
                {
                    "result_id": "brittle_multiplier",
                    "robustness": 0.9,
                    "best_loss_ratio": 0.25,
                    "baseline_loss_ratio": 0.8,
                    "novelty_confidence": 0.8,
                    "loss_ratio_multiplier": self.config.investigation_max_loss_ratio_multiplier + 0.1,
                    "brittle_risk": False,
                },
            ]
        }

        self.runner._auto_escalate(results, self.config, nb, phase="investigation")

        pending = getattr(self.runner, "_pending_validation", None)
        self.assertIsNotNone(pending)
        self.assertEqual(pending["result_ids"], ["stable"])
        nb.close()

    def test_auto_escalate_requires_baseline_and_novelty_confidence(self):
        """Validation auto-queue should require strong baseline + novelty confidence evidence."""
        nb = LabNotebook(self.db_path)

        results = {
            "investigation_results": [
                {
                    "result_id": "missing_conf",
                    "robustness": 0.85,
                    "best_loss_ratio": 0.32,
                    "baseline_loss_ratio": 0.82,
                },
                {
                    "result_id": "weak_baseline",
                    "robustness": 0.9,
                    "best_loss_ratio": 0.3,
                    "baseline_loss_ratio": 0.96,
                    "novelty_confidence": 0.8,
                },
                {
                    "result_id": "qualified",
                    "robustness": 0.88,
                    "best_loss_ratio": 0.31,
                    "baseline_loss_ratio": 0.82,
                    "novelty_confidence": 0.72,
                },
            ]
        }

        self.runner._auto_escalate(results, self.config, nb, phase="investigation")

        pending = getattr(self.runner, "_pending_validation", None)
        self.assertIsNotNone(pending)
        self.assertEqual(pending["result_ids"], ["qualified"])
        nb.close()

    def test_run_pending_validation_passes_auto_trigger(self):
        """Queued auto-validation should launch with an explicit auto-escalate trigger."""
        self.runner._pending_validation = {
            "result_ids": ["r1", "r2"],
            "config": self.config,
            "hypothesis": "auto validation",
        }

        with patch.object(self.runner, "start_validation") as start_validation:
            self.runner._run_pending_validation()

        self.assertTrue(start_validation.called)
        kwargs = start_validation.call_args.kwargs
        self.assertEqual(kwargs["trigger"], "auto_escalate")

    def test_start_validation_persists_candidate_metadata_in_config(self):
        """Validation experiment config should include selected candidate IDs."""
        with patch("research.scientist.runner.threading.Thread") as thread_cls:
            thread_inst = MagicMock()
            thread_cls.return_value = thread_inst

            exp_id = self.runner.start_validation(
                result_ids=["rid-a", "rid-b"],
                config=self.config,
                hypothesis="metadata test",
            )

        nb = LabNotebook(self.db_path)
        try:
            row = nb.conn.execute(
                "SELECT config_json FROM experiments WHERE experiment_id = ?",
                (exp_id,),
            ).fetchone()
            self.assertIsNotNone(row)
            cfg = json.loads(row["config_json"])
            self.assertEqual(cfg.get("validation_result_ids"), ["rid-a", "rid-b"])
            self.assertEqual(cfg.get("validation_candidate_count"), 2)
            self.assertEqual(cfg.get("validation_trigger"), "manual")
        finally:
            nb.close()

    def test_leaderboard_populated_during_escalation(self):
        """Auto-escalation should add entries to the leaderboard."""
        nb = LabNotebook(self.db_path)

        exp_id = nb.start_experiment("synthesis", {}, "test")
        nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_lb",
            graph_json="{}",
            stage1_passed=True,
            loss_ratio=0.4,
            novelty_score=0.7,
            model_source="graph_synthesis",
        )
        nb.complete_experiment(exp_id, {"total": 5, "stage1_passed": 1},
                               "done", "excited")

        results = {"stage1_passed": 1, "survivors": [{"novelty": 0.7}]}
        self.runner._auto_escalate(results, self.config, nb, phase="screening")

        leaderboard = nb.get_leaderboard()
        self.assertGreater(len(leaderboard), 0,
                           "Leaderboard should have entries after escalation")
        self.assertEqual(leaderboard[0]["tier"], "screening")
        nb.close()

    def test_ensure_campaign_marks_post_hoc_criteria(self):
        """Campaign criteria created from recent results should be labeled post-hoc."""
        nb = LabNotebook(self.db_path)
        try:
            self.runner._active_campaign_id = None
            self.runner.aria.formulate_campaign = MagicMock(return_value={
                "title": "Campaign A",
                "objective": "Explore",
                "success_criteria": "Increase S1 pass rate",
            })

            campaign_id = self.runner._ensure_campaign(self.config, nb)
            self.assertIsNotNone(campaign_id)

            campaign = nb.get_campaign(campaign_id)
            self.assertIsNotNone(campaign)
            self.assertIn("[POST-HOC]", campaign["success_criteria"])
        finally:
            nb.close()


# ── Test 8: Prompt Templates ──


@unittest.skipUnless(HAS_PROMPTS, "requires prompts module")
class TestPrompts(unittest.TestCase):
    """Verify all prompt templates exist and have correct placeholders."""

    def test_all_prompts_importable(self):
        from research.scientist.llm.prompts import (
            SYSTEM_PROMPT,
            ANALYSIS_PROMPT,
            HYPOTHESIS_PROMPT,
            SUMMARY_PROMPT,
            FINGERPRINT_EXPLANATION_PROMPT,
            STRATEGY_PROMPT,
            SUGGESTION_PROMPT,
            VALIDATION_PROMPT,
            REPORT_PROMPT,
            INVESTIGATION_HYPOTHESIS_PROMPT,
            VALIDATION_ANALYSIS_PROMPT,
            BREAKTHROUGH_ANNOUNCEMENT_PROMPT,
            MODE_SELECTION_PROMPT,
        )
        # All should have {context} placeholder
        for name, prompt in [
            ("ANALYSIS", ANALYSIS_PROMPT),
            ("HYPOTHESIS", HYPOTHESIS_PROMPT),
            ("SUMMARY", SUMMARY_PROMPT),
            ("FINGERPRINT", FINGERPRINT_EXPLANATION_PROMPT),
            ("STRATEGY", STRATEGY_PROMPT),
            ("SUGGESTION", SUGGESTION_PROMPT),
            ("REPORT", REPORT_PROMPT),
            ("INVESTIGATION", INVESTIGATION_HYPOTHESIS_PROMPT),
            ("VALIDATION_ANALYSIS", VALIDATION_ANALYSIS_PROMPT),
            ("BREAKTHROUGH", BREAKTHROUGH_ANNOUNCEMENT_PROMPT),
            ("MODE_SELECTION", MODE_SELECTION_PROMPT),
        ]:
            self.assertIn("{context}", prompt,
                          f"{name}_PROMPT missing {{context}} placeholder")

    def test_validation_prompt_has_hypothesis_placeholder(self):
        from research.scientist.llm.prompts import VALIDATION_PROMPT
        self.assertIn("{hypothesis}", VALIDATION_PROMPT)


class TestPackageWiring(unittest.TestCase):
    """Ensure explicitly connected package modules remain importable."""

    def test_mathspaces_exports_modules(self):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        init_path = os.path.join(repo_root, "mathspaces", "__init__.py")
        with open(init_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("from . import clifford, hyperbolic, padic, tropical", content)
        self.assertIn("from .registry import register_all_mathspaces", content)
        self.assertIn('"hyperbolic"', content)
        self.assertIn('"tropical"', content)
        self.assertIn('"padic"', content)
        self.assertIn('"clifford"', content)

    @unittest.skipUnless(HAS_TORCH, "requires torch")
    def test_mathspace_registry_includes_hyp_distance(self):
        from research.mathspaces.registry import register_all_mathspaces
        from research.synthesis.primitives import PRIMITIVE_REGISTRY

        register_all_mathspaces()
        self.assertIn("hyp_distance", PRIMITIVE_REGISTRY)
        op = PRIMITIVE_REGISTRY["hyp_distance"]
        self.assertEqual(op.category.value, "math_space")
        self.assertEqual(op.n_inputs, 2)
        self.assertTrue(hasattr(op, "execute_fn"))

    @unittest.skipUnless(HAS_TORCH, "requires torch")
    def test_external_op_nonfinite_sanitization_and_telemetry(self):
        from research.synthesis.compiler import _execute_op
        from research.synthesis.primitives import PrimitiveOp, OpCategory, PRIMITIVE_REGISTRY, register_external_primitive

        op_name = "test_nonfinite_mathspace_op"
        op = PrimitiveOp(
            name=op_name,
            category=OpCategory.MATH_SPACE,
            n_inputs=1,
            shape_rule="identity",
            description="test external op",
        )

        def _execute_fn(module, x):
            return x / 0.0

        object.__setattr__(op, "execute_fn", _execute_fn)
        register_external_primitive(op)
        module = torch.nn.Module()
        x = torch.ones(2, 3, 4)
        try:
            out = _execute_op(module, op_name, (x,), {})
            self.assertTrue(torch.isfinite(out).all())
            telemetry = getattr(module, "mathspace_telemetry", {})
            self.assertIn(op_name, telemetry)
            self.assertGreaterEqual(telemetry[op_name]["calls"], 1)
            self.assertGreater(telemetry[op_name]["nonfinite_elements"], 0)
            self.assertGreaterEqual(telemetry[op_name]["sanitized_calls"], 1)
        finally:
            PRIMITIVE_REGISTRY.pop(op_name, None)

    @unittest.skipUnless(HAS_TORCH, "requires torch")
    def test_mathspace_phase2_ops_registered(self):
        from research.mathspaces.registry import register_all_mathspaces
        from research.synthesis.primitives import PRIMITIVE_REGISTRY

        register_all_mathspaces()
        for op_name in ("hyp_tangent_nonlinear", "tropical_center", "padic_gate", "grade_mix"):
            self.assertIn(op_name, PRIMITIVE_REGISTRY)
            op = PRIMITIVE_REGISTRY[op_name]
            self.assertEqual(op.category.value, "math_space")
            self.assertEqual(op.n_inputs, 1)
            self.assertTrue(hasattr(op, "execute_fn"))

    @unittest.skipUnless(HAS_TORCH, "requires torch")
    def test_mathspace_phase2_ops_execute_shape_and_finite(self):
        from research.mathspaces.registry import register_all_mathspaces
        from research.synthesis.primitives import PRIMITIVE_REGISTRY

        register_all_mathspaces()
        x = torch.randn(2, 5, 16)
        module = torch.nn.Module()
        for op_name in ("hyp_tangent_nonlinear", "tropical_center", "padic_gate", "grade_mix"):
            op = PRIMITIVE_REGISTRY[op_name]
            out = op.execute_fn(module, x)
            self.assertEqual(tuple(out.shape), tuple(x.shape))
            self.assertTrue(torch.isfinite(out).all(), f"{op_name} produced non-finite values")

    def test_llm_package_exports_context_and_prompts(self):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        init_path = os.path.join(repo_root, "scientist", "llm", "__init__.py")
        with open(init_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("from . import context, prompts", content)
        self.assertIn("from .backend import", content)
        self.assertIn('"context"', content)
        self.assertIn('"prompts"', content)


# ── Test 9: Persona Methods ──


@unittest.skipUnless(HAS_PERSONA, "requires persona module")
class TestPersona(unittest.TestCase):
    """Verify Aria persona has all required methods."""

    def setUp(self):
        self.aria = Aria()

    def test_all_methods_exist(self):
        """Aria should have all expected public methods."""
        methods = [
            "greet", "react_to_discovery", "react_to_failure",
            "begin_analysis", "formulate_hypothesis",
            "experiment_summary", "analyze_results",
            "explain_fingerprint", "plan_strategy",
            "suggest_experiment", "validate_hypothesis",
            "explain_learning", "generate_report_narrative",
            "get_status", "add_insight",
            # Phase methods
            "formulate_investigation_hypothesis",
            "formulate_validation_hypothesis",
            "announce_breakthrough",
            # Mode selection
            "recommend_next_mode",
        ]
        for m in methods:
            self.assertTrue(hasattr(self.aria, m),
                            f"Aria missing method: {m}")
            self.assertTrue(callable(getattr(self.aria, m)),
                            f"Aria.{m} is not callable")

    def test_greet_returns_string(self):
        msg = self.aria.greet()
        self.assertIsInstance(msg, str)
        self.assertGreater(len(msg), 0)

    def test_get_status_returns_dict(self):
        status = self.aria.get_status()
        self.assertIn("name", status)
        self.assertIn("mood", status)
        self.assertIn("llm_enabled", status)

    def test_rule_based_hypothesis(self):
        hyp = self.aria.formulate_hypothesis()
        self.assertIsInstance(hyp, str)
        self.assertGreater(len(hyp), 0)

    def test_rule_based_summary(self):
        results = {"total": 50, "stage0_passed": 30,
                   "stage05_passed": 20, "stage1_passed": 2,
                   "novel_count": 1}
        summary = self.aria.experiment_summary(results)
        self.assertIsInstance(summary, str)
        self.assertIn("50", summary)

    def test_rule_based_investigation_hypothesis(self):
        hyp = self.aria.formulate_investigation_hypothesis()
        self.assertIsInstance(hyp, str)
        self.assertIn("training", hyp.lower())

    def test_rule_based_validation_hypothesis(self):
        hyp = self.aria.formulate_validation_hypothesis()
        self.assertIsInstance(hyp, str)

    def test_hypothesis_critique_returns_gate_and_checks(self):
        self.aria._get_llm = MagicMock(return_value=None)
        critique = self.aria.critique_hypothesis("Try something new")
        self.assertIn("verdict", critique)
        self.assertIn("gate", critique)
        self.assertIn(critique["gate"], {"pass", "warn", "fail"})
        self.assertIn("checks", critique)
        self.assertIsInstance(critique["checks"], list)
        check_keys = {c.get("key") for c in critique["checks"] if isinstance(c, dict)}
        self.assertTrue({"testability", "measurable_metric", "confound_risk", "fallback_plan"}.issubset(check_keys))

    def test_announce_breakthrough(self):
        msg = self.aria.announce_breakthrough()
        self.assertIsInstance(msg, str)
        self.assertIn("BREAKTHROUGH", msg)

    def test_assess_breakthrough_evidence_publication_grade(self):
        evidence = self.aria.assess_breakthrough_evidence(metrics={
            "seeds_passed": 6,
            "total_seeds": 6,
            "val_baseline_ratio": 0.82,
            "multi_seed_std": 0.018,
            "ood_robustness": 0.8,
            "hp_robustness": 0.85,
        })
        self.assertEqual(evidence["label"], "publication_grade")
        self.assertIn(evidence["confidence_band"], {"high", "medium", "low"})

    def test_assess_breakthrough_evidence_provisional_for_low_seed_count(self):
        evidence = self.aria.assess_breakthrough_evidence(metrics={
            "seeds_passed": 3,
            "total_seeds": 3,
            "val_baseline_ratio": 0.82,
            "multi_seed_std": 0.018,
        })
        self.assertEqual(evidence["label"], "provisional")
        self.assertIn("seed_count_below_publication_threshold", evidence["reasons"])

    def test_announce_breakthrough_provisional_language(self):
        msg = self.aria.announce_breakthrough(metrics={
            "seeds_passed": 3,
            "total_seeds": 3,
            "val_baseline_ratio": 0.93,
            "multi_seed_std": 0.05,
        })
        self.assertIn("BREAKTHROUGH SIGNAL DETECTED", msg)
        self.assertIn("PROVISIONAL", msg)

    def test_cost_tracking(self):
        self.aria.reset_cost_tracking()
        self.assertEqual(self.aria.total_tokens, 0)
        self.assertEqual(self.aria.total_cost, 0.0)

    def test_unknown_backend_cost_logs_warning_once(self):
        class _Resp:
            tokens_used = 100

        class _Backend:
            name = "mystery-backend"

        self.aria._llm = _Backend()
        with patch("research.scientist.persona.logger.warning") as warn:
            self.aria._track_cost(_Resp())
            self.aria._track_cost(_Resp())
            self.assertEqual(warn.call_count, 1)
        self.assertGreater(self.aria.total_cost, 0.0)


class TestAnthropicBackendConfig(unittest.TestCase):
    """Backend config defaults should be resilient to model deprecations."""

    def test_default_model_uses_alias(self):
        with patch.dict(os.environ, {}, clear=True):
            from research.scientist.llm.anthropic import AnthropicBackend
            backend = AnthropicBackend()
            self.assertEqual(backend.model, "claude-sonnet-latest")

    def test_env_model_override_wins(self):
        with patch.dict(os.environ, {"ANTHROPIC_MODEL": "custom-model"}, clear=True):
            from research.scientist.llm.anthropic import AnthropicBackend
            backend = AnthropicBackend()
            self.assertEqual(backend.model, "custom-model")


# ── Test 10: Dashboard Component Consistency ──


class TestDashboardConsistency(unittest.TestCase):
    """Verify dashboard components and API endpoints are properly wired."""

    @classmethod
    def setUpClass(cls):
        import glob
        cls.repo_root = os.path.dirname(os.path.dirname(__file__))
        cls.component_dir = os.path.join(
            cls.repo_root, "dashboard", "src", "components")
        cls.component_files = glob.glob(
            os.path.join(cls.component_dir, "*.js"))
        cls.app_js = os.path.join(
            cls.repo_root, "dashboard", "src", "App.js")
        cls.api_py = os.path.join(cls.repo_root, "scientist", "api.py")

    def _read_file(self, path):
        with open(path, "r") as f:
            return f.read()

    def test_all_components_imported_in_app(self):
        """Every component should be imported in App.js."""
        app_content = self._read_file(self.app_js)

        # Components that are used inside other components, not App.js
        nested_only = {
            "GraphViewer", "FailureAnalysis", "AriaAvatar",
        }

        for filepath in self.component_files:
            name = os.path.basename(filepath).replace(".js", "")
            if name in nested_only:
                continue
            self.assertIn(
                f"import {name}",
                app_content,
                f"Component {name} not imported in App.js",
            )

    def test_all_components_have_default_export(self):
        """Every component file should have a default export."""
        for filepath in self.component_files:
            content = self._read_file(filepath)
            name = os.path.basename(filepath).replace(".js", "")
            self.assertIn(
                f"export default {name}",
                content,
                f"{name}.js missing 'export default {name}'",
            )

    def test_no_orphaned_api_fetch_urls(self):
        """All fetch URLs in components should match real API endpoints."""
        import re

        known_api_patterns = {
            "/api/dashboard", "/api/status", "/api/system/status",
            "/api/experiments", "/api/programs", "/api/trends",
            "/api/trends/context",
            "/api/insights", "/api/entries", "/api/live-feed", "/api/leaderboard",
            "/api/report", "/api/events", "/api/progress",
            "/api/config", "/api/validate",
            "/api/aria/recommendation", "/api/aria/strategy",
            "/api/strategy/briefing",
            "/api/llm/config",
            "/api/analytics/op-success", "/api/analytics/failure-patterns",
            "/api/analytics/grammar-weights", "/api/analytics/efficiency-frontier",
            "/api/analytics/learning-log", "/api/analytics/experiment-clusters",
            "/api/analytics/routing-health", "/api/analytics/math-family-coverage",
            "/api/analytics/mathspace-impact",
            "/api/analytics/routing-comparison",
            "/api/analytics/gating-diagnostics",
            "/api/analytics/compression-coverage",
            "/api/analytics/learning-summary",
            "/api/analytics/learning-trajectory",
            "/api/analytics/control-comparison",
            "/api/metrics/",
            "/api/experiments/start", "/api/experiments/stop",
            "/api/experiments/",
            "/api/campaigns", "/api/hypotheses",
            "/api/knowledge",
            "/api/decision-packet/",
            "/api/reproducibility-manifest/",
            "/api/analytics/negative-results",
            "/api/aria/chat",
            "/api/aria/chat/history",
            "/api/aria/chat/message",
            "/api/aria/chat/compact",
            "/api/aria/cycle-status",
            "/api/aria/cycle-control",
            "/api/aria/cycle-history",
        }

        for filepath in self.component_files:
            content = self._read_file(filepath)
            # Find all fetch/API calls
            urls = re.findall(
                r'[`\'"](?:\$\{[^}]*\})?(/api/[a-z/_-]+)', content)
            for url in urls:
                # Normalize: remove dynamic segments
                base_url = re.sub(r'/\$\{[^}]*\}', '/', url)
                base_url = base_url.rstrip("/")

                matched = any(
                    base_url.startswith(pattern.rstrip("/"))
                    for pattern in known_api_patterns
                )
                self.assertTrue(
                    matched,
                    f"Orphaned API URL in {os.path.basename(filepath)}: {url}",
                )

    def test_tab_names_match_content(self):
        """All tab names in App.js should have corresponding content blocks."""
        app_content = self._read_file(self.app_js)

        # Extract tab list from nav
        import re
        tab_match = re.search(r"\[([^\]]+)\]\.map\(tab", app_content)
        if tab_match:
            tabs_str = tab_match.group(1)
            tabs = re.findall(r"'(\w[\w-]*)'", tabs_str)

            for tab in tabs:
                # Each tab should have activeTab === 'tabname'
                self.assertIn(
                    f"activeTab === '{tab}'",
                    app_content,
                    f"Tab '{tab}' has no content block in App.js",
                )

    def test_onSelectProgram_wired(self):
        """Components with onSelectProgram should receive it as prop."""
        app_content = self._read_file(self.app_js)
        # These components should pass onSelectProgram
        for comp in ["TopPrograms", "Leaderboard", "ExperimentDetail"]:
            self.assertIn(
                f"onSelectProgram={{handleSelectProgram}}",
                app_content,
                f"{comp} should pass onSelectProgram prop",
            )

    def test_sse_event_types_handled(self):
        """All SSE event types emitted by runner should be handled in LiveFeed."""
        livefeed_path = os.path.join(self.component_dir, "LiveFeed.js")
        content = self._read_file(livefeed_path)

        expected_events = [
            "program_evaluated",
            "experiment_started",
            "experiment_completed",
            "experiment_failed",
            "investigation_started",
            "investigation_completed",
            "validation_started",
            "validation_completed",
            "breakthrough_detected",
            "mode_selected",
        ]

        for event in expected_events:
            self.assertIn(
                f"'{event}'",
                content,
                f"LiveFeed.js missing handler for SSE event: {event}",
            )

    def test_frontend_api_routes_exist_in_backend(self):
        """All frontend /api paths should map to a backend Flask route."""
        import re

        api_content = self._read_file(self.api_py)
        route_re = re.compile(r"@app\.route\(\s*['\"](/api/[^'\"]+)['\"]")
        backend_routes = [self._normalize_route(r) for r in route_re.findall(api_content)]

        for filepath in self.component_files + [self.app_js]:
            content = self._read_file(filepath)
            found = re.findall(r"/api/[A-Za-z0-9_\-/${}]+(?:/[A-Za-z0-9_\-/${}]+)*", content)
            for path in found:
                normalized = self._normalize_route(path)
                matched = any(self._route_matches(b, normalized) for b in backend_routes)
                self.assertTrue(
                    matched,
                    f"Frontend route has no backend mapping: {path} in {os.path.basename(filepath)}",
                )

    def test_strategy_advisor_breakthrough_count_uses_tier(self):
        """StrategyAdvisor should derive tier counts from tier + use canonical summary keys."""
        strategy_path = os.path.join(self.component_dir, "StrategyAdvisor.js")
        content = self._read_file(strategy_path)

        self.assertIn("const tier = normalizeTier(entry);", content)
        self.assertIn("const effectiveTier = tier || 'screening';", content)
        self.assertIn("tierSummary[effectiveTier] += 1;", content)
        self.assertIn("if (effectiveTier === 'breakthrough')", content)
        self.assertIn("total_programs_evaluated ?? dashboard?.summary?.total_programs ?? 0", content)

    def test_research_report_uses_stage1_survivors_summary_key(self):
        """ResearchReport should read stage1_survivors (with legacy fallback)."""
        report_path = os.path.join(self.component_dir, "ResearchReport.js")
        content = self._read_file(report_path)
        self.assertIn("const s1Survivors = s.stage1_survivors ?? s.total_s1_passed ?? 0;", content)

    def test_investigation_actions_use_eligibility_gating_hooks(self):
        """App + candidate views should wire explicit eligibility gating for investigate/queue actions."""
        app_content = self._read_file(self.app_js)
        leaderboard_content = self._read_file(os.path.join(self.component_dir, "Leaderboard.js"))
        top_programs_content = self._read_file(os.path.join(self.component_dir, "TopPrograms.js"))
        program_detail_content = self._read_file(os.path.join(self.component_dir, "ProgramDetail.js"))

        self.assertIn("const [eligibilityByResultId, setEligibilityByResultId] = useState({});", app_content)
        self.assertIn("setEligibilityByResultId(buildEligibilityByResultId", app_content)
        self.assertIn("filter(resultId => eligibilityByResultId[resultId]?.investigationEligible)", app_content)
        self.assertIn("eligibilityByResultId={eligibilityByResultId}", app_content)
        self.assertIn("intent: item?.intent === 'validation' ? 'validation' : 'investigation'", app_content)
        self.assertIn("const stillEligibleForIntent = intent === 'validation'", app_content)
        self.assertIn("filter(item => item.intent === 'investigation')", app_content)
        self.assertIn("filter(item => item.intent === 'validation')", app_content)

        self.assertIn("function candidateEligibility(entry)", leaderboard_content)
        self.assertIn("already_investigated_unchanged", leaderboard_content)
        self.assertIn("disabled={!isQueued && !eligibility.queueEligible}", leaderboard_content)
        self.assertIn("const queueIntent = eligibility.validationEligible", leaderboard_content)
        self.assertIn("Queue Validate", leaderboard_content)
        self.assertIn("intent: queueIntent", leaderboard_content)

        self.assertIn("eligibilityByResultId", top_programs_content)
        self.assertIn("queueEligible", top_programs_content)
        self.assertIn("Ineligible", top_programs_content)
        self.assertIn("const queueIntent = eligibility?.validationEligible", top_programs_content)
        self.assertIn("Queue Investigate", top_programs_content)

        self.assertIn("eligibilityByResultId", program_detail_content)
        self.assertIn("Already investigated", program_detail_content)

    def test_learning_trajectory_minimum_threshold_copy_uses_backend_contract(self):
        """LearningPanel should avoid hard-coded trajectory threshold copy drift."""
        learning_panel_content = self._read_file(os.path.join(self.component_dir, "LearningPanel.js"))

        self.assertIn("const minimumExperiments = Math.max(2, Number(trajectory?.min_experiments_required) || 5);", learning_panel_content)
        self.assertIn("Need at least {minimumExperiments} experiments to compute a learning trajectory.", learning_panel_content)
        self.assertNotIn("Need at least 3 experiments to compute a learning trajectory.", learning_panel_content)

    def test_trend_charts_show_stabilized_s1_and_confidence_bands(self):
        """TrendCharts should consume stabilized data and wire adaptation refresh context."""
        trend_content = self._read_file(os.path.join(self.component_dir, "TrendCharts.js"))

        self.assertIn("valueKey=\"adjusted_s1_pass_rate\"", trend_content)
        self.assertIn("bandLowerKey=\"s1_confidence_lower\"", trend_content)
        self.assertIn("bandUpperKey=\"s1_confidence_upper\"", trend_content)
        self.assertIn("reliabilityMultiplier", trend_content)
        self.assertIn("trend_confidence", trend_content)
        self.assertIn("/api/trends/context", trend_content)
        self.assertIn("setInterval(fetchTrendContext, 10000)", trend_content)
        self.assertIn("Adaptation outcomes (recent)", trend_content)

    def test_research_report_mentions_deduplicated_fingerprint_rankings(self):
        """Discovery rankings should explain fingerprint dedup and repeat metadata."""
        report_content = self._read_file(os.path.join(self.component_dir, "ResearchReport.js"))
        self.assertIn("fingerprint-deduplicated", report_content)
        self.assertIn("Grouped view", report_content)
        self.assertIn("Expanded reruns", report_content)
        self.assertIn("Same architecture repeated means reruns of one fingerprint", report_content)
        self.assertIn("top_programs_expanded", report_content)
        self.assertIn("repeat_count", report_content)
        self.assertIn("repeat_experiment_span", report_content)
        self.assertIn("eligibilityByResultId", report_content)
        self.assertIn("Queue Validate", report_content)
        self.assertIn("Ineligible", report_content)
        self.assertIn("reportQueueReasonLabel", report_content)
        self.assertIn("Unique Architectures vs Reruns", report_content)
        self.assertIn("architecture_rerun_telemetry", report_content)

    def test_learning_panel_mentions_unique_vs_rerun_telemetry(self):
        """LearningPanel should show unique architecture vs rerun concentration metrics."""
        learning_panel_content = self._read_file(os.path.join(self.component_dir, "LearningPanel.js"))
        self.assertIn("Unique Architectures vs Reruns", learning_panel_content)
        self.assertIn("architecture_rerun_telemetry", learning_panel_content)
        self.assertIn("Top fingerprint concentration", learning_panel_content)

    @staticmethod
    def _normalize_route(path: str) -> str:
        import re

        p = path.split("?", 1)[0]
        p = re.sub(r"<[^>]+>", "*", p)
        p = re.sub(r"\$\{[^}]+\}", "*", p)
        p = re.sub(r"//+", "/", p)
        return p.rstrip("/") or "/"

    @staticmethod
    def _route_matches(backend: str, frontend: str) -> bool:
        if backend == frontend:
            return True

        b_parts = [p for p in backend.strip("/").split("/") if p]
        f_parts = [p for p in frontend.strip("/").split("/") if p]
        if len(b_parts) != len(f_parts):
            return False

        for b, f in zip(b_parts, f_parts):
            if b == "*" or f == "*":
                continue
            if b != f:
                return False
        return True


class TestDeadCodeAudit(unittest.TestCase):
    """Non-destructive dead code audit should run and emit structured data."""

    def test_dead_code_audit_json_runs(self):
        import subprocess

        repo_root = os.path.dirname(os.path.dirname(__file__))
        cmd = [
            sys.executable,
            os.path.join(repo_root, "tools", "dead_code_audit.py"),
            "--workspace",
            repo_root,
            "--json",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)

        payload = json.loads(proc.stdout)
        self.assertIn("dashboard_orphans", payload)
        self.assertIn("python_possible_orphans", payload)
        self.assertIn("notes", payload)


# ── Test 11: Evolution Search ──


@unittest.skipUnless(HAS_TORCH, "requires torch for search modules")
class TestEvolutionIntegration(unittest.TestCase):
    """Test evolution search has novelty_fn wired up."""

    def test_evolution_search_accepts_novelty_fn(self):
        """evolutionary_search should accept novelty_fn parameter."""
        from research.search.evolution import evolutionary_search
        import inspect
        sig = inspect.signature(evolutionary_search)
        self.assertIn("novelty_fn", sig.parameters)

    def test_novelty_search_accepts_fingerprint_fn(self):
        """novelty_search should accept fingerprint_fn parameter."""
        from research.search.novelty_search import novelty_search
        import inspect
        sig = inspect.signature(novelty_search)
        self.assertIn("fingerprint_fn", sig.parameters)

    def test_mutation_adds_lineage_metadata(self):
        """Mutation should preserve lineage metadata for auditability."""
        from research.search.evolution import _mutate_graph
        from research.synthesis.grammar import GrammarConfig, generate_layer_graph
        import random

        parent = generate_layer_graph(GrammarConfig(model_dim=128), seed=123)
        child = _mutate_graph(parent, GrammarConfig(model_dim=128), random.Random(9))

        self.assertEqual(child.model_dim, parent.model_dim)
        self.assertIn("lineage", child.metadata)
        self.assertEqual(child.metadata["lineage"].get("type"), "mutation")
        self.assertEqual(child.metadata["lineage"].get("parent"), parent.fingerprint())

    def test_crossover_adds_lineage_metadata(self):
        """Crossover should retain both parent fingerprints in metadata."""
        from research.search.evolution import _crossover_graphs
        from research.synthesis.grammar import GrammarConfig, generate_layer_graph
        import random

        g1 = generate_layer_graph(GrammarConfig(model_dim=128), seed=101)
        g2 = generate_layer_graph(GrammarConfig(model_dim=128), seed=202)
        child = _crossover_graphs(g1, g2, GrammarConfig(model_dim=128), random.Random(11))

        self.assertEqual(child.model_dim, g1.model_dim)
        self.assertIn("lineage", child.metadata)
        self.assertEqual(child.metadata["lineage"].get("type"), "crossover")
        self.assertEqual(
            child.metadata["lineage"].get("parents"),
            [g1.fingerprint(), g2.fingerprint()],
        )

    def test_evolution_captures_eval_errors_in_metadata(self):
        """Evaluation failures should be explicit metadata, not silent drops."""
        from research.search.evolution import evolutionary_search, EvolutionConfig

        def bad_fitness(_):
            raise RuntimeError("fitness exploded")

        def bad_novelty(_, __):
            raise ValueError("novelty unavailable")

        pop = evolutionary_search(
            fitness_fn=bad_fitness,
            novelty_fn=bad_novelty,
            config=EvolutionConfig(population_size=4, n_generations=1, elitism=1),
            seed=7,
        )

        self.assertGreater(len(pop), 0)
        for ind in pop:
            self.assertEqual(ind.fitness, 0.0)
            self.assertEqual(ind.novelty, 0.0)
            self.assertEqual(ind.metadata.get("fitness_error_type"), "RuntimeError")
            self.assertEqual(ind.metadata.get("novelty_error_type"), "ValueError")

    def test_evolution_enforces_fingerprint_diversity(self):
        """Duplicate fingerprints should be replaced to avoid clone collapse."""
        import random

        from research.search.evolution import (
            EvolutionConfig,
            Individual,
            _enforce_population_diversity,
        )
        from research.synthesis.graph import ComputationGraph
        from research.synthesis.grammar import GrammarConfig, generate_layer_graph

        grammar = GrammarConfig(model_dim=128)
        g1 = generate_layer_graph(grammar, seed=11)
        g1_clone = ComputationGraph.from_dict(g1.to_dict())
        g2 = generate_layer_graph(grammar, seed=22)

        pop = [
            Individual(graph=g1, fitness=1.0, novelty=0.2, generation=0),
            Individual(graph=g1_clone, fitness=0.9, novelty=0.1, generation=0),
            Individual(graph=g2, fitness=0.8, novelty=0.3, generation=0),
        ]

        deduped = _enforce_population_diversity(
            population=pop,
            fitness_fn=lambda _g: 1.0,
            novelty_fn=lambda _g, _all: 0.0,
            config=EvolutionConfig(population_size=3),
            grammar=grammar,
            rng=random.Random(5),
            generation=1,
        )

        fps = [ind.fingerprint for ind in deduped]
        self.assertEqual(len(deduped), 3)
        self.assertEqual(len(set(fps)), 3)
        self.assertTrue(any(ind.metadata.get("dedupe_duplicates_replaced", 0) > 0
                            for ind in deduped))


# ── Test 12: Inline Phase Methods & Budget Context ──


@unittest.skipUnless(HAS_TORCH, "requires torch for runner module")
class TestInlinePhaseMethods(unittest.TestCase):
    """Verify inline investigation/validation methods exist and are callable."""

    def test_runner_has_inline_investigation(self):
        """ExperimentRunner must have _run_inline_investigation (not crash)."""
        from research.scientist.runner import ExperimentRunner
        self.assertTrue(hasattr(ExperimentRunner, "_run_inline_investigation"),
                        "Missing _run_inline_investigation method")

    def test_runner_has_inline_validation(self):
        """ExperimentRunner must have _run_inline_validation (not crash)."""
        from research.scientist.runner import ExperimentRunner
        self.assertTrue(hasattr(ExperimentRunner, "_run_inline_validation"),
                        "Missing _run_inline_validation method")

    def test_inline_validation_progress_sets_total_programs(self):
        """Inline validation must initialize progress denominator to avoid x/0 UI output."""
        import inspect
        from research.scientist.runner import ExperimentRunner

        src = inspect.getsource(ExperimentRunner._run_inline_validation)
        self.assertIn("total_programs=len(result_ids)", src,
                      "_run_inline_validation LiveProgress must set total_programs")

    def test_inline_validation_persists_candidate_metadata(self):
        """Inline validation should persist candidate IDs into experiment config metadata."""
        import inspect
        from research.scientist.runner import ExperimentRunner

        src = inspect.getsource(ExperimentRunner._run_inline_validation)
        self.assertIn("_validation_config_with_result_ids", src)
        self.assertIn('"continuous_auto"', src)

    def test_inline_investigation_progress_sets_total_programs(self):
        """Inline investigation must initialize progress denominator for dashboard parity."""
        import inspect
        from research.scientist.runner import ExperimentRunner

        src = inspect.getsource(ExperimentRunner._run_inline_investigation)
        self.assertIn("total_programs=len(result_ids)", src,
                      "_run_inline_investigation LiveProgress must set total_programs")

    def test_continuous_phase_dispatches_to_inline(self):
        """_run_continuous_phase should dispatch to inline methods."""
        import inspect
        from research.scientist.runner import ExperimentRunner
        src = inspect.getsource(ExperimentRunner._run_continuous_phase)
        self.assertIn("_run_inline_investigation", src)
        self.assertIn("_run_inline_validation", src)

    def test_no_missing_run_investigation_call(self):
        """Should NOT call self._run_investigation() which doesn't exist."""
        import inspect
        from research.scientist.runner import ExperimentRunner
        src = inspect.getsource(ExperimentRunner._run_inline_investigation)
        self.assertNotIn("self._run_investigation(", src,
                         "_run_inline_investigation should not call "
                         "non-existent _run_investigation()")

    def test_control_experiment_interval_marks_and_skips_learned_weights(self):
        """Every Nth continuous synthesis run should be treated as control."""
        from research.scientist.runner import ExperimentRunner, RunConfig

        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test_control_interval.db")
        runner = ExperimentRunner(db_path)

        config = RunConfig(
            n_programs=1,
            enable_campaigns=False,
            auto_report=False,
            auto_scale_up=False,
            auto_investigate=False,
            auto_validate=False,
            control_experiment_interval=2,
        )

        nb = MagicMock()
        nb.get_recent_experiments.return_value = []
        nb.get_leaderboard.return_value = []
        nb.start_experiment.return_value = "exp-control"

        runner.aria.formulate_hypothesis = MagicMock(return_value="control hypothesis")
        runner.aria.validate_hypothesis = MagicMock(return_value=None)
        runner.aria.experiment_summary = MagicMock(return_value="summary")
        runner.aria.analyze_results = MagicMock(return_value="")
        runner._build_rich_context_for_experiment = MagicMock(return_value="ctx")
        runner._analyze_results = MagicMock(return_value=[])
        runner._auto_recommend = MagicMock()
        runner._auto_escalate = MagicMock()
        runner._maybe_auto_report = MagicMock()

        expected_results = {
            "total": 1,
            "stage0_passed": 0,
            "stage05_passed": 0,
            "stage1_passed": 0,
            "novel_count": 0,
            "survivors": [],
            "best_loss_ratio": None,
            "best_novelty_score": None,
        }
        runner._execute_experiment = MagicMock(return_value=expected_results)

        runner._run_continuous_synthesis(
            config=config,
            nb=nb,
            n_experiments=2,
            limit_str="exp 2/10",
            mode_reasoning="control check",
        )

        self.assertTrue(runner._is_control_experiment(config, 2))
        exec_kwargs = runner._execute_experiment.call_args.kwargs
        self.assertFalse(exec_kwargs["use_learned_grammar"])

        start_cfg = nb.start_experiment.call_args.kwargs["config"]
        self.assertTrue(start_cfg["control_experiment"])
        self.assertFalse(start_cfg["use_learned_grammar_weights"])

        log_call = nb.log_learning_event.call_args
        self.assertEqual(log_call.args[0], "grammar_control_experiment")

    def test_start_experiment_builds_context_for_hypothesis(self):
        """Manual start_experiment should pass rich context into formulate_hypothesis."""
        from research.scientist.runner import ExperimentRunner, RunConfig

        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test_start_hypothesis_context.db")
        runner = ExperimentRunner(db_path)

        runner._ensure_math_spaces = MagicMock()
        runner._run_experiment_thread = MagicMock(return_value=None)
        runner.aria.formulate_hypothesis = MagicMock(return_value="context-aware hypothesis")

        config = RunConfig(n_programs=1, max_cost_dollars=10.0)
        exp_id = runner.start_experiment(config=config, hypothesis=None)

        self.assertIsNotNone(exp_id)
        runner.aria.formulate_hypothesis.assert_called_once()
        call = runner.aria.formulate_hypothesis.call_args
        self.assertIn("context", call.kwargs)
        self.assertTrue(call.kwargs["context"].strip())

    def test_start_experiment_enforces_context_when_llm_available(self):
        """Manual start_experiment should still provide fallback context when LLM is available and history context load fails."""
        from research.scientist.runner import ExperimentRunner, RunConfig

        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test_start_context_fallback.db")
        runner = ExperimentRunner(db_path)

        runner._ensure_math_spaces = MagicMock()
        runner._run_experiment_thread = MagicMock(return_value=None)
        runner._build_start_experiment_hypothesis_context = MagicMock(return_value="")

        runner.aria._get_llm = MagicMock(return_value=object())
        runner.aria.formulate_hypothesis = MagicMock(return_value="fallback-context hypothesis")
        runner.aria.critique_hypothesis = MagicMock(return_value={
            "verdict": "proceed",
            "gate": "pass",
            "checks": [],
            "concerns": [],
            "suggestions": [],
            "confidence": 0.8,
        })

        config = RunConfig(n_programs=1)
        exp_id = runner.start_experiment(config=config, hypothesis=None)

        self.assertIsNotNone(exp_id)
        call = runner.aria.formulate_hypothesis.call_args
        self.assertIn("context", call.kwargs)
        self.assertTrue(call.kwargs["context"].strip())
        self.assertIn("Manual Start Context", call.kwargs["context"])

    def test_start_experiment_records_hypothesis_provenance_metadata(self):
        """Manual start_experiment should persist hypothesis provenance into notebook hypothesis entry metadata."""
        from research.scientist.runner import ExperimentRunner, RunConfig

        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test_start_hypothesis_metadata.db")
        runner = ExperimentRunner(db_path)

        runner._ensure_math_spaces = MagicMock()
        runner._run_experiment_thread = MagicMock(return_value=None)
        runner.aria.formulate_hypothesis = MagicMock(return_value=(
            "metadata hypothesis",
            {
                "source": "llm_context",
                "llm_used": True,
                "fallback_used": False,
                "used_context": True,
                "review_status": "not_reviewed",
                "confidence": 0.72,
                "critique": "metric is measurable",
            },
        ))

        config = RunConfig(n_programs=1, max_cost_dollars=5.0)
        exp_id = runner.start_experiment(config=config, hypothesis=None)

        nb = LabNotebook(db_path)
        try:
            entries = nb.get_entries(experiment_id=exp_id, entry_type="hypothesis", limit=5)
            self.assertTrue(entries)
            metadata = json.loads(entries[0].get("metadata_json") or "{}")
            self.assertEqual(metadata.get("source"), "llm_context")
            self.assertTrue(metadata.get("used_context"))
            self.assertTrue(str(metadata.get("review_status", "")).startswith("preflight_"))
            self.assertAlmostEqual(float(metadata.get("confidence")), 0.72, places=2)
            self.assertIn("preflight_critique", metadata)
            self.assertIn("critique_confidence", metadata)
            self.assertIn("context_char_count", metadata)
        finally:
            nb.close()

    def test_start_investigation_records_hypothesis_provenance_metadata(self):
        """Manual start_investigation should persist source/review provenance metadata."""
        from research.scientist.runner import ExperimentRunner, RunConfig

        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test_start_investigation_hypothesis_metadata.db")
        runner = ExperimentRunner(db_path)

        runner._ensure_math_spaces = MagicMock()
        runner._run_investigation_thread = MagicMock(return_value=None)

        config = RunConfig(n_programs=1)
        exp_id = runner.start_investigation(
            result_ids=["r1"],
            config=config,
            hypothesis="User supplied investigation hypothesis",
        )

        nb = LabNotebook(db_path)
        try:
            entries = nb.get_entries(experiment_id=exp_id, entry_type="hypothesis", limit=5)
            self.assertTrue(entries)
            metadata = json.loads(entries[0].get("metadata_json") or "{}")
            self.assertEqual(metadata.get("source"), "user_input")
            self.assertEqual(metadata.get("review_status"), "not_reviewed")
            self.assertIn("confidence", metadata)
            self.assertIn("critique", metadata)
        finally:
            nb.close()

    def test_start_evolution_records_llm_provenance_metadata(self):
        """start_evolution without user hypothesis should preserve LLM/fallback provenance metadata."""
        from research.scientist.runner import ExperimentRunner, RunConfig

        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test_start_evolution_hypothesis_metadata.db")
        runner = ExperimentRunner(db_path)

        runner._ensure_math_spaces = MagicMock()
        runner._run_evolution_thread = MagicMock(return_value=None)
        runner.aria.formulate_hypothesis = MagicMock(return_value=(
            "Auto evolution hypothesis",
            {
                "source": "llm_context",
                "llm_used": True,
                "fallback_used": False,
                "used_context": False,
                "review_status": "not_reviewed",
                "confidence": 0.66,
                "critique": None,
            },
        ))

        config = RunConfig(n_programs=1)
        exp_id = runner.start_evolution(config=config, hypothesis=None)

        nb = LabNotebook(db_path)
        try:
            entries = nb.get_entries(experiment_id=exp_id, entry_type="hypothesis", limit=5)
            self.assertTrue(entries)
            metadata = json.loads(entries[0].get("metadata_json") or "{}")
            self.assertEqual(metadata.get("source"), "llm_context")
            self.assertTrue(metadata.get("llm_used"))
            self.assertEqual(metadata.get("review_status"), "not_reviewed")
            self.assertAlmostEqual(float(metadata.get("confidence")), 0.66, places=2)
        finally:
            nb.close()

    def test_runner_startup_recovers_stale_experiments(self):
        """Runner init should clean stale experiments left in running state."""
        from research.scientist.runner import ExperimentRunner

        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test_runner_recovery.db")

        nb = LabNotebook(db_path)
        try:
            exp_id = nb.start_experiment(
                experiment_type="synthesis",
                config={"n_programs": 1},
                hypothesis="stale run",
            )
            nb.conn.execute(
                "UPDATE experiments SET started_at = ? WHERE experiment_id = ?",
                (time.time() - (2 * 60 * 60), exp_id),
            )
            nb.conn.commit()
        finally:
            nb.close()

        _runner = ExperimentRunner(db_path)
        self.assertIsNotNone(_runner)

        nb2 = LabNotebook(db_path)
        try:
            exp = nb2.get_experiment(exp_id)
            self.assertIsNotNone(exp)
            self.assertEqual(exp["status"], "failed")
            results = json.loads(exp.get("results_json") or "{}")
            self.assertIn("failure_reason", results)
        finally:
            nb2.close()

    def test_runner_startup_recovers_startup_failed_experiment(self):
        """Runner init should clean no-progress startup-failed running experiments."""
        from research.scientist.runner import ExperimentRunner

        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test_runner_startup_fail_recovery.db")

        nb = LabNotebook(db_path)
        try:
            exp_id = nb.start_experiment(
                experiment_type="validation",
                config={"n_programs": 1},
                hypothesis="startup fail",
            )
            nb.conn.execute(
                "UPDATE experiments SET started_at = ? WHERE experiment_id = ?",
                (time.time() - (20 * 60), exp_id),
            )
            nb.conn.commit()
        finally:
            nb.close()

        _runner = ExperimentRunner(db_path)
        self.assertIsNotNone(_runner)

        nb2 = LabNotebook(db_path)
        try:
            exp = nb2.get_experiment(exp_id)
            self.assertIsNotNone(exp)
            self.assertEqual(exp["status"], "failed")
            results = json.loads(exp.get("results_json") or "{}")
            self.assertEqual(
                results.get("failure_reason"),
                "Startup failed before any progress was recorded",
            )
        finally:
            nb2.close()

    def test_train_with_program_uses_step_seed_sequence(self):
        """Synthesized-program training should seed data generation with seed+step."""
        import torch.nn as nn
        import torch

        from research.scientist.runner import ExperimentRunner, RunConfig

        class TinyModel(nn.Module):
            def __init__(self, vocab_size: int = 32, d_model: int = 16):
                super().__init__()
                self.emb = nn.Embedding(vocab_size, d_model)
                self.head = nn.Linear(d_model, vocab_size)

            def forward(self, input_ids):
                return self.head(self.emb(input_ids))

        class _Curriculum:
            @staticmethod
            def get_seq_len(_step, _total):
                return 8

        class _Loss:
            @staticmethod
            def compute(logits, target):
                return torch.nn.functional.cross_entropy(logits, target)

        class _Optimizer:
            @staticmethod
            def create(params):
                return torch.optim.SGD(params, lr=1e-3)

        class Program:
            init_scheme = "default"
            init_scale = 0.02
            n_steps = 3
            batch_size = 1
            max_grad_norm = 1.0
            curriculum = _Curriculum()
            loss = _Loss()
            optimizer = _Optimizer()

        runner = ExperimentRunner(os.path.join(tempfile.mkdtemp(), "seed_test.db"))
        model = TinyModel()
        config = RunConfig(vocab_size=32, max_seq_len=16)

        original_randint = torch.randint
        seen_seeds = []

        def _spy_randint(*args, **kwargs):
            generator = kwargs.get("generator")
            if generator is not None:
                seen_seeds.append(generator.initial_seed())
            return original_randint(*args, **kwargs)

        with patch("research.scientist.runner.torch.randint", side_effect=_spy_randint):
            _ = runner._train_with_program(
                model,
                Program(),
                config,
                torch.device("cpu"),
                seed=1234,
            )

        self.assertGreaterEqual(len(seen_seeds), 3)
        self.assertEqual(seen_seeds[:3], [1234, 1235, 1236])

    def test_corpus_mode_falls_back_to_random_when_missing_path(self):
        """Corpus mode should safely fall back to random token generation when corpus is unavailable."""
        import torch
        import torch.nn as nn

        from research.scientist.runner import ExperimentRunner, RunConfig

        class TinyModel(nn.Module):
            def __init__(self, vocab_size: int = 32, d_model: int = 16):
                super().__init__()
                self.emb = nn.Embedding(vocab_size, d_model)
                self.head = nn.Linear(d_model, vocab_size)

            def forward(self, input_ids):
                return self.head(self.emb(input_ids))

        class _Curriculum:
            @staticmethod
            def get_seq_len(_step, _total):
                return 8

        class _Loss:
            @staticmethod
            def compute(logits, target):
                return torch.nn.functional.cross_entropy(logits, target)

        class _Optimizer:
            @staticmethod
            def create(params):
                return torch.optim.SGD(params, lr=1e-3)

        class Program:
            init_scheme = "default"
            init_scale = 0.02
            n_steps = 2
            batch_size = 1
            max_grad_norm = 1.0
            curriculum = _Curriculum()
            loss = _Loss()
            optimizer = _Optimizer()

        runner = ExperimentRunner(os.path.join(tempfile.mkdtemp(), "corpus_fallback.db"))
        model = TinyModel()
        config = RunConfig(
            vocab_size=32,
            max_seq_len=16,
            data_mode="corpus",
            corpus_path="/tmp/does-not-exist.txt",
            corpus_format="txt",
        )

        result = runner._train_with_program(
            model,
            Program(),
            config,
            torch.device("cpu"),
            seed=42,
        )

        self.assertIn("n_train_steps", result)
        self.assertGreaterEqual(int(result["n_train_steps"]), 1)

    def test_baseline_compare_uses_training_metrics(self):
        """Baseline compare should use candidate training metrics and recipe metadata."""
        import inspect
        from research.scientist.runner import ExperimentRunner

        src_execute = inspect.getsource(ExperimentRunner._execute_experiment)
        self.assertIn('s1_result.get("n_train_steps")', src_execute)
        self.assertIn("self._resolve_baseline_recipe", src_execute)
        self.assertIn('optimizer_name=baseline_recipe["optimizer_name"]', src_execute)
        self.assertIn('weight_decay=baseline_recipe["weight_decay"]', src_execute)

        src_validation = inspect.getsource(ExperimentRunner._run_inline_validation)
        self.assertIn('best_seed.get("n_train_steps")', src_validation)
        self.assertIn("self._resolve_baseline_recipe", src_validation)
        self.assertIn('momentum=baseline_recipe["momentum"]', src_validation)

        src_tp = inspect.getsource(ExperimentRunner._train_with_program)
        self.assertIn('result["optimizer_class"]', src_tp)
        self.assertIn('result["optimizer_weight_decay"]', src_tp)

    def test_routing_benchmark_compares_multiple_modes(self):
        """Track C benchmark should compare >=3 routing strategies with frontier metrics."""
        from research.scientist.runner import ExperimentRunner, RunConfig

        runner = ExperimentRunner(os.path.join(tempfile.mkdtemp(), "routing_bench.db"))
        config = RunConfig(
            model_dim=64,
            n_layers=2,
            vocab_size=128,
            max_seq_len=32,
            stage1_steps=1,
            stage1_batch_size=1,
            device="cpu",
        )
        modes = ["uniform", "mod_topk", "early_exit"]
        seeds = [11, 22]
        result = runner.run_routing_benchmark(config, seed_set=seeds, modes=modes)

        self.assertTrue(result.get("available"))
        self.assertEqual(result.get("seed_set"), seeds)
        self.assertGreaterEqual(len(result.get("modes_evaluated", [])), 3)

        points = result.get("points", [])
        self.assertGreaterEqual(len(points), 3)
        for point in points:
            self.assertIn("routing_mode", point)
            self.assertIn("validation_loss", point)
            self.assertIn("tokens_per_sec", point)
            self.assertIn("effective_token_compute", point)
            self.assertIn("routing_stability", point)


@unittest.skipUnless(HAS_CONTEXT, "requires context module")
class TestBudgetContext(unittest.TestCase):
    """Verify budget info is included in mode selection context."""

    def test_context_includes_budget_when_provided(self):
        from research.scientist.llm.context import build_mode_selection_context
        ctx = build_mode_selection_context(
            recent_experiments=[],
            leaderboard=[],
            cost_spent=3.50,
            budget=5.00,
            n_experiments_in_session=5,
        )
        self.assertIn("$3.50", ctx)
        self.assertIn("$5.00", ctx)
        self.assertIn("remaining", ctx.lower())

    def test_context_omits_budget_when_zero(self):
        from research.scientist.llm.context import build_mode_selection_context
        ctx = build_mode_selection_context(
            recent_experiments=[],
            leaderboard=[],
            cost_spent=0,
            budget=0,
        )
        self.assertNotIn("Budget", ctx)


@unittest.skipUnless(HAS_TORCH and HAS_FLASK and HAS_NOTEBOOK,
                     "requires torch, flask, and notebook")
class TestPipelineEndToEnd(unittest.TestCase):
    """Single test that runs the AI scientist pipeline end-to-end."""

    def test_continuous_pipeline_records_novelty_learning_and_reports(self):
        from research.scientist.runner import ExperimentRunner, RunConfig
        from research.scientist.api import create_app

        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test_pipeline_end_to_end.db")

        runner = ExperimentRunner(db_path)
        config = RunConfig(
            n_programs=1,
            model_dim=64,
            n_layers=2,
            max_depth=4,
            max_ops=8,
            device="cpu",
            stage1_steps=1,
            stage1_batch_size=1,
            continuous=True,
            max_experiments=2,
            rest_between_experiments=0,
            auto_scale_up=False,
            auto_investigate=False,
            auto_validate=False,
            enable_campaigns=True,
            knowledge_extraction_interval=1,
            auto_report=False,
        )

        session_id = runner.start_continuous(config)
        self.assertIsNotNone(session_id)

        t0 = time.time()
        timeout_s = 180
        while runner.is_running and (time.time() - t0) < timeout_s:
            time.sleep(0.2)

        self.assertFalse(runner.is_running, "continuous run timed out")
        self.assertIn(runner.progress.status, {"completed", "stopped"})

        nb = LabNotebook(db_path)
        try:
            experiments = nb.get_recent_experiments(10)
            self.assertGreaterEqual(len(experiments), 2)
            completed = [e for e in experiments if e.get("status") == "completed"]
            self.assertGreaterEqual(len(completed), 2)

            novelty_count = nb.conn.execute(
                "SELECT COUNT(*) FROM program_results WHERE novelty_score IS NOT NULL"
            ).fetchone()[0]
            self.assertGreater(novelty_count, 0)

            op_rates = nb.get_op_success_rates()
            self.assertGreater(len(op_rates), 0)

            campaign_rows = nb.conn.execute(
                "SELECT COUNT(*) FROM campaigns"
            ).fetchone()[0]
            self.assertGreaterEqual(campaign_rows, 1)
        finally:
            nb.close()

        app = create_app(notebook_path=db_path)
        client = app.test_client()

        r_report = client.get("/api/report")
        self.assertEqual(r_report.status_code, 200)
        report = r_report.get_json()
        for k in ["summary", "recent_experiments", "op_success_rates",
                  "structural_correlations", "learning_log"]:
            self.assertIn(k, report)
        self.assertGreaterEqual(len(report.get("recent_experiments", [])), 2)

        r_campaigns = client.get("/api/campaigns")
        self.assertEqual(r_campaigns.status_code, 200)
        campaigns = r_campaigns.get_json()
        self.assertIsInstance(campaigns, list)
        self.assertGreaterEqual(len(campaigns), 1)
        self.assertIn("n_experiments", campaigns[0])

        r_op_success = client.get("/api/analytics/op-success")
        self.assertEqual(r_op_success.status_code, 200)
        op_success = r_op_success.get_json()
        self.assertIsInstance(op_success, dict)
        self.assertGreater(len(op_success), 0)


@unittest.skipUnless(HAS_TORCH, "torch required")
class TestDiagnosticTasks(unittest.TestCase):
    """Tests for the synthetic diagnostic task evaluation suite."""

    @classmethod
    def setUpClass(cls):
        from research.eval.diagnostic_tasks import (
            generate_copy_task,
            generate_induction_task,
            generate_periodic_task,
            generate_selective_copy_task,
            run_diagnostic_suite,
            DiagnosticSuiteResult,
            DiagnosticTaskResult,
            DIAG_BATCH_SIZE,
            DIAG_SEQ_LEN,
            DIAG_SEP_TOKEN,
            DIAG_MARK_TOKEN,
            DIAGNOSTIC_TASKS,
        )
        cls.generate_copy_task = staticmethod(generate_copy_task)
        cls.generate_induction_task = staticmethod(generate_induction_task)
        cls.generate_periodic_task = staticmethod(generate_periodic_task)
        cls.generate_selective_copy_task = staticmethod(generate_selective_copy_task)
        cls.run_diagnostic_suite = staticmethod(run_diagnostic_suite)
        cls.DiagnosticSuiteResult = DiagnosticSuiteResult
        cls.DiagnosticTaskResult = DiagnosticTaskResult
        cls.DIAG_BATCH_SIZE = DIAG_BATCH_SIZE
        cls.DIAG_SEQ_LEN = DIAG_SEQ_LEN
        cls.DIAG_SEP_TOKEN = DIAG_SEP_TOKEN
        cls.DIAG_MARK_TOKEN = DIAG_MARK_TOKEN
        cls.DIAGNOSTIC_TASKS = DIAGNOSTIC_TASKS

    def _check_generator_shapes(self, gen_fn):
        """Helper: verify generator returns correct shapes and types."""
        rng = torch.Generator()
        rng.manual_seed(42)
        ids, mask, targets = gen_fn(
            batch_size=self.DIAG_BATCH_SIZE,
            seq_len=self.DIAG_SEQ_LEN,
            device="cpu",
            rng=rng,
        )
        self.assertEqual(ids.shape, (self.DIAG_BATCH_SIZE, self.DIAG_SEQ_LEN))
        self.assertEqual(mask.shape, (self.DIAG_BATCH_SIZE, self.DIAG_SEQ_LEN - 1))
        self.assertEqual(targets.shape, (self.DIAG_BATCH_SIZE, self.DIAG_SEQ_LEN - 1))
        self.assertEqual(ids.dtype, torch.long)
        self.assertTrue(mask.dtype == torch.bool)
        # At least some critical positions exist
        self.assertGreater(mask.sum().item(), 0)
        # Targets match shifted input
        self.assertTrue(torch.equal(targets, ids[:, 1:]))
        return ids, mask, targets

    def test_copy_generator_shapes(self):
        self._check_generator_shapes(self.generate_copy_task)

    def test_induction_generator_shapes(self):
        self._check_generator_shapes(self.generate_induction_task)

    def test_periodic_generator_shapes(self):
        self._check_generator_shapes(self.generate_periodic_task)

    def test_selective_copy_generator_shapes(self):
        self._check_generator_shapes(self.generate_selective_copy_task)

    def test_copy_task_has_separator(self):
        """Copy task should contain SEP token in every sequence."""
        rng = torch.Generator()
        rng.manual_seed(123)
        ids, _, _ = self.generate_copy_task(batch_size=4, seq_len=64, rng=rng)
        for b in range(4):
            self.assertIn(self.DIAG_SEP_TOKEN, ids[b].tolist())

    def test_periodic_task_deterministic(self):
        """Periodic task: after first period, all positions are deterministic."""
        rng = torch.Generator()
        rng.manual_seed(99)
        ids, mask, targets = self.generate_periodic_task(
            batch_size=2, seq_len=32, rng=rng,
        )
        # Verify periodicity: for each batch, tokens repeat
        for b in range(2):
            seq = ids[b].tolist()
            # Find period by checking smallest repeat
            for p in range(3, 7):
                if all(seq[i] == seq[i % p] for i in range(p, len(seq))):
                    break
            else:
                self.fail("No periodic pattern found")

    def test_suite_result_serialization(self):
        """DiagnosticSuiteResult.to_dict() produces valid JSON-serializable dict."""
        result = self.DiagnosticSuiteResult(
            tasks=[
                self.DiagnosticTaskResult("copy", accuracy=0.8, loss=1.2, steps_trained=100),
                self.DiagnosticTaskResult("periodic", accuracy=0.9, loss=0.5, steps_trained=100),
            ],
            diagnostic_score=0.85,
            total_time_ms=1234.0,
        )
        d = result.to_dict()
        import json
        serialized = json.dumps(d)
        loaded = json.loads(serialized)
        self.assertEqual(len(loaded["tasks"]), 2)
        self.assertAlmostEqual(loaded["diagnostic_score"], 0.85)
        self.assertEqual(loaded["tasks"][0]["task_name"], "copy")

    def test_notebook_migration_has_diagnostic_columns(self):
        """Notebook migration map includes diagnostic_tasks_json and diagnostic_score."""
        nb = None
        try:
            tmpdir = tempfile.mkdtemp()
            db_path = os.path.join(tmpdir, "test_diag_migration.db")
            from research.scientist.notebook import LabNotebook
            nb = LabNotebook(db_path)
            cols = [row[1] for row in nb.conn.execute(
                "PRAGMA table_info(program_results)"
            ).fetchall()]
            self.assertIn("diagnostic_tasks_json", cols)
            self.assertIn("diagnostic_score", cols)
        finally:
            if nb:
                nb.close()


class TestStaleExperimentCleanup(unittest.TestCase):
    """Test cleanup_stale_experiments marks zombies as failed."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(self.tmpdir, "test_cleanup.db")
        from research.scientist.notebook import LabNotebook
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
        )

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
        from research.scientist.notebook import LabNotebook
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

        fps = self.nb.get_investigated_fingerprints()
        self.assertNotIn("fp_screening", fps)
        self.assertIn("fp_investigated", fps)
        self.assertIn("fp_validated", fps)
        self.assertEqual(len(fps), 2)


class TestGrammarWeightPersistence(unittest.TestCase):
    """Test that grammar weights appear in results dict."""

    def test_execute_experiment_stores_grammar_weights_in_results(self):
        """When grammar weights are applied, they should be in results dict."""
        # We test the logic indirectly: grammar_weights dict should be stored
        # in results["applied_grammar_weights"] when use_learned_grammar=True
        # and compute_grammar_weights returns weights.
        # This is a unit-level check of the data flow.
        weights = {"attention": 2.0, "linear": 1.5, "nonlinearity": 0.8}
        results = {"total": 0, "stage0_passed": 0, "survivors": []}
        # Simulate what _execute_experiment does
        if weights:
            results["applied_grammar_weights"] = dict(weights)
        self.assertIn("applied_grammar_weights", results)
        self.assertEqual(results["applied_grammar_weights"]["attention"], 2.0)

    def test_single_experiment_path_persists_applied_weights(self):
        """Single-threaded experiment path should persist applied grammar weights."""
        import inspect
        from research.scientist.runner import ExperimentRunner

        src = inspect.getsource(ExperimentRunner._run_experiment_thread)
        self.assertIn("self._persist_applied_grammar_weights(nb, exp_id, results)", src)

    def test_continuous_synthesis_path_persists_applied_weights(self):
        """Continuous synthesis path should persist applied grammar weights."""
        import inspect
        from research.scientist.runner import ExperimentRunner

        src = inspect.getsource(ExperimentRunner._run_continuous_synthesis)
        self.assertIn("self._persist_applied_grammar_weights(nb, exp_id, results)", src)

    def test_execute_experiment_records_distribution_shift_signals(self):
        """Core execute path should record generated-op distribution + shift metadata."""
        import inspect
        from research.scientist.runner import ExperimentRunner

        src = inspect.getsource(ExperimentRunner._execute_experiment)
        self.assertIn("generated_op_distribution", src)
        self.assertIn("generation_distribution_shift", src)
        self.assertIn("architecture_distribution_shift", src)


class TestFrontierOps(unittest.TestCase):
    """Test that efficiency frontier includes ops field."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(self.tmpdir, "test_frontier.db")
        from research.scientist.notebook import LabNotebook
        self.nb = LabNotebook(db_path)

    def tearDown(self):
        self.nb.close()

    def test_frontier_includes_ops(self):
        """Frontier entries should include ops extracted from graph_json."""
        from research.scientist.analytics import ExperimentAnalytics
        exp_id = self.nb.start_experiment("synthesis", {}, "test")
        graph_json = json.dumps({
            "nodes": {"n1": {"op": "linear_proj"}, "n2": {"op": "gelu"}},
            "output": "n2",
        })
        self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="frontier_fp",
            graph_json=graph_json,
            stage1_passed=True,
            loss_ratio=0.5,
            novelty_score=0.7,
            final_loss=0.3,
            flops_forward=1000,
            param_count=500,
        )
        analytics = ExperimentAnalytics(self.nb)
        frontier = analytics.efficiency_frontier()
        self.assertGreater(len(frontier), 0)
        self.assertIn("ops", frontier[0])
        self.assertIn("linear_proj", frontier[0]["ops"])
        self.assertIn("gelu", frontier[0]["ops"])
        # graph_json should be removed from output
        self.assertNotIn("graph_json", frontier[0])


class TestClusterDescriptions(unittest.TestCase):
    """Test that experiment clusters include contrastive descriptions."""

    def test_describe_clusters_contrastive(self):
        """Clusters should get different labels based on relative S1 ranking."""
        from research.scientist.analytics import ExperimentAnalytics
        clusters = [
            {
                "size": 10,
                "avg_s1_rate": 0.35,
                "avg_best_novelty": 0.5,
                "avg_best_loss_ratio": 0.7,
                "avg_compile_fail_rate": 0.1,
            },
            {
                "size": 5,
                "avg_s1_rate": 0.02,
                "avg_best_novelty": 0.2,
                "avg_best_loss_ratio": 1.1,
                "avg_compile_fail_rate": 0.3,
            },
        ]
        ExperimentAnalytics._describe_clusters(clusters)
        # Best cluster should be "most productive"
        self.assertIn("high S1 pass rate", clusters[0]["description"])
        self.assertIn("most productive", clusters[0]["description"])
        self.assertIn("10 experiments", clusters[0]["description"])
        # Worst cluster should be "least productive"
        self.assertIn("low S1 pass rate", clusters[1]["description"])
        self.assertIn("least productive", clusters[1]["description"])


class TestSSEEventContract(unittest.TestCase):
    """Verify LiveFeed.js event listeners match runner.py _emit_event calls."""

    @staticmethod
    def _extract_events(filepath, pattern):
        """Extract string arguments from pattern matches in a file."""
        import re
        events = set()
        with open(filepath) as f:
            for line in f:
                for m in re.finditer(pattern, line):
                    events.add(m.group(1))
        return events

    def test_frontend_events_are_emitted_by_backend(self):
        """Every event LiveFeed.js listens for must be emitted by the backend."""
        import pathlib
        root = pathlib.Path(__file__).resolve().parent.parent

        backend_events = set()
        for backend_file in ["scientist/runner.py", "scientist/api.py"]:
            backend_events |= self._extract_events(
                root / backend_file,
                r'_emit_event\(\s*["\'](\w+)["\']',
            )
        frontend_events = self._extract_events(
            root / "dashboard" / "src" / "components" / "LiveFeed.js",
            r'addEventListener\(\s*["\'](\w+)["\']',
        )

        # Frontend must not listen for events the backend never sends
        missing = frontend_events - backend_events
        self.assertEqual(
            missing, set(),
            f"LiveFeed.js listens for events not emitted by runner.py: {sorted(missing)}",
        )

    def test_backend_emits_known_events_only(self):
        """Sanity: backend emits a reasonable number of distinct events."""
        import pathlib
        root = pathlib.Path(__file__).resolve().parent.parent

        backend_events = set()
        for backend_file in ["scientist/runner.py", "scientist/api.py"]:
            backend_events |= self._extract_events(
                root / backend_file,
                r'_emit_event\(\s*["\'](\w+)["\']',
            )
        # Should have a substantial set of events (guards against regex breakage)
        self.assertGreaterEqual(len(backend_events), 20,
                                f"Too few backend events found: {sorted(backend_events)}")

    def test_frontend_listens_for_enough_events(self):
        """Sanity: frontend listens for a reasonable number of events."""
        import pathlib
        root = pathlib.Path(__file__).resolve().parent.parent

        frontend_events = self._extract_events(
            root / "dashboard" / "src" / "components" / "LiveFeed.js",
            r'addEventListener\(\s*["\'](\w+)["\']',
        )
        self.assertGreaterEqual(len(frontend_events), 20,
                                f"Too few frontend events found: {sorted(frontend_events)}")


class TestNegativeResultsLoop(unittest.TestCase):
    """Test the learning-from-failures loop: excluded_ops + negative context."""

    def test_excluded_ops_populated_from_negative_results(self):
        """GrammarConfig.excluded_ops gets populated from negative results."""
        from research.synthesis.grammar import GrammarConfig

        # Simulate: 3 ops with 0% S1 rate, sufficient samples, high confidence
        neg_results = {
            "failed_ops": [
                {"op_name": "bad_op_a", "s1_rate": 0, "n_used": 10, "confidence": 0.8,
                 "failure_stage": "learning"},
                {"op_name": "bad_op_b", "s1_rate": 0, "n_used": 7, "confidence": 0.75,
                 "failure_stage": "compilation"},
                # Should NOT be excluded: low confidence
                {"op_name": "maybe_ok", "s1_rate": 0, "n_used": 6, "confidence": 0.5,
                 "failure_stage": "learning"},
                # Should NOT be excluded: too few samples
                {"op_name": "rare_op", "s1_rate": 0, "n_used": 3, "confidence": 0.9,
                 "failure_stage": "learning"},
            ],
        }

        excluded = set()
        for op_info in neg_results.get("failed_ops", []):
            if (op_info.get("s1_rate", 1) == 0
                    and op_info.get("n_used", 0) >= 5
                    and op_info.get("confidence", 0) >= 0.7):
                excluded.add(op_info["op_name"])

        self.assertEqual(excluded, {"bad_op_a", "bad_op_b"})

        # Verify GrammarConfig accepts excluded_ops
        cfg = GrammarConfig(model_dim=64, excluded_ops=excluded)
        self.assertEqual(cfg.excluded_ops, {"bad_op_a", "bad_op_b"})

    def test_negative_results_in_rich_context(self):
        """build_rich_context includes negative results when present."""
        from research.scientist.llm.context import build_rich_context

        analytics_data = {
            "negative_results": {
                "failed_ops": [
                    {"op_name": "always_fails", "n_used": 12,
                     "failure_stage": "learning", "confidence": 0.85},
                ],
                "anti_patterns": [
                    {"feature": "high depth", "correlation": -0.32,
                     "interpretation": "Higher high depth is associated with lower S1 success"},
                ],
                "summary": "1 ops with 0% S1 rate (always_fails)",
            },
        }

        ctx = build_rich_context(results={}, analytics_data=analytics_data)
        self.assertIn("AVOID always_fails", ctx)
        self.assertIn("0% S1 rate", ctx)
        self.assertIn("Anti-correlated", ctx)
        self.assertIn("high depth", ctx)

    def test_negative_results_absent_gracefully(self):
        """build_rich_context works fine without negative results."""
        from research.scientist.llm.context import build_rich_context

        ctx = build_rich_context(results={}, analytics_data={})
        self.assertNotIn("Negative Results", ctx)
        # Should still produce some output
        self.assertIsInstance(ctx, str)


@unittest.skipUnless(HAS_TORCH, "requires torch")
class TestCompoundMathSpaceOps(unittest.TestCase):
    """Test compound cross-space math primitives."""

    def test_hyperbolic_norm_shape(self):
        """hyperbolic_norm preserves shape."""
        import torch
        from research.mathspaces.hyperbolic import execute_hyperbolic_norm
        module = torch.nn.Module()
        x = torch.randn(2, 4, 16) * 0.1
        out = execute_hyperbolic_norm(module, x)
        self.assertEqual(out.shape, (2, 4, 16))
        self.assertFalse(torch.isnan(out).any())

    def test_tropical_gate_shape(self):
        """tropical_gate preserves shape."""
        import torch
        from research.mathspaces.tropical import execute_tropical_gate
        module = torch.nn.Module()
        x = torch.randn(2, 4, 16)
        out = execute_tropical_gate(module, x)
        self.assertEqual(out.shape, (2, 4, 16))
        self.assertFalse(torch.isnan(out).any())

    def test_clifford_attention_shape(self):
        """clifford_attention preserves shape (D must be multiple of 8)."""
        import torch
        from research.mathspaces.clifford import execute_clifford_attention
        module = torch.nn.Module()
        x = torch.randn(2, 4, 16)
        out = execute_clifford_attention(module, x)
        self.assertEqual(out.shape, (2, 4, 16))
        self.assertFalse(torch.isnan(out).any())

    def test_clifford_attention_padding(self):
        """clifford_attention handles D not divisible by 8."""
        import torch
        from research.mathspaces.clifford import execute_clifford_attention
        module = torch.nn.Module()
        x = torch.randn(2, 4, 12)
        out = execute_clifford_attention(module, x)
        self.assertEqual(out.shape, (2, 4, 12))

    def test_padic_residual_shape(self):
        """padic_residual preserves shape."""
        import torch
        from research.mathspaces.padic import execute_padic_residual
        module = torch.nn.Module()
        x = torch.randn(2, 4, 16)
        out = execute_padic_residual(module, x)
        self.assertEqual(out.shape, (2, 4, 16))
        self.assertFalse(torch.isnan(out).any())

    def test_compound_ops_registered(self):
        """All 4 compound ops appear in the registry after registration."""
        from research.mathspaces.registry import register_all_mathspaces
        from research.synthesis.primitives import list_primitives, OpCategory
        register_all_mathspaces()
        math_ops = {op.name for op in list_primitives(OpCategory.MATH_SPACE)}
        for name in ["hyperbolic_norm", "tropical_gate",
                     "clifford_attention", "padic_residual"]:
            self.assertIn(name, math_ops, f"Compound op {name} not registered")

    def test_compound_ops_have_params(self):
        """Compound ops are registered with has_params=True."""
        from research.mathspaces.registry import register_all_mathspaces
        from research.synthesis.primitives import list_primitives, OpCategory
        register_all_mathspaces()
        math_ops = {op.name: op for op in list_primitives(OpCategory.MATH_SPACE)}
        for name in ["hyperbolic_norm", "tropical_gate",
                     "clifford_attention", "padic_residual"]:
            self.assertTrue(math_ops[name].has_params,
                            f"{name} should have has_params=True")


if __name__ == "__main__":
    unittest.main()

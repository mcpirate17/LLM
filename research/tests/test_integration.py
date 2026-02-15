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
        """Composite score should increase as candidates pass more phases."""
        score_screening = self.nb.compute_composite_score(
            screening_lr=0.5, screening_nov=0.7)
        score_investigation = self.nb.compute_composite_score(
            screening_lr=0.5, screening_nov=0.7,
            inv_lr=0.4, inv_robust=0.6)
        score_validation = self.nb.compute_composite_score(
            screening_lr=0.5, screening_nov=0.7,
            inv_lr=0.4, inv_robust=0.6,
            val_lr=0.3, val_baseline=0.85)

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
        ]
        for f in fields:
            self.assertTrue(hasattr(config, f), f"Missing field: {f}")


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

    def test_api_trends(self):
        r = self.client.get("/api/trends")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIsInstance(data, list)

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

    def test_api_leaderboard(self):
        r = self.client.get("/api/leaderboard")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("entries", data)
        self.assertIn("by_tier", data)
        self.assertIn("total", data)
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

    def test_api_llm_config(self):
        r = self.client.get("/api/llm/config")
        self.assertEqual(r.status_code, 200)

    # ── POST endpoints ──

    def test_api_stop_when_not_running(self):
        r = self.client.post("/api/experiments/stop")
        self.assertEqual(r.status_code, 409)

    def test_api_start_requires_result_ids_for_investigation(self):
        r = self.client.post("/api/experiments/start",
                             json={"mode": "investigation"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("result_ids", r.get_json()["error"])

    def test_api_start_requires_result_ids_for_validation(self):
        r = self.client.post("/api/experiments/start",
                             json={"mode": "validation"})
        self.assertEqual(r.status_code, 400)

    def test_api_start_requires_result_ids_for_scale_up(self):
        r = self.client.post("/api/experiments/start",
                             json={"mode": "scale_up"})
        self.assertEqual(r.status_code, 400)

    def test_api_validate_pipeline(self):
        r = self.client.post("/api/validate", json={"n": 2})
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("generated", data)
        self.assertIn("healthy", data)

    def test_api_404_for_unknown_endpoint(self):
        r = self.client.get("/api/nonexistent")
        self.assertEqual(r.status_code, 404)


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
                {"result_id": "r1", "robustness": 0.7, "best_loss_ratio": 0.4},
                {"result_id": "r2", "robustness": 0.3, "best_loss_ratio": 0.6},
            ]
        }

        self.runner._auto_escalate(results, self.config, nb, phase="investigation")

        pending = getattr(self.runner, "_pending_validation", None)
        self.assertIsNotNone(pending,
                             "Validation should be queued after robust investigation")
        self.assertEqual(len(pending["result_ids"]), 1)  # only r1 qualifies
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

    def test_announce_breakthrough(self):
        msg = self.aria.announce_breakthrough()
        self.assertIsInstance(msg, str)
        self.assertIn("BREAKTHROUGH", msg)

    def test_cost_tracking(self):
        self.aria.reset_cost_tracking()
        self.assertEqual(self.aria.total_tokens, 0)
        self.assertEqual(self.aria.total_cost, 0.0)


# ── Test 10: Dashboard Component Consistency ──


class TestDashboardConsistency(unittest.TestCase):
    """Verify dashboard components and API endpoints are properly wired."""

    @classmethod
    def setUpClass(cls):
        import glob
        cls.component_dir = os.path.join(
            os.path.dirname(__file__), "..", "dashboard", "src", "components")
        cls.component_files = glob.glob(
            os.path.join(cls.component_dir, "*.js"))
        cls.app_js = os.path.join(
            os.path.dirname(__file__), "..", "dashboard", "src", "App.js")

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
            "/api/insights", "/api/entries", "/api/leaderboard",
            "/api/report", "/api/events", "/api/progress",
            "/api/config", "/api/validate",
            "/api/aria/recommendation", "/api/aria/strategy",
            "/api/llm/config",
            "/api/analytics/op-success", "/api/analytics/failure-patterns",
            "/api/analytics/grammar-weights", "/api/analytics/efficiency-frontier",
            "/api/analytics/learning-log",
            "/api/metrics/",
            "/api/experiments/start", "/api/experiments/stop",
            "/api/campaigns", "/api/hypotheses",
            "/api/knowledge",
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


if __name__ == "__main__":
    unittest.main()

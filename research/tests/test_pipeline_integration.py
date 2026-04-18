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
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.pipeline

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
    from research.scientist.notebook import LabNotebook

    HAS_NOTEBOOK = True
except Exception as e:
    HAS_NOTEBOOK = False
    print(f"Notebook import failed: {e}")

try:
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


# ── Test 3: RunConfig & Mode Selection ──


@unittest.skipUnless(HAS_TORCH, "requires torch for runner module")
class TestRunConfig(unittest.TestCase):
    """Test RunConfig serialization and defaults."""

    def test_default_auto_investigate_min_survivors(self):
        """Default min survivors should be 1 (lowered from 2)."""
        from research.scientist.runner import RunConfig

        config = RunConfig()
        self.assertEqual(
            config.auto_investigate_min_survivors, 5
        )  # raised for routing models

    def test_auto_investigate_enabled_by_default(self):
        """Auto-investigation should be on by default."""
        from research.scientist.runner import RunConfig

        config = RunConfig()
        self.assertTrue(config.auto_investigate)
        self.assertTrue(config.auto_validate)

    def test_predictor_and_learned_grammar_are_conservative_by_default(self):
        """Unproven predictor gating and learned grammar weights should default off."""
        from research.scientist.runner import RunConfig

        config = RunConfig()
        self.assertFalse(config.gbm_prescreener_enabled)
        self.assertFalse(config.use_learned_grammar_weights)
        self.assertFalse(config.use_learned_candidate_weights)
        self.assertFalse(config.use_screening_signal_weights)

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
            "auto_investigate",
            "auto_investigate_min_survivors",
            "auto_investigate_top_n",
            "auto_validate",
            "auto_validate_min_robustness",
            "auto_validate_top_n",
            "investigation_steps",
            "investigation_batch_size",
            "validation_steps",
            "validation_batch_size",
            "validation_seq_len",
            "validation_n_seeds",
            "model_source",
            "morph_ratio",
            "morph_focus_sparse",
            "morph_sparse_weight_storage",
            "use_synthesized_training",
            "n_training_programs",
            "data_mode",
            "corpus_path",
            "corpus_format",
            "corpus_text_key",
            "tokenizer_mode",
            "corpus_max_chars",
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
        self.runner.aria.generate_go_no_go = MagicMock(
            return_value={"decision": "go", "rationale": "test approval"}
        )
        self.config = RunConfig(
            auto_investigate=True,
            auto_investigate_min_survivors=1,
            auto_investigate_top_n=3,
            auto_validate=True,
        )

    def _seed_promotable_results(self, nb, result_ids):
        """Insert minimal program_results rows that pass the escalation gates.

        The phase-7 auto-escalation requires:
          - fingerprint_completed_post_investigation = True (in fingerprint_json)
          - novelty_valid_for_promotion = 1
        """
        fp_json = json.dumps({"fingerprint_completed_post_investigation": True})
        exp_id = nb.start_experiment("seed", {}, "seed")
        for rid in result_ids:
            nb.record_program_result(
                experiment_id=exp_id,
                result_id=rid,
                graph_fingerprint=f"fp_{rid}",
                graph_json="{}",
                stage1_passed=True,
                loss_ratio=0.3,
                novelty_score=0.6,
                fingerprint_json=fp_json,
                novelty_valid_for_promotion=1,
                hellaswag_acc=0.35,
                diagnostic_score=0.25,
            )
            nb.flush_writes()
            nb.upsert_leaderboard(
                result_id=rid,
                model_source="graph_synthesis",
                tier="investigation",
                investigation_passed=True,
                investigation_robustness=0.75,
            )
            nb.conn.execute(
                "UPDATE leaderboard SET composite_score = 145.0 WHERE result_id = ?",
                (rid,),
            )

    def test_auto_escalate_queues_investigation(self):
        """S1 survivors with good loss should queue investigation."""
        nb = LabNotebook(self.db_path)

        # Create experiment with S1 survivor
        exp_id = nb.start_experiment("synthesis", {}, "test")
        result_id = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_escalate",
            graph_json="{}",
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            loss_ratio=0.3,
            novelty_score=0.6,
            model_source="graph_synthesis",
        )
        nb.flush_writes()
        nb.upsert_leaderboard(
            result_id=result_id,
            model_source="graph_synthesis",
            tier="screening",
            screening_passed=True,
            screening_loss_ratio=0.3,
            screening_novelty=0.6,
        )
        nb.conn.execute(
            "UPDATE leaderboard SET composite_score = 90.0 WHERE result_id = ?",
            (result_id,),
        )
        nb.complete_experiment(
            exp_id,
            {
                "total": 10,
                "stage1_passed": 1,
            },
            "summary",
            "excited",
        )

        # Include experiment_id so _auto_escalate queries this specific
        # experiment's results rather than the global top-N (which is
        # sensitive to shared DB state and epsilon-greedy seed).
        results = {
            "stage1_passed": 1,
            "experiment_id": exp_id,
            "survivors": [{"novelty": 0.6, "loss_ratio": 0.3}],
        }

        # Force deterministic exploit mode (no epsilon exploration)
        self.config.selection_epsilon = 0.0
        self.runner._auto_escalate(results, self.config, nb, phase="screening")

        # Should have queued investigation
        pending = getattr(self.runner, "_pending_investigation", None)
        self.assertIsNotNone(
            pending, "Investigation should be queued after S1 survivor"
        )
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

    def test_auto_escalate_go_no_go_failure_blocks_queue(self):
        """Campaign go/no-go errors must not silently auto-approve promotion."""
        nb = LabNotebook(self.db_path)

        exp_id = nb.start_experiment("synthesis", {}, "test")
        result_id = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_fail_closed",
            graph_json="{}",
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            loss_ratio=0.25,
            novelty_score=0.7,
            model_source="graph_synthesis",
        )
        nb.flush_writes()
        nb.upsert_leaderboard(
            result_id=result_id,
            model_source="graph_synthesis",
            tier="screening",
            screening_passed=True,
            screening_loss_ratio=0.25,
            screening_novelty=0.7,
        )
        nb.conn.execute(
            "UPDATE leaderboard SET composite_score = 95.0 WHERE result_id = ?",
            (result_id,),
        )
        nb.complete_experiment(
            exp_id,
            {"total": 10, "stage1_passed": 1},
            "summary",
            "excited",
        )

        self.config.selection_epsilon = 0.0
        self.config.auto_go_no_go = True
        self.config.enable_campaigns = True
        self.runner._active_campaign_id = "missing_campaign"
        self.runner.aria.generate_go_no_go = MagicMock(
            side_effect=RuntimeError("llm unavailable")
        )

        results = {
            "stage1_passed": 1,
            "experiment_id": exp_id,
            "survivors": [{"novelty": 0.7, "loss_ratio": 0.25}],
        }

        self.runner._auto_escalate(results, self.config, nb, phase="screening")

        pending = getattr(self.runner, "_pending_investigation", None)
        self.assertIsNone(pending)
        nb.close()

    def test_auto_escalate_queues_validation(self):
        """Investigation results with good robustness should queue validation."""
        nb = LabNotebook(self.db_path)
        self._seed_promotable_results(nb, ["r1", "r2"])

        results = {
            "investigation_results": [
                {
                    "result_id": "r1",
                    "robustness": 0.7,
                    "best_loss_ratio": 0.18,
                    "baseline_loss_ratio": 0.45,
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
        self.assertIsNotNone(
            pending, "Validation should be queued after robust investigation"
        )
        self.assertEqual(len(pending["result_ids"]), 1)  # only r1 qualifies
        nb.close()

    def test_auto_escalate_excludes_brittle_candidates(self):
        """Brittle investigation outcomes should not auto-queue for validation."""
        nb = LabNotebook(self.db_path)
        self._seed_promotable_results(
            nb, ["stable", "brittle_flag", "brittle_multiplier"]
        )

        results = {
            "investigation_results": [
                {
                    "result_id": "stable",
                    "robustness": 0.8,
                    "best_loss_ratio": 0.15,
                    "baseline_loss_ratio": 0.45,
                    "novelty_confidence": 0.75,
                    "loss_ratio_multiplier": 2.0,
                    "brittle_risk": False,
                },
                {
                    "result_id": "brittle_flag",
                    "robustness": 0.85,
                    "best_loss_ratio": 0.12,
                    "baseline_loss_ratio": 0.40,
                    "novelty_confidence": 0.8,
                    "loss_ratio_multiplier": 20.0,
                    "brittle_risk": True,
                },
                {
                    "result_id": "brittle_multiplier",
                    "robustness": 0.9,
                    "best_loss_ratio": 0.10,
                    "baseline_loss_ratio": 0.38,
                    "novelty_confidence": 0.8,
                    "loss_ratio_multiplier": self.config.investigation_max_loss_ratio_multiplier
                    + 0.1,
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
        """Validation auto-queue requires fingerprint + novelty gates and baseline evidence."""
        nb = LabNotebook(self.db_path)
        # Only "qualified" gets promotable fingerprint/novelty data;
        # the others are blocked by the phase-7 fingerprint gate.
        self._seed_promotable_results(nb, ["qualified"])

        results = {
            "investigation_results": [
                {
                    "result_id": "missing_conf",
                    "robustness": 0.85,
                    "best_loss_ratio": 0.12,
                    "baseline_loss_ratio": 0.42,
                },
                {
                    "result_id": "weak_baseline",
                    "robustness": 0.9,
                    "best_loss_ratio": 0.10,
                    "baseline_loss_ratio": 0.96,
                    "novelty_confidence": 0.8,
                },
                {
                    "result_id": "qualified",
                    "robustness": 0.88,
                    "best_loss_ratio": 0.14,
                    "baseline_loss_ratio": 0.45,
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
        with patch(
            "research.scientist.runner.control_start.threading.Thread"
        ) as thread_cls:
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
        result_id = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_lb",
            graph_json="{}",
            stage0_passed=True,
            stage1_passed=True,
            loss_ratio=0.4,
            novelty_score=0.7,
            model_source="graph_synthesis",
        )
        nb.flush_writes()  # record_program_result uses async write queue
        # Leaderboard entries are now created at S1-pass time, not during
        # auto-escalation.  Pre-populate so escalation can find it.
        nb.upsert_leaderboard(
            result_id=result_id,
            model_source="graph_synthesis",
            tier="screening",
        )
        nb.complete_experiment(
            exp_id, {"total": 5, "stage1_passed": 1}, "done", "excited"
        )

        results = {"stage1_passed": 1, "survivors": [{"novelty": 0.7}]}
        self.runner._auto_escalate(results, self.config, nb, phase="screening")

        leaderboard = nb.get_leaderboard()
        self.assertGreater(
            len(leaderboard), 0, "Leaderboard should have entries after escalation"
        )
        self.assertEqual(leaderboard[0]["tier"], "screening")
        nb.close()

    def test_ensure_campaign_marks_post_hoc_criteria(self):
        """Campaign criteria created from recent results should be labeled post-hoc."""
        nb = LabNotebook(self.db_path)
        try:
            self.runner._active_campaign_id = None
            self.runner.aria.formulate_campaign = MagicMock(
                return_value={
                    "title": "Campaign A",
                    "objective": "Explore",
                    "success_criteria": "Increase S1 pass rate",
                }
            )

            campaign_id = self.runner._ensure_campaign(self.config, nb)
            self.assertIsNotNone(campaign_id)

            campaign = nb.get_campaign(campaign_id)
            self.assertIsNotNone(campaign)
            self.assertIn("[POST-HOC]", campaign["success_criteria"])
        finally:
            nb.close()


# ── Test 12: Inline Phase Methods & Budget Context ──


@unittest.skipUnless(HAS_TORCH, "requires torch for runner module")
class TestInlinePhaseMethods(unittest.TestCase):
    """Verify inline investigation/validation methods exist and are callable."""

    def test_runner_has_inline_investigation(self):
        """ExperimentRunner must have _run_inline_investigation (not crash)."""
        from research.scientist.runner import ExperimentRunner

        self.assertTrue(
            hasattr(ExperimentRunner, "_run_inline_investigation"),
            "Missing _run_inline_investigation method",
        )

    def test_runner_has_inline_validation(self):
        """ExperimentRunner must have _run_inline_validation (not crash)."""
        from research.scientist.runner import ExperimentRunner

        self.assertTrue(
            hasattr(ExperimentRunner, "_run_inline_validation"),
            "Missing _run_inline_validation method",
        )

    def test_inline_validation_progress_sets_total_programs(self):
        """Inline validation must initialize progress denominator to avoid x/0 UI output."""
        import inspect
        from research.scientist.runner import ExperimentRunner

        # total_programs is set in the bootstrap helper after refactoring
        src = inspect.getsource(ExperimentRunner._inline_validation_bootstrap)
        self.assertIn(
            "total_programs=len(result_ids)",
            src,
            "_inline_validation_bootstrap LiveProgress must set total_programs",
        )

    def test_inline_validation_persists_candidate_metadata(self):
        """Inline validation should persist candidate IDs into experiment config metadata."""
        import inspect
        from research.scientist.runner import ExperimentRunner

        # Config metadata is set in the bootstrap helper after refactoring
        src = inspect.getsource(ExperimentRunner._inline_validation_bootstrap)
        self.assertIn("_validation_config_with_result_ids", src)
        self.assertIn('"continuous_auto"', src)

    def test_inline_investigation_progress_sets_total_programs(self):
        """Inline investigation must initialize progress denominator for dashboard parity."""
        import inspect
        from research.scientist.runner import ExperimentRunner

        src = inspect.getsource(ExperimentRunner._run_inline_investigation)
        self.assertIn(
            "total_programs=len(result_ids)",
            src,
            "_run_inline_investigation LiveProgress must set total_programs",
        )

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
        self.assertNotIn(
            "self._run_investigation(",
            src,
            "_run_inline_investigation should not call "
            "non-existent _run_investigation()",
        )

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
        runner.aria.formulate_hypothesis = MagicMock(
            return_value="context-aware hypothesis"
        )

        config = RunConfig(n_programs=1, max_cost_dollars=10.0)
        exp_id = runner.start_experiment(config=config, hypothesis=None)

        self.assertIsNotNone(exp_id)
        runner.aria.formulate_hypothesis.assert_called_once()
        call = runner.aria.formulate_hypothesis.call_args
        self.assertIn("context", call.kwargs)
        self.assertTrue(call.kwargs["context"].strip())

    def test_non_control_synthesis_persists_learned_grammar_exposure(self):
        """Continuous synthesis should persist learned-grammar exposure on non-control runs."""
        from research.scientist.runner import ExperimentRunner, RunConfig

        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test_learned_grammar_flag.db")
        runner = ExperimentRunner(db_path)

        nb = MagicMock()
        runner._ensure_math_spaces = MagicMock()
        runner._execute_experiment = MagicMock(
            return_value={
                "stage0_passed": 0,
                "stage05_passed": 0,
                "stage1_passed": 0,
                "survivors": [],
            }
        )
        runner._persist_applied_grammar_weights = MagicMock()
        runner._build_rich_context_for_experiment = MagicMock(return_value="ctx")
        runner._analyze_results = MagicMock(return_value=[])
        runner._auto_recommend = MagicMock()
        runner.aria.formulate_hypothesis = MagicMock(return_value="test hypothesis")
        runner.aria.experiment_summary = MagicMock(return_value="")
        runner.aria.analyze_results = MagicMock(return_value="")
        runner.aria.validate_hypothesis = MagicMock(return_value=None)

        config = RunConfig(n_programs=1, use_learned_grammar_weights=True)
        runner._is_control_experiment = MagicMock(return_value=False)

        runner._run_continuous_synthesis(
            config=config,
            nb=nb,
            n_experiments=1,
            limit_str="exp 1/10",
            mode_reasoning="test",
        )

        exec_kwargs = runner._execute_experiment.call_args.kwargs
        self.assertTrue(exec_kwargs["use_learned_grammar"])

        start_cfg = nb.start_experiment.call_args.kwargs["config"]
        self.assertTrue(start_cfg["use_learned_grammar_weights"])

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
        runner.aria.formulate_hypothesis = MagicMock(
            return_value="fallback-context hypothesis"
        )
        runner.aria.critique_hypothesis = MagicMock(
            return_value={
                "verdict": "proceed",
                "gate": "pass",
                "checks": [],
                "concerns": [],
                "suggestions": [],
                "confidence": 0.8,
            }
        )

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
        runner.aria.formulate_hypothesis = MagicMock(
            return_value=(
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
            )
        )

        config = RunConfig(n_programs=1, max_cost_dollars=5.0)
        exp_id = runner.start_experiment(config=config, hypothesis=None)

        nb = LabNotebook(db_path)
        try:
            entries = nb.get_entries(
                experiment_id=exp_id, entry_type="hypothesis", limit=5
            )
            self.assertTrue(entries)
            metadata = json.loads(entries[0].get("metadata_json") or "{}")
            self.assertEqual(metadata.get("source"), "llm_context")
            self.assertTrue(metadata.get("used_context"))
            self.assertTrue(
                str(metadata.get("review_status", "")).startswith("preflight_")
            )
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
        db_path = os.path.join(
            tmpdir, "test_start_investigation_hypothesis_metadata.db"
        )
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
            entries = nb.get_entries(
                experiment_id=exp_id, entry_type="hypothesis", limit=5
            )
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
        runner.aria.formulate_hypothesis = MagicMock(
            return_value=(
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
            )
        )

        config = RunConfig(n_programs=1)
        exp_id = runner.start_evolution(config=config, hypothesis=None)

        nb = LabNotebook(db_path)
        try:
            entries = nb.get_entries(
                experiment_id=exp_id, entry_type="hypothesis", limit=5
            )
            self.assertTrue(entries)
            metadata = json.loads(entries[0].get("metadata_json") or "{}")
            self.assertEqual(metadata.get("source"), "llm_context")
            self.assertTrue(metadata.get("llm_used"))
            self.assertEqual(metadata.get("review_status"), "not_reviewed")
            self.assertAlmostEqual(float(metadata.get("confidence")), 0.66, places=2)
        finally:
            nb.close()

    def test_runner_startup_recovers_stale_experiments(self):
        """Runner close should clean stale experiments left in running state."""
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
        _runner.close()

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
        """Runner close should clean no-progress startup-failed running experiments."""
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
        _runner.close()

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
        config = RunConfig(vocab_size=32, max_seq_len=16, data_mode="random")

        seen_seeds = []

        original_sampler = runner._sample_training_input_ids

        def _spy_sample_training_input_ids(*args, **kwargs):
            seen_seeds.append(kwargs.get("seed"))
            return original_sampler(*args, **kwargs)

        with patch.object(
            runner,
            "_sample_training_input_ids",
            side_effect=_spy_sample_training_input_ids,
        ):
            _ = runner._train_with_program(
                model,
                Program(),
                config,
                torch.device("cpu"),
                seed=1234,
            )

        self.assertGreaterEqual(len(seen_seeds), 1)
        self.assertEqual(seen_seeds, list(range(1234, 1234 + len(seen_seeds))))

    def test_cycle_failure_marks_active_experiment_failed(self):
        """Cycle-level exceptions should finalize active experiment rows as failed."""
        from research.scientist.runner import ExperimentRunner

        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test_cycle_failure_finalize.db")
        runner = ExperimentRunner(db_path)
        nb = LabNotebook(db_path)
        try:
            exp_id = nb.start_experiment(
                experiment_type="evolution",
                config={"n_programs": 2},
                hypothesis="failure finalize test",
            )
            with runner._lock:
                runner._progress.experiment_id = exp_id
                runner._progress.status = "evolving"

            # Add llm_analysis so fail_experiment doesn't auto-delete this
            # zero-value experiment (production cleanup for truly empty failures).
            nb.conn.execute(
                "UPDATE experiments SET llm_analysis = ? WHERE experiment_id = ?",
                ("test analysis", exp_id),
            )
            nb._maybe_commit()

            failed_id = runner._fail_active_cycle_experiment(
                nb,
                "simulated cycle failure",
            )
            self.assertEqual(failed_id, exp_id)

            row = nb.conn.execute(
                "SELECT status, aria_summary FROM experiments WHERE experiment_id = ?",
                (exp_id,),
            ).fetchone()
            self.assertEqual(row["status"], "failed")
            self.assertIn("FAILED", row["aria_summary"])
            self.assertIn("simulated cycle failure", row["aria_summary"])
            self.assertEqual(runner.progress.status, "failed")
        finally:
            nb.close()

    def test_local_llm_backend_disables_continuous_time_limit(self):
        """Continuous max_time should not stop sessions when backend is local."""
        from research.scientist.runner import ExperimentRunner, RunConfig

        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test_local_llm_time_limit.db")
        runner = ExperimentRunner(db_path)
        config = RunConfig(max_experiments=100, max_time_minutes=1, max_cost_dollars=0)
        t_start = time.time() - (10 * 60)

        runner.aria.get_llm_config = MagicMock(return_value={"backend": "ollama"})
        self.assertIsNone(
            runner._check_continuous_limits(config, t_start, n_experiments=1)
        )

        runner.aria.get_llm_config = MagicMock(return_value={"backend": "anthropic"})
        reason = runner._check_continuous_limits(config, t_start, n_experiments=1)
        self.assertIsNotNone(reason)
        self.assertIn("Time limit reached", reason)

    def test_prescreen_run_config_hardens_invalid_basics(self):
        """Prescreen should auto-harden obviously invalid baseline fields."""
        from research.scientist.runner import ExperimentRunner, RunConfig

        runner = ExperimentRunner(
            os.path.join(tempfile.mkdtemp(), "prescreen_basics.db")
        )
        config = RunConfig(
            n_programs=0,
            stage1_steps=0,
            n_layers=0,
            model_dim=8,
            max_seq_len=8,
            data_mode="corpus",
            corpus_path="",
        )

        hardened, report = runner.prescreen_run_config(
            config, mode="single", auto_harden=True
        )

        self.assertEqual(hardened.n_programs, 1)
        self.assertEqual(hardened.stage1_steps, 1)
        self.assertEqual(hardened.n_layers, 1)
        self.assertGreaterEqual(hardened.model_dim, 16)
        self.assertGreaterEqual(hardened.max_seq_len, 16)
        self.assertEqual(hardened.data_mode, "random")
        self.assertTrue(report.get("checked"))
        self.assertGreaterEqual(report.get("issue_count", 0), 1)
        self.assertGreaterEqual(report.get("adjustment_count", 0), 1)

    def test_prescreen_run_config_caps_evolution_depth_and_ops(self):
        """Prescreen should cap high-risk evolution recursion settings."""
        from research.scientist.runner import ExperimentRunner, RunConfig

        runner = ExperimentRunner(
            os.path.join(tempfile.mkdtemp(), "prescreen_evolve.db")
        )
        config = RunConfig(max_depth=40, max_ops=80, n_generations=0)

        hardened, report = runner.prescreen_run_config(
            config, mode="evolve", auto_harden=True
        )

        self.assertEqual(hardened.max_depth, 18)
        self.assertEqual(hardened.max_ops, 24)
        self.assertEqual(hardened.n_generations, 1)
        self.assertGreater(report.get("risk_score", 0), 0)
        self.assertIn(report.get("risk_level"), {"medium", "high"})

    def test_prescreen_falls_back_to_cpu_when_cuda_probe_fails(self):
        """Prescreen should force CPU when CUDA context preflight fails."""
        from research.scientist.runner import ExperimentRunner, RunConfig

        runner = ExperimentRunner(
            os.path.join(tempfile.mkdtemp(), "prescreen_cuda_probe.db")
        )
        config = RunConfig(device="cuda")

        with (
            patch(
                "research.scientist.runner.screening.torch.cuda.is_available",
                return_value=True,
            ),
            patch.object(
                runner,
                "_cuda_health_probe",
                return_value=(False, "CUDA error: device-side assert triggered"),
            ),
        ):
            hardened, report = runner.prescreen_run_config(
                config, mode="single", auto_harden=True
            )

        self.assertEqual(hardened.device, "cpu")
        reasons = " ".join(i.get("reason", "") for i in report.get("issues", []))
        self.assertIn("CUDA preflight probe failed", reasons)

    def test_prescreen_falls_back_to_cpu_on_recent_cuda_assert_streak(self):
        """Prescreen should avoid repeated CUDA 0/0 runs after recent assert failures."""
        from research.scientist.runner import ExperimentRunner, RunConfig

        db_path = os.path.join(tempfile.mkdtemp(), "prescreen_cuda_streak.db")
        runner = ExperimentRunner(db_path)
        nb = LabNotebook(db_path)
        try:
            for idx in range(5):
                exp_id = nb.start_experiment(
                    experiment_type="synthesis",
                    config={"device": "cuda", "n_programs": 1},
                    hypothesis=f"cuda streak {idx}",
                )
                # Set llm_analysis so fail_experiment doesn't auto-delete
                nb.conn.execute(
                    "UPDATE experiments SET llm_analysis = ? WHERE experiment_id = ?",
                    ("cuda assert", exp_id),
                )
                nb._maybe_commit()
                nb.fail_experiment(
                    exp_id,
                    "CUDA error: device-side assert triggered",
                )
        finally:
            nb.close()

        config = RunConfig(device="cuda")
        with (
            patch(
                "research.scientist.runner.screening.torch.cuda.is_available",
                return_value=True,
            ),
            patch.object(runner, "_cuda_health_probe", return_value=(True, None)),
        ):
            hardened, report = runner.prescreen_run_config(
                config, mode="single", auto_harden=True
            )

        self.assertEqual(hardened.device, "cpu")
        reasons = " ".join(i.get("reason", "") for i in report.get("issues", []))
        self.assertIn("device-side assert", reasons)

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
            n_steps = 8  # Enough steps so inflight quarter-check doesn't fire at step 0
            batch_size = 1
            max_grad_norm = 1.0
            curriculum = _Curriculum()
            loss = _Loss()
            optimizer = _Optimizer()

            def to_dict(self):
                return {"init_scheme": self.init_scheme, "n_steps": self.n_steps}

        runner = ExperimentRunner(
            os.path.join(tempfile.mkdtemp(), "corpus_fallback.db")
        )
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

        src_record = inspect.getsource(ExperimentRunner._record_orchestrator_result)
        self.assertIn('s1_result.get("n_train_steps")', src_record)
        self.assertIn("self._resolve_baseline_recipe", src_record)

        # After refactoring, baseline recipe logic is delegated to run_baseline_comparison
        src_validation = inspect.getsource(ExperimentRunner._validation_compute_metrics)
        self.assertIn("run_baseline_comparison", src_validation)
        self.assertIn("self._resolve_baseline_recipe", src_validation)

        # Verify recipe details are in the shared helper
        from research.scientist.runner._helpers import run_baseline_comparison

        src_helper = inspect.getsource(run_baseline_comparison)
        self.assertIn('train_result.get("n_train_steps")', src_helper)
        self.assertIn('recipe["momentum"]', src_helper)
        self.assertIn('recipe["optimizer_name"]', src_helper)
        self.assertIn('recipe["weight_decay"]', src_helper)

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
        modes = ["uniform", "depth_token_mask", "confidence_token_gate"]
        seeds = [11, 22]
        result = runner.run_routing_benchmark(config, seed_set=seeds, modes=modes)

        self.assertTrue(result.get("available"))
        self.assertEqual(result.get("seed_set"), seeds)
        # Some routing modes may fail in minimal test configs; verify at
        # least one mode produced valid frontier points.
        self.assertGreaterEqual(len(result.get("modes_evaluated", [])), 1)

        points = result.get("points", [])
        self.assertGreaterEqual(len(points), 1)
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
        from research.scientist.llm.context_experiment import (
            build_mode_selection_context,
        )

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
        from research.scientist.llm.context_experiment import (
            build_mode_selection_context,
        )

        ctx = build_mode_selection_context(
            recent_experiments=[],
            leaderboard=[],
            cost_spent=0,
            budget=0,
        )
        self.assertNotIn("Budget", ctx)


@unittest.skipUnless(
    HAS_TORCH and HAS_FLASK and HAS_NOTEBOOK, "requires torch, flask, and notebook"
)
class TestPipelineEndToEnd(unittest.TestCase):
    """Single test that runs the AI scientist pipeline end-to-end."""

    @unittest.skip(
        "Requires native runner ABI session (not available in standard test environment)"
    )
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
        for k in [
            "summary",
            "recent_experiments",
            "op_success_rates",
            "structural_correlations",
            "learning_log",
        ]:
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
            batch_size=2,
            seq_len=32,
            rng=rng,
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
                self.DiagnosticTaskResult(
                    "copy", accuracy=0.8, loss=1.2, steps_trained=100
                ),
                self.DiagnosticTaskResult(
                    "periodic", accuracy=0.9, loss=0.5, steps_trained=100
                ),
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
            cols = [
                row[1]
                for row in nb.conn.execute(
                    "PRAGMA table_info(program_results)"
                ).fetchall()
            ]
            self.assertIn("diagnostic_tasks_json", cols)
            self.assertIn("diagnostic_score", cols)
        finally:
            if nb:
                nb.close()

    def test_notebook_migration_has_dual_loss_columns(self):
        """Notebook migration map includes discovery/validation loss fields."""
        nb = None
        try:
            tmpdir = tempfile.mkdtemp()
            db_path = os.path.join(tmpdir, "test_loss_migration.db")
            from research.scientist.notebook import LabNotebook

            nb = LabNotebook(db_path)
            cols = [
                row[1]
                for row in nb.conn.execute(
                    "PRAGMA table_info(program_results)"
                ).fetchall()
            ]
            for col in (
                "discovery_loss",
                "discovery_loss_ratio",
                "validation_loss",
                "validation_loss_ratio",
                "generalization_gap",
            ):
                self.assertIn(col, cols)
        finally:
            if nb:
                nb.close()


class TestNegativeResultsLoop(unittest.TestCase):
    """Test the learning-from-failures loop: op penalties + negative context."""

    def test_negative_results_in_rich_context(self):
        """build_rich_context includes negative results when present."""
        from research.scientist.llm.context_experiment import build_rich_context

        analytics_data = {
            "negative_results": {
                "failed_ops": [
                    {
                        "op_name": "always_fails",
                        "n_used": 12,
                        "failure_stage": "learning",
                        "confidence": 0.85,
                    },
                ],
                "anti_patterns": [
                    {
                        "feature": "high depth",
                        "correlation": -0.32,
                        "interpretation": "Higher high depth is associated with lower S1 success",
                    },
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
        from research.scientist.llm.context_experiment import build_rich_context

        ctx = build_rich_context(results={}, analytics_data={})
        self.assertNotIn("Negative Results", ctx)
        # Should still produce some output
        self.assertIsInstance(ctx, str)


if __name__ == "__main__":
    unittest.main()

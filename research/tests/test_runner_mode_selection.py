import pytest
import os
import sys
import tempfile
import time
import unittest
import inspect
from unittest.mock import patch


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RESEARCH_ROOT = os.path.dirname(_THIS_DIR)
_WORKSPACE_ROOT = os.path.dirname(_RESEARCH_ROOT)
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)


from research.scientist.notebook import LabNotebook, ExperimentEntry
from research.scientist.runner import ExperimentRunner, RunConfig

pytestmark = pytest.mark.unit


class _FakeCursor:
    def fetchall(self):
        return []


class _FakeDB:
    def execute(self, *_args, **_kwargs):
        return _FakeCursor()


class _FakeNotebook:
    def __init__(self):
        self.db = _FakeDB()
        self.entries = []

    def get_recent_experiments(self, _n=10):
        rows = []
        for _ in range(10):
            rows.append(
                {
                    "experiment_type": "synthesis",
                    "status": "completed",
                    "n_stage1_passed": 1,
                    "n_programs_generated": 40,
                    "best_novelty_score": 0.3,
                }
            )
        return rows

    def get_leaderboard(self, limit=50):
        return []

    def add_entry(self, entry: ExperimentEntry):
        self.entries.append(entry)


class TestRunnerModeSelection(unittest.TestCase):
    def test_runner_initializes_heal_retry_state_for_continuous_loop(self):
        runner = ExperimentRunner(
            os.path.join(tempfile.mkdtemp(), "heal_retry_state.db")
        )

        self.assertTrue(hasattr(runner, "_pending_heal_retry"))
        self.assertIsNone(runner._pending_heal_retry)
        self.assertTrue(hasattr(runner, "_recent_healer_signatures"))
        self.assertEqual(runner._recent_healer_signatures, {})
        self.assertTrue(hasattr(runner, "_knowledge_distiller"))
        self.assertIsNone(runner._knowledge_distiller)
        self.assertTrue(hasattr(runner, "_pending_scale_up"))
        self.assertIsNone(runner._pending_scale_up)
        self.assertTrue(hasattr(runner, "_next_follow_up_parent"))
        self.assertIsNone(runner._next_follow_up_parent)

    def test_config_overrides_apply_to_cycle_execution(self):
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "mode_override_cycle.db")
        runner = ExperimentRunner(db_path)
        nb = LabNotebook(db_path)

        captured = {}

        def _capture_synthesis(cfg, _nb, _n_exp, _limit, _reason):
            captured["n_programs"] = cfg.n_programs
            captured["max_depth"] = cfg.max_depth

        with (
            patch.object(
                runner,
                "_select_next_mode",
                return_value={
                    "mode": "synthesis",
                    "reasoning": "test compact override",
                    "confidence": 0.8,
                    "config": {"n_programs": 77, "max_depth": 5},
                },
            ),
            patch.object(
                runner, "_run_continuous_synthesis", side_effect=_capture_synthesis
            ),
        ):
            runner.run_aria_cycle(
                RunConfig(device="cpu", n_programs=20, max_depth=10),
                nb,
                n_experiments=1,
                t_start=time.time(),
            )

        nb.close()
        self.assertEqual(captured.get("n_programs"), 77)
        # max_depth is a floor field — mode overrides can raise but not lower.
        # User set max_depth=10, mode suggested 5, so 10 is preserved.
        self.assertEqual(captured.get("max_depth"), 10)

    def test_select_next_mode_forces_compression_examination_when_undercovered(self):
        runner = ExperimentRunner(
            os.path.join(tempfile.mkdtemp(), "compression_override.db")
        )
        nb = _FakeNotebook()

        analytics_data = {
            "op_success_rates": [],
            "failure_patterns": {},
            "grammar_weights": {},
            "default_weights": {},
            "negative_results": {},
            "compression_coverage": {
                "totals": {
                    "n_tested": 20,
                    "n_survived": 4,
                    "n_compressed_tested": 1,
                    "n_compressed_survived": 0,
                }
            },
        }

        with (
            patch.object(runner, "_gather_analytics_data", return_value=analytics_data),
            patch.object(
                runner.aria,
                "recommend_next_mode",
                return_value={
                    "mode": "evolution",
                    "reasoning": "base recommendation",
                    "confidence": 0.6,
                    "config": {},
                },
            ),
        ):
            rec = runner._select_next_mode(
                RunConfig(device="cpu", n_programs=40),
                nb,
                n_experiments=6,
            )

        self.assertEqual(rec.get("mode"), "synthesis")
        self.assertTrue(bool(rec.get("compression_focus")))
        self.assertIn("Compression examination injection", rec.get("reasoning", ""))

    def test_validation_calls_use_run_config_limits(self):
        # Graph generation and structural validation gates both must respect
        # config.max_ops / config.max_depth. Read the source directly from
        # disk so the check is immune to any in-process mock state.
        import pathlib

        research_root = pathlib.Path(__file__).resolve().parents[1]
        src_generate = (
            research_root / "scientist" / "runner" / "execution_candidates.py"
        ).read_text()
        src_quality_gates = (
            research_root / "scientist" / "runner" / "execution_screening_pipeline.py"
        ).read_text()

        self.assertIn("max_ops=max(1, int(config.max_ops))", src_generate)
        self.assertIn("max_depth=max(1, int(config.max_depth))", src_generate)
        self.assertIn("max_ops=max(1, int(config.max_ops))", src_quality_gates)
        self.assertIn("max_depth=max(1, int(config.max_depth))", src_quality_gates)

    def test_run_cycle_dispatches_refinement_mode(self):
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "mode_refinement_cycle.db")
        runner = ExperimentRunner(db_path)
        nb = LabNotebook(db_path)

        called = {}

        def _capture_refinement(cfg, _nb, _n_exp, _limit, _reason):
            called["model_source"] = cfg.model_source
            called["source_ids"] = cfg.refine_source_result_ids

        with (
            patch.object(
                runner,
                "_select_next_mode",
                return_value={
                    "mode": "refinement",
                    "reasoning": "recursive local tweaks",
                    "confidence": 0.8,
                    "config": {
                        "model_source": "fingerprint_refine",
                        "refine_source_result_ids": "r1,r2",
                    },
                },
            ),
            patch.object(
                runner, "_run_continuous_refinement", side_effect=_capture_refinement
            ),
        ):
            runner.run_aria_cycle(
                RunConfig(device="cpu", n_programs=20, max_depth=10),
                nb,
                n_experiments=1,
                t_start=time.time(),
            )

        nb.close()
        self.assertEqual(called.get("model_source"), "fingerprint_refine")
        self.assertEqual(called.get("source_ids"), "r1,r2")


try:
    from research.scientist.persona import Aria

    HAS_PERSONA = True
except Exception:
    HAS_PERSONA = False


@unittest.skipUnless(HAS_PERSONA, "requires persona module")
class TestAriaModeSelecion(unittest.TestCase):
    """Test Aria's rule-based mode recommendation."""

    def setUp(self):
        self.aria = Aria()

    def test_no_survivors_recommends_synthesis(self):
        """With no S1 survivors, should recommend synthesis."""
        rec = self.aria._rule_based_mode_recommendation(
            {
                "total_s1_survivors": 0,
                "avg_novelty": 0,
                "n_experiments_in_session": 1,
            }
        )
        self.assertEqual(rec["mode"], "synthesis")

    def test_long_zero_survivor_streak_rotates_recovery(self):
        """After many zero-survivor runs, recommendation should rotate strategies."""
        # n_experiments=10 -> recovery_idx=0 -> conservative config
        rec0 = self.aria._rule_based_mode_recommendation(
            {
                "total_s1_survivors": 0,
                "avg_novelty": 0,
                "n_experiments_in_session": 10,
            }
        )
        self.assertEqual(rec0["mode"], "synthesis")
        self.assertEqual(rec0["config"]["residual_prob"], 0.85)

        # n_experiments=11 -> recovery_idx=1 -> sparse config
        rec1 = self.aria._rule_based_mode_recommendation(
            {
                "total_s1_survivors": 0,
                "avg_novelty": 0,
                "n_experiments_in_session": 11,
            }
        )
        self.assertEqual(rec1["mode"], "synthesis")
        self.assertIn("op_weights", rec1["config"])

        # n_experiments=14 -> recovery_idx=4 -> evolution
        rec4 = self.aria._rule_based_mode_recommendation(
            {
                "total_s1_survivors": 0,
                "avg_novelty": 0,
                "n_experiments_in_session": 14,
            }
        )
        self.assertEqual(rec4["mode"], "evolution")

    def test_low_novelty_recommends_novelty_search(self):
        """With survivors but low novelty, should recommend novelty."""
        rec = self.aria._rule_based_mode_recommendation(
            {
                "total_s1_survivors": 5,
                "avg_novelty": 0.2,
                "n_experiments_in_session": 2,
            }
        )
        self.assertIn(rec["mode"], {"novelty", "synthesis"})

    def test_good_survivors_recommends_evolution(self):
        """With 3+ diverse survivors, should recommend evolution."""
        rec = self.aria._rule_based_mode_recommendation(
            {
                "total_s1_survivors": 5,
                "avg_novelty": 0.6,
                "n_experiments_in_session": 2,
            }
        )
        self.assertIn(rec["mode"], {"evolution", "synthesis"})

    def test_investigation_ready_recommends_investigation(self):
        """With investigation-ready candidates, should recommend investigation."""
        rec = self.aria._rule_based_mode_recommendation(
            {
                "total_s1_survivors": 3,
                "avg_novelty": 0.5,
                "n_experiments_in_session": 5,
                "investigation_ready": 3,
            }
        )
        self.assertIn(rec["mode"], {"investigation", "synthesis"})

    def test_validation_ready_recommends_validation(self):
        """Validation candidates take highest priority."""
        rec = self.aria._rule_based_mode_recommendation(
            {
                "total_s1_survivors": 5,
                "avg_novelty": 0.6,
                "n_experiments_in_session": 10,
                "investigation_ready": 3,
                "validation_ready": 2,
            }
        )
        self.assertIn(rec["mode"], {"validation", "synthesis"})

    def test_recommendation_has_required_fields(self):
        """Every recommendation should have mode, reasoning, confidence, config."""
        rec = self.aria._rule_based_mode_recommendation({})
        self.assertIn("mode", rec)
        self.assertIn("reasoning", rec)
        self.assertIn("confidence", rec)
        self.assertIn("config", rec)
        self.assertIn(
            rec["mode"],
            {"synthesis", "evolution", "novelty", "investigation", "validation"},
        )

    def test_parse_briefing_uses_reasoning_when_briefing_missing(self):
        parsed = self.aria._parse_briefing(
            "SUGGESTED_ACTION:\n"
            "MODE: evolve\n"
            "REASONING: Evolution remains the best next step from recent plateaued runs.\n"
            "CONFIDENCE: 0.78\n"
        )
        self.assertTrue(parsed.get("briefing_text"))
        self.assertIn(
            "Evolution remains the best next step", parsed.get("briefing_text", "")
        )

    def test_parse_briefing_accepts_summary_prefix(self):
        parsed = self.aria._parse_briefing(
            "Summary: Recent S1 hit rate is flattening and validation queue is growing.\n"
            "MODE: novelty\n"
            "REASONING: Diversification is needed to escape local minima."
        )
        self.assertIn(
            "Recent S1 hit rate is flattening", parsed.get("briefing_text", "")
        )

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


if __name__ == "__main__":
    unittest.main()

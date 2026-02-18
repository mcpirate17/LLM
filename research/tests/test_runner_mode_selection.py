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
            rows.append({
                "experiment_type": "synthesis",
                "status": "completed",
                "n_stage1_passed": 1,
                "n_programs_generated": 40,
                "best_novelty_score": 0.3,
            })
        return rows

    def get_leaderboard(self, limit=50):
        return []

    def add_entry(self, entry: ExperimentEntry):
        self.entries.append(entry)


class TestRunnerModeSelection(unittest.TestCase):
    def test_config_overrides_apply_to_cycle_execution(self):
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "mode_override_cycle.db")
        runner = ExperimentRunner(db_path)
        nb = LabNotebook(db_path)

        captured = {}

        def _capture_synthesis(cfg, _nb, _n_exp, _limit, _reason):
            captured["n_programs"] = cfg.n_programs
            captured["max_depth"] = cfg.max_depth

        with patch.object(runner, "_select_next_mode", return_value={
            "mode": "synthesis",
            "reasoning": "test compact override",
            "confidence": 0.8,
            "config": {"n_programs": 77, "max_depth": 5},
        }), patch.object(runner, "_run_continuous_synthesis", side_effect=_capture_synthesis):
            runner.run_aria_cycle(
                RunConfig(device="cpu", n_programs=20, max_depth=10),
                nb,
                n_experiments=1,
                t_start=time.time(),
            )

        nb.close()
        self.assertEqual(captured.get("n_programs"), 77)
        self.assertEqual(captured.get("max_depth"), 5)

    def test_select_next_mode_forces_compression_examination_when_undercovered(self):
        runner = ExperimentRunner(os.path.join(tempfile.mkdtemp(), "compression_override.db"))
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

        with patch.object(runner, "_gather_analytics_data", return_value=analytics_data), \
                patch.object(runner.aria, "recommend_next_mode", return_value={
                    "mode": "evolution",
                    "reasoning": "base recommendation",
                    "confidence": 0.6,
                    "config": {},
                }):
            rec = runner._select_next_mode(
                RunConfig(device="cpu", n_programs=40),
                nb,
                n_experiments=6,
            )

        self.assertEqual(rec.get("mode"), "synthesis")
        self.assertTrue(bool(rec.get("compression_focus")))
        self.assertIn("Compression examination injection", rec.get("reasoning", ""))

    def test_validation_calls_use_run_config_limits(self):
        src_execute = inspect.getsource(ExperimentRunner._execute_experiment)
        src_generate = inspect.getsource(ExperimentRunner._generate_candidates)

        self.assertIn("max_ops=max(1, int(config.max_ops))", src_execute)
        self.assertIn("max_depth=max(1, int(config.max_depth))", src_execute)
        self.assertIn("max_ops=max(1, int(config.max_ops))", src_generate)
        self.assertIn("max_depth=max(1, int(config.max_depth))", src_generate)


if __name__ == "__main__":
    unittest.main()

import pytest
import os
import tempfile
import unittest

from research.scientist.runner import ExperimentRunner, RunConfig

pytestmark = pytest.mark.unit


class _NotebookBreakthrough:
    def get_leaderboard(self, tier=None, limit=5, sort_by="composite_score"):
        if tier != "breakthrough":
            return []
        return [{
            "result_id": "r1",
            "composite_score": 0.93,
            "novelty_confidence": 0.88,
        }]


class _NotebookEmpty:
    def get_leaderboard(self, tier=None, limit=5, sort_by="composite_score"):
        return []


class TestSwitchEpicGuardrails(unittest.TestCase):
    def test_switch_epic_triggers_on_qualified_breakthrough(self):
        runner = ExperimentRunner(os.path.join(tempfile.mkdtemp(), "switch_epic_breakthrough.db"))
        cfg = RunConfig(device="cpu", switch_epic_breakthrough_confidence_min=0.8)
        verdict = runner._evaluate_switch_epic_guardrails(
            config=cfg,
            nb=_NotebookBreakthrough(),
            cycle_index=5,
        )
        self.assertTrue(verdict["should_switch_epic"])
        self.assertIn("decision_ready_breakthrough_detected", verdict["reasons"])

    def test_switch_epic_triggers_on_stagnation_window(self):
        runner = ExperimentRunner(os.path.join(tempfile.mkdtemp(), "switch_epic_stagnation.db"))
        runner._aria_cycle_history = [
            {"delta_stage1_survivors": 0},
            {"delta_stage1_survivors": 0},
            {"delta_stage1_survivors": 0},
            {"delta_stage1_survivors": 0},
            {"delta_stage1_survivors": 0},
        ]
        runner._last_cycle_summary = {"delta_stage1_survivors": 0}

        cfg = RunConfig(device="cpu", switch_epic_stagnation_cycles=6)
        verdict = runner._evaluate_switch_epic_guardrails(
            config=cfg,
            nb=_NotebookEmpty(),
            cycle_index=6,
        )
        self.assertTrue(verdict["should_switch_epic"])
        self.assertIn("stagnation_without_gate_advancement", verdict["reasons"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from research.scientist.notebook import LabNotebook
from research.scientist.runner.results_analysis import _ResultsAnalysisMixin


class _FakeGraph:
    def fingerprint(self) -> str:
        return "fp_stage0_search_proxy"


class _FakeSandboxResult:
    error_type = "RuntimeError"
    error = "shape mismatch"


class _StubResultsAnalysis(_ResultsAnalysisMixin):
    def _extract_graph_metrics(self, graph):
        return {}

    def _extract_sandbox_metrics(self, sandbox_result):
        return {
            "error_type": sandbox_result.error_type,
            "error_message": sandbox_result.error,
        }

    def _merge_s1_telemetry(self, graph_metrics, s1_result):
        return None


class TestResultsAnalysisAccounting(unittest.TestCase):
    def test_search_proxy_stage0_failure_is_labeled_stage0(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test.db"
            nb = LabNotebook(db_path)
            exp_id = nb.start_experiment("evolution", {}, "search-proxy")
            runner = _StubResultsAnalysis()

            with patch(
                "research.scientist.runner.results_analysis.graph_to_json",
                return_value="{}",
            ):
                runner._on_program_evaluated(
                    graph=_FakeGraph(),
                    fitness=0.0,
                    sandbox_result=_FakeSandboxResult(),
                    s1_result=None,
                    eval_counters={"total": 0, "s0": 0, "s1": 0},
                    nb=nb,
                    exp_id=exp_id,
                    model_source="evolution",
                    behavioral_fingerprint=None,
                )
            nb.flush_writes()

            row = nb.conn.execute(
                "SELECT stage0_passed, stage1_passed, stage_at_death "
                "FROM program_results WHERE experiment_id = ?",
                (exp_id,),
            ).fetchone()

            self.assertIsNotNone(row)
            self.assertEqual(int(row["stage0_passed"] or 0), 0)
            self.assertEqual(int(row["stage1_passed"] or 0), 0)
            self.assertEqual(row["stage_at_death"], "stage0")

            nb.close()

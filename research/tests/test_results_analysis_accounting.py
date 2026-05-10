from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from research.scientist.notebook import LabNotebook
from research.scientist.runner.results_analysis import _ResultsAnalysisMixin

_POST_S1_METRICS = {
    "wikitext_perplexity": 119.0,
    "hellaswag_acc": 0.32,
    "blimp_overall_accuracy": 0.69,
    "induction_screening_auc": 0.58,
    "binding_screening_auc": 0.55,
    "binding_screening_composite": 0.17,
    "ar_legacy_auc": 0.23,
}


def _passed_s1_result(final_loss: float = 0.4) -> dict:
    return {
        "passed": True,
        "final_loss": final_loss,
        "initial_loss": 1.0,
        **_POST_S1_METRICS,
    }


class _FakeGraph:
    def __init__(self, fp: str = "fp_stage0_search_proxy"):
        self._fp = fp

    def fingerprint(self) -> str:
        return self._fp


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
        for key in _POST_S1_METRICS:
            if key in s1_result:
                graph_metrics[key] = s1_result[key]
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
                "FROM program_results_compat WHERE experiment_id = ?",
                (exp_id,),
            ).fetchone()

            self.assertIsNotNone(row)
            self.assertEqual(int(row["stage0_passed"] or 0), 0)
            self.assertEqual(int(row["stage1_passed"] or 0), 0)
            self.assertEqual(row["stage_at_death"], "stage0")

            nb.close()

    def test_duplicate_fingerprint_is_skipped_without_crashing_evolution_eval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test.db"
            nb = LabNotebook(db_path)
            first_exp = nb.start_experiment("synthesis", {}, "seed")
            nb.record_program_result(
                experiment_id=first_exp,
                graph_fingerprint="fp_existing_evolution",
                graph_json="{}",
                stage0_passed=True,
                stage05_passed=True,
                stage1_passed=True,
                loss_ratio=0.4,
                model_source="graph_synthesis",
                **_POST_S1_METRICS,
            )
            nb.flush_writes()

            exp_id = nb.start_experiment("evolution", {}, "dup attempt")
            runner = _StubResultsAnalysis()
            counters = {"total": 0, "s0": 0, "s1": 0}

            with patch(
                "research.scientist.runner.results_analysis.graph_to_json",
                return_value="{}",
            ):
                runner._on_program_evaluated(
                    graph=_FakeGraph("fp_existing_evolution"),
                    fitness=0.8,
                    sandbox_result=None,
                    s1_result=_passed_s1_result(),
                    eval_counters=counters,
                    nb=nb,
                    exp_id=exp_id,
                    model_source="evolution",
                    behavioral_fingerprint=None,
                )
            nb.flush_writes()

            rows = nb.conn.execute(
                "SELECT COUNT(*) AS n FROM program_results_compat WHERE graph_fingerprint = ?",
                ("fp_existing_evolution",),
            ).fetchone()
            self.assertEqual(int(rows["n"]), 1)
            evo_rows = nb.conn.execute(
                "SELECT COUNT(*) AS n FROM program_results_compat WHERE experiment_id = ?",
                (exp_id,),
            ).fetchone()
            self.assertEqual(int(evo_rows["n"]), 0)
            self.assertEqual(counters.get("skipped_cross_experiment_dedup"), 1)
            nb.close()

    def test_successful_evolution_survivor_creates_screening_entry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test.db"
            nb = LabNotebook(db_path)
            exp_id = nb.start_experiment("evolution", {}, "screening path")
            runner = _StubResultsAnalysis()

            with patch(
                "research.scientist.runner.results_analysis.graph_to_json",
                return_value="{}",
            ):
                runner._on_program_evaluated(
                    graph=_FakeGraph("fp_evolution_screening"),
                    fitness=0.8,
                    sandbox_result=None,
                    s1_result=_passed_s1_result(),
                    eval_counters={"total": 0, "s0": 0, "s1": 0},
                    nb=nb,
                    exp_id=exp_id,
                    model_source="evolution",
                    behavioral_fingerprint=None,
                )
            nb.flush_writes()

            row = nb.conn.execute(
                "SELECT result_id FROM program_results_compat WHERE graph_fingerprint = ?",
                ("fp_evolution_screening",),
            ).fetchone()
            self.assertIsNotNone(row)
            lb = nb.conn.execute(
                "SELECT tier FROM leaderboard WHERE result_id = ?",
                (row["result_id"],),
            ).fetchone()
            self.assertIsNotNone(lb)
            self.assertEqual(lb["tier"], "screening")
            nb.close()

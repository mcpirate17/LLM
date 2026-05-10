"""Tests for the cross-experiment fingerprint dedup gate (slice 4)."""

from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from research.scientist.notebook import LabNotebook
from research.scientist.notebook.notebook_programs import DuplicateFingerprintError


class TestCrossExperimentDedupGate(unittest.TestCase):
    """Application + schema-level enforcement of the cross-experiment dedup gate."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "lab_notebook.db"
        self.nb = LabNotebook(str(self.db_path))

    def tearDown(self) -> None:
        self.nb.close()
        self._tmp.cleanup()

    def _record_first(self, fp: str = "fp_canonical") -> tuple[str, str]:
        exp_id = self.nb.start_experiment("synthesis", {}, "first record")
        rid = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint=fp,
            graph_json='{"nodes": {"0": {"op_name": "linear_proj"}}}',
            stage1_passed=True,
            loss_ratio=0.5,
            model_source="graph_synthesis",
            trust_label="test_fixture",
        )
        self.nb.flush_writes()
        self.assertTrue(rid)
        return exp_id, rid

    def test_unannotated_duplicate_raises(self) -> None:
        """A second insert of the same fingerprint without a reason must raise."""
        first_exp_id, first_rid = self._record_first()
        second_exp_id = self.nb.start_experiment("synthesis", {}, "duplicate attempt")
        with self.assertRaises(DuplicateFingerprintError) as ctx:
            self.nb.record_program_result(
                experiment_id=second_exp_id,
                graph_fingerprint="fp_canonical",
                graph_json='{"nodes": {"0": {"op_name": "linear_proj"}}}',
                stage1_passed=True,
                loss_ratio=0.4,
                model_source="evolution",
                trust_label="test_fixture",
            )
        err = ctx.exception
        self.assertEqual(err.fingerprint, "fp_canonical")
        self.assertEqual(err.existing_result_id, first_rid)
        self.assertEqual(err.attempted_experiment_id, second_exp_id)
        self.assertEqual(err.attempted_model_source, "evolution")

    def test_intentional_rerun_reason_bypasses_gate(self) -> None:
        """An annotated duplicate succeeds and persists the reason."""
        self._record_first()
        second_exp_id = self.nb.start_experiment("synthesis", {}, "validation rerun")
        rid = self.nb.record_program_result(
            experiment_id=second_exp_id,
            graph_fingerprint="fp_canonical",
            graph_json='{"nodes": {}}',
            intentional_rerun_reason="validation_promotion",
            stage1_passed=True,
            loss_ratio=0.3,
            model_source="validation",
            trust_label="test_fixture",
        )
        self.nb.flush_writes()
        self.assertTrue(rid)
        row = self.nb.conn.execute(
            "SELECT intentional_rerun_reason FROM program_results_compat WHERE result_id = ?",
            (rid,),
        ).fetchone()
        self.assertEqual(row["intentional_rerun_reason"], "validation_promotion")

    def test_first_record_is_unaffected(self) -> None:
        """Recording a brand-new fingerprint never trips the gate."""
        exp_id = self.nb.start_experiment("synthesis", {}, "fresh fp")
        rid = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_brand_new",
            graph_json='{"nodes": {}}',
            stage1_passed=True,
            loss_ratio=0.4,
            model_source="graph_synthesis",
            trust_label="test_fixture",
        )
        self.nb.flush_writes()
        self.assertTrue(rid)

    def test_schema_trigger_blocks_direct_insert(self) -> None:
        """A direct INSERT bypassing record_program_result still fails at the trigger.

        The native aria-db wrapper translates SQLite's IntegrityError from a
        RAISE(ABORT, ...) trigger into ``OperationalError``; the canonical
        sqlite3 driver would raise ``IntegrityError``. Either is fine — the
        rejection message is the contract.
        """
        self._record_first()
        with self.assertRaises(
            (sqlite3.IntegrityError, sqlite3.OperationalError)
        ) as ctx:
            self.nb.conn.execute(
                "INSERT INTO program_results "
                "(result_id, experiment_id, graph_fingerprint, graph_json, "
                "timestamp, stage1_passed) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    "DIRECT_INSERT",
                    "raw_exp",
                    "fp_canonical",
                    "{}",
                    0,
                    0,
                ),
            )
        self.assertIn(
            "duplicate graph_fingerprint without intentional_rerun_reason",
            str(ctx.exception),
        )

    def test_schema_trigger_allows_direct_insert_with_reason(self) -> None:
        """Direct INSERT with a reason is allowed (mirrors the application gate)."""
        self._record_first()
        self.nb.conn.execute(
            "INSERT INTO program_results "
            "(result_id, experiment_id, graph_fingerprint, graph_json, "
            "timestamp, stage1_passed, intentional_rerun_reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "DIRECT_INSERT_WITH_REASON",
                "raw_exp",
                "fp_canonical",
                "{}",
                0,
                0,
                "test_intentional",
            ),
        )
        self.nb.conn.commit()
        row = self.nb.conn.execute(
            "SELECT intentional_rerun_reason FROM program_results WHERE result_id = ?",
            ("DIRECT_INSERT_WITH_REASON",),
        ).fetchone()
        self.assertEqual(row["intentional_rerun_reason"], "test_intentional")


if __name__ == "__main__":
    unittest.main()

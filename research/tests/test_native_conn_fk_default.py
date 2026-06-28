"""Pin aria-db's foreign_keys PRAGMA to OFF, matching Python sqlite3 default.

Background (2026-04-16): aria-db's Rust connection layer shipped with
``PRAGMA foreign_keys=ON`` set at open time. Python's ``sqlite3`` module
defaults to OFF; the rest of the codebase was written against that
default. The mismatch caused silent ``FOREIGN KEY constraint failed``
errors in the writer thread — writes were dropped, reads kept working,
and hours of program_result rows went missing while the runtime event
spool stayed current.

This test inserts a program_result whose ``experiment_id`` references
no real experiment row (an FK violation under ON, a benign insert under
OFF). The row must land on the supported ``program_results_compat``
read surface. A flip back to ON would fail this test loudly.

If you need FKs enforced in production, reintroduce them table-by-table
via ``DEFERRABLE INITIALLY DEFERRED`` on the schema, not via a
connection-wide pragma — and fix the write-order bugs first.
"""

from __future__ import annotations

import uuid

import pytest

from research.scientist.notebook import LabNotebook


@pytest.fixture
def temp_notebook(tmp_path):
    """Yield a LabNotebook backed by a throwaway SQLite file.

    The production notebook lives at research/lab_notebook.db and is
    held open by a long-running server. If tests open their own
    connection against that path and then exit, SQLite's per-process
    teardown checkpoints and truncates the shared WAL — which strands
    any in-flight writes the server had buffered. Use a temp path to
    stay out of that blast radius.
    """
    nb = LabNotebook(db_path=str(tmp_path / "lab_notebook_test.db"))
    try:
        yield nb
    finally:
        nb.flush_writes()


@pytest.fixture
def temp_notebook_legacy(tmp_path):
    """Yield a legacy-sqlite notebook to pin parity with aria-db pragmas."""
    nb = LabNotebook(
        db_path=str(tmp_path / "lab_notebook_legacy_test.db"),
        use_native=False,
    )
    try:
        yield nb
    finally:
        nb.flush_writes()


def test_native_conn_foreign_keys_pragma_is_off(temp_notebook) -> None:
    """Connection-level FK enforcement must stay OFF so writes land."""
    temp_notebook.flush_writes()
    row = temp_notebook.conn.execute("PRAGMA foreign_keys").fetchone()
    assert row is not None
    value = dict(row).get("foreign_keys")
    assert value == 0, (
        f"aria-db connection has foreign_keys={value!r}; expected 0 (OFF). "
        "Flipping this to ON produces silent writer-thread drops on any "
        "INSERT that precedes its parent row."
    )


def test_legacy_conn_foreign_keys_pragma_is_off(temp_notebook_legacy) -> None:
    """Legacy sqlite fallback must match aria-db FK-off behavior."""
    temp_notebook_legacy.flush_writes()
    row = temp_notebook_legacy.conn.execute("PRAGMA foreign_keys").fetchone()
    assert row is not None
    value = dict(row).get("foreign_keys")
    assert value == 0, (
        f"legacy sqlite connection has foreign_keys={value!r}; expected 0 (OFF). "
        "The API/dashboard request path uses use_native=False notebooks."
    )


def test_program_result_insert_lands_without_matching_experiment(
    temp_notebook,
) -> None:
    """Regression: insert with an unknown experiment_id must round-trip.

    Under FK=ON the writer thread would log an error to stderr and move
    on; under FK=OFF the row lands normally.
    """
    temp_notebook.flush_writes()

    tid = uuid.uuid4().hex[:12]
    fake_experiment_id = f"fk-default-test-{tid}"

    rid = temp_notebook.record_program_result(
        experiment_id=fake_experiment_id,
        graph_fingerprint=f"fp-{tid}",
        graph_json='{"nodes":[],"edges":[]}',
        result_id=tid,
        stage0_passed=1,
        stage1_passed=0,
        loss_ratio=0.42,
        error_type="fk_default_regression",
        error_message="regression — safe to delete",
        bypass_quality_gate=True,
    )
    assert rid == tid
    temp_notebook.flush_writes(timeout=10.0)

    row = temp_notebook.conn.execute(
        "SELECT result_id, experiment_id FROM program_results_compat WHERE result_id = ?",
        (tid,),
    ).fetchone()
    assert row is not None, (
        "record_program_result returned a result_id but the row did "
        "not land in program_results_compat — likely a silent writer-thread "
        "drop. Check PRAGMA foreign_keys and the aria-db stderr output."
    )
    assert dict(row)["experiment_id"] == fake_experiment_id

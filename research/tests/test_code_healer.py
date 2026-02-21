import os
import tempfile

import pytest

from research.healer import CodeHealer
from research.healer.core import HealerTaskSpec
from research.scientist.notebook import LabNotebook


def test_code_healer_state_machine_completes_with_allowed_commands():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "healer.db")
    nb = LabNotebook(db_path)
    try:
        exp_id = nb.start_experiment("synthesis", {"n_programs": 1}, "healer test")
    finally:
        nb.close()

    healer = CodeHealer(db_path)
    result = healer.open_and_run(
        HealerTaskSpec(
            experiment_id=exp_id,
            trigger_type="integrity_failure",
            scope="Verify healer path",
            reproduction_steps=["python -m py_compile scientist/preregistration.py"],
            acceptance_tests=["python -m py_compile scientist/preregistration.py"],
            trigger_payload={"source": "unit_test"},
        )
    )
    assert result["state"] in {"completed", "failed"}

    nb2 = LabNotebook(db_path)
    try:
        rows = nb2.conn.execute("SELECT * FROM healer_tasks").fetchall()
        assert rows
        assert rows[0]["experiment_id"] == exp_id
        assert rows[0]["patch_summary"]
        assert rows[0]["risk_assessment"]
    finally:
        nb2.close()


def test_code_healer_blocks_disallowed_command():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "healer_block.db")
    healer = CodeHealer(db_path)
    task_id = healer.open_task(
        HealerTaskSpec(
            experiment_id=None,
            trigger_type="repeated_exception",
            scope="blocked command",
            reproduction_steps=["rm -rf /tmp/not_allowed"],
            acceptance_tests=["python -m py_compile scientist/preregistration.py"],
            trigger_payload={},
        )
    )
    with pytest.raises(Exception):
        healer.run_task(task_id)

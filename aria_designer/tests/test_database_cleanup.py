from __future__ import annotations

from pathlib import Path

from app import database as db


def test_cleanup_orphaned_workflows_skips_workflows_with_pending_proposals(tmp_path):
    db_path = Path(tmp_path) / "designer_test.db"
    db.init_db(db_path)

    wf_id = "wf_with_proposal"
    ts = "2000-01-01T00:00:00Z"
    db.save_workflow(wf_id, "Old Workflow", '{"nodes":[],"edges":[]}', created_at=ts, updated_at=ts)
    db.save_proposal("prop_1", wf_id, '{"ops":[]}', "old pending", created_at=ts)

    cleaned = db.cleanup_orphaned_workflows(max_age_hours=1)
    assert cleaned == 0
    assert db.get_workflow(wf_id) is not None

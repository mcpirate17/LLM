from __future__ import annotations

import pytest

from research.scientist.api import create_app
from research.scientist.notebook import LabNotebook

pytestmark = pytest.mark.api


def test_recommendation_signals_endpoint_returns_aggregate_payload(tmp_path):
    db_path = str(tmp_path / "signals.db")
    nb = LabNotebook(db_path)
    try:
        now = 1_700_000_000.0
        nb.conn.execute(
            """INSERT INTO op_success_rates
               (op_name, n_used, n_stage0_passed, n_stage05_passed, n_stage1_passed, last_updated)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("rmsnorm_pre", 20, 20, 18, 16, now),
        )
        nb.conn.execute(
            """INSERT INTO failure_signatures
               (signature, n_failures, n_successes, error_types, last_updated)
               VALUES (?, ?, ?, ?, ?)""",
            ("bad_op->worse_op", 9, 1, "nan_error", now),
        )
        nb.record_insight(
            "success_factor",
            "Top-performing ops (S1 rate): rmsnorm_pre(80%), linear_proj(70%), rope_rotate(60%). These compose well into learnable architectures.",
            experiment_id="exp-signals",
            confidence=0.8,
        )
        nb.conn.commit()
    finally:
        nb.close()

    app = create_app(notebook_path=db_path)
    client = app.test_client()

    resp = client.get("/api/analytics/recommendation-signals")
    assert resp.status_code == 200
    data = resp.get_json()

    assert isinstance(data, dict)
    assert data.get("source") == "research.analytics"
    assert "generated_at" in data
    assert isinstance(data.get("op_priors"), list)
    assert isinstance(data.get("toxic_ops"), list)
    assert isinstance(data.get("insights"), list)
    assert isinstance(data.get("summary"), dict)
    assert any(row.get("op_name") == "rmsnorm_pre" for row in data.get("op_priors", []))

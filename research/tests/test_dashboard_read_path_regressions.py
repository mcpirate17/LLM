from __future__ import annotations

import json
import subprocess
import sys
import time

import pytest

from research.scientist.api import create_app
from research.scientist.notebook import LabNotebook

pytestmark = [pytest.mark.api]


def _seed_minimal_dashboard_db(tmp_path):
    db_path = tmp_path / "dashboard_regressions.db"
    nb = LabNotebook(str(db_path))
    exp_id = nb.start_experiment(
        experiment_type="screening",
        config={"source": "test"},
        hypothesis="dashboard read-path regression seed",
        require_preregistration=False,
    )
    nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="fp-minimal",
        graph_json=json.dumps({"nodes": []}),
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=False,
        novelty_score=0.0,
    )
    nb.complete_experiment(
        exp_id,
        {"total": 1, "stage0_passed": 1, "stage05_passed": 1, "stage1_passed": 0},
        "seeded for dashboard regression coverage",
        "curious",
    )
    nb.close()
    return str(db_path)


def test_read_only_flush_writes_is_effectively_free(tmp_path):
    db_path = _seed_minimal_dashboard_db(tmp_path)
    nb = LabNotebook(db_path, read_only=True, use_native=False)
    t0 = time.perf_counter()
    for _ in range(100):
        nb.flush_writes()
    elapsed = time.perf_counter() - t0
    nb.close()
    assert elapsed < 0.25, f"read-only flush_writes took too long: {elapsed:.3f}s"


def test_dashboard_read_path_does_not_backfill_graph_features(tmp_path, monkeypatch):
    db_path = _seed_minimal_dashboard_db(tmp_path)

    def _forbid_backfill(self, *args, **kwargs):
        raise AssertionError("dashboard GET should not backfill graph features")

    monkeypatch.setattr(
        LabNotebook,
        "_ensure_graph_features",
        _forbid_backfill,
        raising=True,
    )

    app = create_app(notebook_path=db_path)
    client = app.test_client()
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["is_running"] is False


def test_experiment_analytics_import_does_not_load_torch():
    script = """
import sys
before = set(sys.modules)
from research.scientist.analytics import ExperimentAnalytics
after = set(sys.modules)
heavy = sorted(
    name for name in (after - before)
    if name == 'torch' or name.startswith('torch.')
)
print(ExperimentAnalytics.__name__)
print(len(heavy))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd="/home/tim/Projects/LLM",
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = result.stdout.strip().splitlines()
    assert stdout[0] == "ExperimentAnalytics"
    assert stdout[1] == "0", result.stdout

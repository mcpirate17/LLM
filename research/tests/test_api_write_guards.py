from __future__ import annotations

from flask import Flask

from research.scientist.api_routes import _utils, diagnostics_bp
from research.scientist.api_routes.deps import ApiRouteContext


def test_report_cache_diagnostics_reads_readonly_and_cleanup_uses_writer(monkeypatch):
    calls: list[bool] = []

    class FakeNotebook:
        def __init__(self, read_only: bool):
            self._read_only = read_only

        def get_report_snapshot_stats(self):
            assert self._read_only is True
            return {"rows": 0}

        def cleanup_report_snapshots(self, **_kwargs):
            assert self._read_only is False
            return {"deleted": 0}

        def close(self):
            pass

    def fake_get_notebook(_path: str, *, read_only: bool):
        calls.append(read_only)
        return FakeNotebook(read_only)

    monkeypatch.setattr(_utils, "get_notebook", fake_get_notebook)
    monkeypatch.setattr(diagnostics_bp, "get_notebook", fake_get_notebook)

    app = Flask(__name__)
    diagnostics_bp.register_diagnostics_routes(
        app,
        ApiRouteContext(
            notebook_path="unused.db",
            dashboard_index_path=lambda: None,
            dashboard_missing_response=lambda: ("missing", 404),
            is_asset_path=lambda _path: False,
        ),
    )
    client = app.test_client()

    response = client.get("/api/diagnostics/report-cache")
    assert response.status_code == 200
    assert calls == [True]

    calls.clear()
    response = client.get("/api/diagnostics/report-cache?cleanup=1")
    assert response.status_code == 200
    assert calls == [True, False]

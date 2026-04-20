from __future__ import annotations

from pathlib import Path

from research.scientist.api_routes import _designer


def test_designer_service_status_prefers_dashboard_dist(monkeypatch, tmp_path):
    dist_dir = tmp_path / "ui" / "dist"
    dist_dir.mkdir(parents=True)
    (dist_dir / "index.html").write_text("ok", encoding="utf-8")

    class _Resp:
        status_code = 200

    def fake_get(url, timeout=1.0):
        assert url == "http://127.0.0.1:8091/health"
        return _Resp()

    monkeypatch.setattr(_designer, "_ARIA_DESIGNER_ROOT", tmp_path)
    monkeypatch.setattr(
        _designer,
        "_ARIA_DESIGNER_DASHBOARD_UI",
        "http://127.0.0.1:5000/designer-proxy/",
    )
    monkeypatch.setattr(_designer._requests, "get", fake_get)
    _designer._invalidate_health_cache()

    status = _designer.designer_service_status()

    assert status["api_up"] is True
    assert status["ui_up"] is True
    assert status["running"] is True
    assert status["ui_mode"] == "dashboard-dist"
    assert status["ui_health_url"] == "http://127.0.0.1:5000/designer-proxy/"


def test_start_designer_services_uses_runtime_scripts(monkeypatch, tmp_path):
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir(parents=True)
    runtime_up = tools_dir / "run_up.sh"
    runtime_down = tools_dir / "run_down.sh"
    runtime_up.write_text("#!/bin/sh\n", encoding="utf-8")
    runtime_down.write_text("#!/bin/sh\n", encoding="utf-8")

    popen_calls = []
    run_calls = []
    status_calls = {"n": 0}

    class DummyPopen:
        def __init__(self, args, **kwargs):
            popen_calls.append((args, kwargs))
            self.pid = 424242

    def fake_run(args, **kwargs):
        run_calls.append((args, kwargs))
        return None

    def fake_status():
        status_calls["n"] += 1
        if status_calls["n"] == 1:
            return {"api_up": False, "ui_up": True, "running": False}
        return {"api_up": True, "ui_up": True, "running": True}

    monkeypatch.setattr(_designer, "_ARIA_DESIGNER_ROOT", tmp_path)
    monkeypatch.setattr(_designer, "designer_service_status", fake_status)
    monkeypatch.setattr(_designer.subprocess, "run", fake_run)
    monkeypatch.setattr(_designer.subprocess, "Popen", DummyPopen)
    monkeypatch.setattr(_designer.time, "sleep", lambda _: None)

    result = _designer.start_designer_services(force_restart=False)

    assert result["ok"] is True
    assert result["pid"] == 424242
    assert popen_calls
    assert Path(popen_calls[0][0][0]) == runtime_up
    assert run_calls == []

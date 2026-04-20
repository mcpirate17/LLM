from __future__ import annotations

from unittest.mock import patch

from research.scientist import api as api_mod
from research.scientist.api_routes import _helpers
from research.scientist.api_routes import system_bp
from research.scientist.notebook import LabNotebook


class _FakeAria:
    def __init__(self):
        self.get_llm_config_calls = 0
        self.NAME = "Aria"
        self.TITLE = "Scientist"
        self.AVATAR = "A"
        self._llm_initialized = False
        self.state = type(
            "State",
            (),
            {
                "mood": "idle",
                "energy": 1.0,
                "experiments_today": 0,
                "discoveries_today": 0,
                "current_hypothesis": "",
                "research_focus": "analysis",
                "insights": [],
            },
        )()

    def get_llm_config(self):
        self.get_llm_config_calls += 1
        return {"backend": None, "available": False, "configured": False}

    def _sanitize_hypothesis(self, text):
        return text


def test_create_app_does_not_eagerly_load_persisted_llm_config(tmp_path):
    db_path = str(tmp_path / "lazy.db")
    with (
        patch.object(
            api_mod, "_ensure_default_dashboard_build", lambda static_folder: None
        ),
        patch.object(_helpers, "load_persisted_llm_config") as load_mock,
    ):
        api_mod.create_app(
            notebook_path=db_path,
            static_folder=str(tmp_path / "missing_static"),
        )
    load_mock.assert_not_called()


def test_get_aria_for_notebook_loads_persisted_config_once(tmp_path, monkeypatch):
    db_path = str(tmp_path / "lazy.db")
    calls = {"load": 0}
    fake_aria = _FakeAria()
    monkeypatch.setattr(_helpers, "get_aria", lambda: fake_aria)
    monkeypatch.setattr(
        _helpers,
        "load_persisted_llm_config",
        lambda notebook_path: calls.__setitem__("load", calls["load"] + 1),
    )
    _helpers._PERSISTED_LLM_CONFIG_LOADED.clear()

    assert _helpers.get_aria_for_notebook(db_path) is fake_aria
    assert _helpers.get_aria_for_notebook(db_path) is fake_aria
    assert calls["load"] == 1


def _init_notebook(db_path: str) -> None:
    nb = LabNotebook(db_path, use_native=False)
    nb.close()


def test_llm_config_route_is_passive(tmp_path, monkeypatch):
    db_path = str(tmp_path / "lazy.db")
    _init_notebook(db_path)
    fake_aria = _FakeAria()

    monkeypatch.setattr(
        api_mod, "_ensure_default_dashboard_build", lambda static_folder: None
    )
    monkeypatch.setattr(_helpers, "get_aria", lambda: fake_aria)
    _helpers._PERSISTED_LLM_CONFIG_LOADED.clear()

    with patch.object(_helpers, "load_persisted_llm_config") as load_mock:
        app = api_mod.create_app(
            notebook_path=db_path,
            static_folder=str(tmp_path / "missing_static"),
        )
        client = app.test_client()
        response = client.get("/api/llm/config")

    assert response.status_code == 200
    load_mock.assert_not_called()
    assert fake_aria.get_llm_config_calls == 0


def test_dashboard_summary_does_not_lazy_load_persisted_llm_config(
    tmp_path, monkeypatch
):
    db_path = str(tmp_path / "lazy.db")
    _init_notebook(db_path)
    fake_aria = _FakeAria()
    monkeypatch.setattr(
        api_mod, "_ensure_default_dashboard_build", lambda static_folder: None
    )
    monkeypatch.setattr(_helpers, "get_aria", lambda: fake_aria)
    load_mock = patch.object(_helpers, "load_persisted_llm_config")
    with load_mock as mocked_load:
        _helpers._PERSISTED_LLM_CONFIG_LOADED.clear()
        app = api_mod.create_app(
            notebook_path=db_path,
            static_folder=str(tmp_path / "missing_static"),
        )
        client = app.test_client()
        response = client.get("/api/dashboard/summary")

    assert response.status_code == 200
    mocked_load.assert_not_called()


def test_system_status_uses_passive_llm_snapshot(tmp_path, monkeypatch):
    db_path = str(tmp_path / "lazy.db")
    _init_notebook(db_path)
    fake_aria = _FakeAria()
    monkeypatch.setattr(
        api_mod, "_ensure_default_dashboard_build", lambda static_folder: None
    )
    monkeypatch.setattr(_helpers, "get_aria", lambda: fake_aria)
    monkeypatch.setattr(
        system_bp,
        "get_passive_llm_config",
        lambda notebook_path, aria=None: {
            "backend": "anthropic",
            "available": False,
            "configured": True,
            "reachable": False,
            "initialized": False,
            "source": "persisted",
        },
    )
    with patch.object(_helpers, "load_persisted_llm_config") as load_mock:
        app = api_mod.create_app(
            notebook_path=db_path,
            static_folder=str(tmp_path / "missing_static"),
        )
        client = app.test_client()
        response = client.get("/api/system/status")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["llm"]["configured"] is True
    assert payload["llm"]["backend"] == "anthropic"
    load_mock.assert_not_called()

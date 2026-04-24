from __future__ import annotations

import builtins
import importlib
import sys

import pytest


def test_native_conn_import_survives_missing_aria_db():
    module_names = (
        "research.scientist.notebook.native_conn",
        "research.scientist.notebook.notebook_core",
    )
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "aria_db":
            raise ModuleNotFoundError("No module named 'aria_db'")
        return original_import(name, *args, **kwargs)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(builtins, "__import__", blocked_import)
        for module_name in module_names:
            sys.modules.pop(module_name, None)

        native_conn = importlib.import_module("research.scientist.notebook.native_conn")
        notebook_core = importlib.import_module(
            "research.scientist.notebook.notebook_core"
        )

        assert native_conn.aria_db is None
        assert hasattr(notebook_core, "_NotebookCore")
        with pytest.raises(ModuleNotFoundError, match="aria_db is required"):
            native_conn.NativeConnectionWrapper("/tmp/missing-aria-db.sqlite")

    for module_name in module_names:
        sys.modules.pop(module_name, None)
    importlib.import_module("research.scientist.notebook.native_conn")
    importlib.import_module("research.scientist.notebook.notebook_core")


def test_lab_notebook_falls_back_to_sqlite_when_aria_db_missing(tmp_path):
    module_names = (
        "research.scientist.notebook.native_conn",
        "research.scientist.notebook.notebook_core",
    )
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "aria_db":
            raise ModuleNotFoundError("No module named 'aria_db'")
        return original_import(name, *args, **kwargs)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(builtins, "__import__", blocked_import)
        for module_name in module_names:
            sys.modules.pop(module_name, None)

        from research.scientist import notebook as notebook_pkg

        monkeypatch.setattr(notebook_pkg, "_LAB_NOTEBOOK_CLASS", None)
        LabNotebook = getattr(notebook_pkg, "LabNotebook")

        nb = LabNotebook(tmp_path / "fallback.db", skip_migrate=True)
        try:
            assert nb._use_native is False
            assert type(nb.conn).__name__ == "_SqliteConnectionAdapter"
            row = nb.conn.execute("SELECT 1 AS ok").fetchone()
            assert dict(row)["ok"] == 1
        finally:
            nb.close()

    for module_name in module_names:
        sys.modules.pop(module_name, None)
    importlib.import_module("research.scientist.notebook.native_conn")
    importlib.import_module("research.scientist.notebook.notebook_core")

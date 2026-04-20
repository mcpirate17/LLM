from pathlib import Path

from aria_designer.runtime import importer


def test_get_notebook_uses_read_only_legacy_path(monkeypatch, tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    db_path.write_text("", encoding="utf-8")
    calls = {}

    class FakeNotebook:
        pass

    def fake_lab_notebook(path, **kwargs):
        calls["path"] = path
        calls["kwargs"] = kwargs
        return FakeNotebook()

    monkeypatch.setattr(importer, "_RESEARCH_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "research.scientist.notebook.LabNotebook",
        fake_lab_notebook,
    )

    nb = importer._get_notebook()

    assert isinstance(nb, FakeNotebook)
    assert Path(calls["path"]) == db_path
    assert calls["kwargs"] == {"read_only": True, "use_native": False}

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from research.scientist.notebook import LabNotebook
from research.scientist.runtime_events.bootstrap import runtime_events_root_for


def test_lab_notebook_rejects_mock_db_path_before_creating_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mock_path = MagicMock()
    mock_path.__fspath__.return_value = "<MagicMock name='mock.db_path' id='123'>"

    with pytest.raises(TypeError):
        LabNotebook(mock_path)

    assert not any(tmp_path.iterdir())


def test_runtime_events_reject_in_memory_notebook_path():
    with pytest.raises(ValueError):
        runtime_events_root_for(":memory:")


def test_lab_notebook_in_memory_does_not_create_repo_local_memory_file(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    nb = LabNotebook(":memory:", use_native=False)
    try:
        assert str(nb.db_path) == ":memory:"
        assert not (tmp_path / ":memory:").exists()
    finally:
        nb.close()

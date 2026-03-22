"""Tests for data_transform/select_columns component fallback."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


_COMPONENT_PATH = (
    Path(__file__).parent.parent
    / "components"
    / "data_transform"
    / "select_columns"
    / "kernel_fallback.py"
)


def _handler():
    spec = importlib.util.spec_from_file_location(
        "select_columns_handler", str(_COMPONENT_PATH)
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.ComponentHandler()


def test_select_by_names_uses_schema_mapping():
    handler = _handler()
    data = torch.tensor(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
        ]
    )
    out = handler.forward(
        {"data": data},
        {
            "selection_mode": "names",
            "schema_columns": "id,age,score",
            "selected_columns": "score,id",
            "keep_order": True,
            "drop_invalid": True,
            "schema_validation": "strict",
        },
    )
    selected = out["selected"]
    assert selected.shape == (2, 2)
    assert torch.equal(selected[:, 0], data[:, 2])
    assert torch.equal(selected[:, 1], data[:, 0])


def test_select_by_indices_filters_invalid_when_allowed():
    handler = _handler()
    data = torch.randn(5, 4)
    out = handler.forward(
        {"data": data},
        {
            "selection_mode": "indices",
            "selected_indices": "3,9,1",
            "drop_invalid": True,
            "schema_validation": "none",
        },
    )
    selected = out["selected"]
    assert selected.shape == (5, 2)
    assert torch.equal(selected[:, 0], data[:, 3])
    assert torch.equal(selected[:, 1], data[:, 1])


def test_strict_mode_rejects_unknown_name():
    handler = _handler()
    data = torch.randn(3, 2)

    try:
        handler.forward(
            {"data": data},
            {
                "selection_mode": "names",
                "schema_columns": "a,b",
                "selected_columns": "a,c",
                "drop_invalid": False,
                "schema_validation": "strict",
            },
        )
    except ValueError as exc:
        assert "Unknown selected_columns" in str(exc)
    else:
        raise AssertionError("Expected ValueError for strict unknown selected column")

"""Tests for data_transform/join component fallback kernel."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


_KERNEL = (
    Path(__file__).resolve().parents[1]
    / "components"
    / "data_transform"
    / "join"
    / "kernel_fallback.py"
)


def _load_handler():
    spec = importlib.util.spec_from_file_location("join_kernel", str(_KERNEL))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.ComponentHandler()


def test_join_inner_matches_only():
    handler = _load_handler()
    left = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    right = torch.tensor([[2.0, 200.0], [3.0, 300.0], [4.0, 400.0]])
    out = handler.forward(
        {"left": left, "right": right},
        {
            "join_mode": "inner",
            "left_key_index": 0,
            "right_key_index": 0,
            "include_key_columns": False,
        },
    )["joined"]
    assert out.shape == (2, 2)
    assert torch.allclose(out, torch.tensor([[20.0, 200.0], [30.0, 300.0]]))


def test_join_outer_keeps_unmatched_rows_with_fill():
    handler = _load_handler()
    left = torch.tensor([[1.0, 10.0], [2.0, 20.0]])
    right = torch.tensor([[2.0, 200.0], [3.0, 300.0]])
    out = handler.forward(
        {"left": left, "right": right},
        {
            "join_mode": "outer",
            "left_key_index": 0,
            "right_key_index": 0,
            "include_key_columns": False,
            "missing_fill_value": -1.0,
        },
    )["joined"]
    assert out.shape == (3, 2)
    assert torch.allclose(
        out,
        torch.tensor(
            [
                [10.0, -1.0],
                [20.0, 200.0],
                [-1.0, 300.0],
            ]
        ),
    )


def test_join_strict_schema_validation_fails():
    handler = _load_handler()
    left = torch.randn(3, 2)
    right = torch.randn(3, 2)
    with pytest.raises(ValueError, match="expected_left_dim"):
        handler.forward(
            {"left": left, "right": right},
            {
                "join_mode": "inner",
                "schema_validation": "strict",
                "expected_left_dim": 4,
                "expected_right_dim": 2,
            },
        )

"""Tests for data_transform/split_train_val_test component fallback."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


_COMPONENT_PATH = (
    Path(__file__).parent.parent
    / "components"
    / "data_transform"
    / "split_train_val_test"
    / "kernel_fallback.py"
)


def _handler():
    spec = importlib.util.spec_from_file_location("split_train_val_test_handler", str(_COMPONENT_PATH))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.ComponentHandler()


def test_split_is_deterministic_for_same_seed():
    handler = _handler()
    data = torch.arange(0, 120, dtype=torch.float32).reshape(40, 3)
    config = {
        "train_ratio": 0.7,
        "val_ratio": 0.2,
        "test_ratio": 0.1,
        "seed": 123,
        "shuffle": True,
        "stratify": False,
    }

    out_a = handler.forward({"data": data}, config)
    out_b = handler.forward({"data": data}, config)

    assert torch.equal(out_a["train"], out_b["train"])
    assert torch.equal(out_a["val"], out_b["val"])
    assert torch.equal(out_a["test"], out_b["test"])
    assert out_a["train"].shape[0] + out_a["val"].shape[0] + out_a["test"].shape[0] == data.shape[0]


def test_stratified_split_preserves_all_rows():
    handler = _handler()

    class_zero = torch.cat([torch.zeros(50, 1), torch.randn(50, 2)], dim=1)
    class_one = torch.cat([torch.ones(50, 1), torch.randn(50, 2)], dim=1)
    data = torch.cat([class_zero, class_one], dim=0)

    config = {
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "seed": 7,
        "shuffle": True,
        "stratify": True,
        "stratify_col": 0,
        "stratify_bins": 4,
    }

    out = handler.forward({"data": data}, config)

    for split_name in ("train", "val", "test"):
        split = out[split_name]
        labels = split[:, 0]
        unique = torch.unique(labels)
        assert set(unique.tolist()).issubset({0.0, 1.0})

    total_rows = out["train"].shape[0] + out["val"].shape[0] + out["test"].shape[0]
    assert total_rows == data.shape[0]


def test_invalid_ratio_config_raises():
    handler = _handler()
    data = torch.randn(12, 4)
    bad_config = {
        "train_ratio": 0.8,
        "val_ratio": 0.3,
        "test_ratio": 0.1,
    }

    try:
        handler.forward({"data": data}, bad_config)
    except ValueError as exc:
        assert "must equal 1.0" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid ratio configuration")

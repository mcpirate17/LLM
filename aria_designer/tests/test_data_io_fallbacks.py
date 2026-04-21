from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


_ROOT = Path(__file__).parent.parent


def _load_component_handler(component_id: str):
    fallback_path = (
        _ROOT / "components" / "data_io" / component_id / "kernel_fallback.py"
    )
    spec = importlib.util.spec_from_file_location(
        f"{component_id}_fallback", fallback_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ComponentHandler()


def test_file_loader_fallback_csv_preserves_2d_shape_for_single_row(tmp_path):
    handler = _load_component_handler("file_loader")
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")

    out = handler.forward(
        {},
        {
            "file_path": str(csv_path),
            "file_format": "csv",
            "delimiter": ",",
            "has_header": True,
        },
    )

    assert out["data"].shape == (1, 3)
    assert torch.equal(out["data"], torch.tensor([[1.0, 2.0, 3.0]]))


def test_file_loader_fallback_txt_streams_values(tmp_path):
    handler = _load_component_handler("file_loader")
    txt_path = tmp_path / "sample.txt"
    txt_path.write_text("1.5\n\n-2.0\n3.25\n", encoding="utf-8")

    out = handler.forward(
        {},
        {
            "file_path": str(txt_path),
            "file_format": "txt",
        },
    )

    assert torch.equal(out["data"], torch.tensor([1.5, -2.0, 3.25]))

"""Native kernel coverage checks for new data_io components."""

from __future__ import annotations

from pathlib import Path

import yaml


_ROOT = Path(__file__).parent.parent
_COMPONENTS = [
    "file_loader",
    "binary_file_reader",
    "file_writer",
]


def test_data_io_native_kernel_files_and_manifest_flags():
    base = _ROOT / "components" / "data_io"
    for comp_id in _COMPONENTS:
        comp_dir = base / comp_id
        manifest_path = comp_dir / "manifest.yaml"
        kernel_path = comp_dir / "kernel.c"

        assert manifest_path.exists(), f"Missing manifest: {manifest_path}"
        assert kernel_path.exists(), f"Missing native kernel: {kernel_path}"

        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        implementation = manifest.get("implementation") or {}
        assert implementation.get("native") == "kernel.c"
        assert implementation.get("python") == "kernel_fallback.py"


def test_data_io_native_kernel_entrypoints_present():
    base = _ROOT / "components" / "data_io"
    required_symbols = [
        "int component_validate(",
        "int component_forward(",
        "void component_cleanup(",
    ]

    for comp_id in _COMPONENTS:
        kernel_path = base / comp_id / "kernel.c"
        source = kernel_path.read_text(encoding="utf-8")
        for symbol in required_symbols:
            assert symbol in source, f"{comp_id}/kernel.c missing entrypoint: {symbol}"

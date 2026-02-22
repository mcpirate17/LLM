from __future__ import annotations

import json
from pathlib import Path


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_native_ir_schema_v1_contract_shape():
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "native_ir.v1.json"
    schema = json.loads(_read(schema_path))

    assert schema.get("$id") == "native_ir.v1"
    assert schema.get("type") == "object"

    required = set(schema.get("required") or [])
    assert {"schema_version", "model_dim", "nodes", "edges", "output_node_id"}.issubset(required)

    props = schema.get("properties") or {}
    assert props.get("schema_version", {}).get("const") == "native_ir.v1"
    assert props.get("nodes", {}).get("type") == "array"
    assert props.get("edges", {}).get("type") == "array"


def test_runner_abi_header_exports_required_symbols():
    header_path = Path(__file__).resolve().parents[1] / "runtime" / "native" / "include" / "runner_abi.h"
    text = _read(header_path)

    required_symbols = [
        "nr_runtime_init",
        "nr_runtime_shutdown",
        "nr_compile",
        "nr_execute",
        "nr_release_model",
        "nr_query_capabilities",
        "nr_set_strict_mode",
        "nr_get_fallback_count",
    ]

    for symbol in required_symbols:
        assert symbol in text, f"Missing runner ABI symbol: {symbol}"


def test_kernel_and_profile_abi_headers_export_required_symbols():
    kernel_header = Path(__file__).resolve().parents[1] / "runtime" / "native" / "include" / "kernel_abi.h"
    profile_header = Path(__file__).resolve().parents[1] / "runtime" / "native" / "include" / "profile_abi.h"

    kernel_text = _read(kernel_header)
    profile_text = _read(profile_header)

    for symbol in ["nk_register", "nk_is_registered", "nk_dispatch", "nk_list_registered"]:
        assert symbol in kernel_text, f"Missing kernel ABI symbol: {symbol}"

    for symbol in ["np_set_event_sink", "np_emit_event", "np_set_memory_sink", "np_get_peak_memory"]:
        assert symbol in profile_text, f"Missing profile ABI symbol: {symbol}"

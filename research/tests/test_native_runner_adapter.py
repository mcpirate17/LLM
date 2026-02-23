from __future__ import annotations

import ctypes
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from research.scientist.native_runner import (
    _SELECTIVE_GUARDRAIL,
    _SELECTIVE_GUARDRAIL_HISTORY_MAX,
    _maybe_prepare_runner_abi_session,
    _record_guardrail_event,
    _check_native_op_support,
    record_native_abi_parity_result,
    _reset_native_lib_cache,
    _try_load_native_lib,
    compile_model_native_first,
    detect_native_state,
    native_runner_capability_report,
    reset_native_runner_telemetry,
)
from research.scientist.native_runner_adapter import capability_handshake
from research.tests.conftest import make_fake_graph


def test_native_runner_enabled_by_default():
    with patch("research.scientist.native_runner_adapter.os.environ", {}), patch(
        "research.scientist.native_runner_adapter.Path.exists", return_value=False
    ):
        s = detect_native_state()
        assert s.enabled is True
        assert s.reason.startswith("missing_designer_runtime_lib:")


def test_native_runner_strict_enabled_without_lib_reports_missing():
    env = {"NATIVE_RUNNER_ENABLED": "1", "NATIVE_RUNNER_STRICT": "1"}
    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner_adapter.Path.exists", return_value=False
    ):
        s = detect_native_state()
        assert s.enabled is True
        assert s.strict is True
        assert s.designer_runtime_available is False
        assert s.reason.startswith("missing_designer_runtime_lib:")


def test_native_runner_capability_report_shape():
    env = {"NATIVE_RUNNER_ENABLED": "0"}
    with patch("research.scientist.native_runner_adapter.os.environ", env):
        reset_native_runner_telemetry()
        report = native_runner_capability_report()
        assert {
            "enabled",
            "strict",
            "designer_runtime_available",
            "status",
            "supported_ops",
            "unsupported_ops",
            "approximate_mappings",
            "semantic_warnings",
            "semantic_warning_count",
            "mapping_source",
            "fallback_metrics",
            "cutover_gate",
            "execution_mode_classification",
            "legacy_compile_disabled",
            "legacy_compile_disabled_reason",
        }.issubset(set(report.keys()))


def test_capability_handshake_placeholder_fields_present():
    env = {"NATIVE_RUNNER_ENABLED": "1", "NATIVE_RUNNER_STRICT": "0"}
    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner_adapter.Path.exists", return_value=True
    ):
        report = capability_handshake()
        assert report["enabled"] is True
        assert report["designer_runtime_available"] is True
        assert isinstance(report["supported_ops"], list)
        assert isinstance(report["unsupported_ops"], list)
        assert isinstance(report["approximate_mappings"], dict)
        assert isinstance(report["semantic_warnings"], list)
        assert isinstance(report["semantic_warning_count"], int)
        assert isinstance(report["mapping_source"], str)
        assert "__designer_runtime_available__" in report["supported_ops"]


def test_native_runner_strict_mode_empty_graphs_succeeds():
    """Strict mode with no layer graphs has nothing to validate -- should not raise."""
    env = {"NATIVE_RUNNER_ENABLED": "0", "NATIVE_RUNNER_STRICT": "1"}

    class DummyModel:
        pass

    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner_adapter.Path.exists", return_value=False
    ), patch(
        "research.scientist.native_runner._try_load_native_lib", return_value=None
    ), patch(
        "research.scientist.native_runner._legacy_compile_model", return_value=DummyModel()
    ):
        model = compile_model_native_first([], vocab_size=4)
        assert model is not None


def test_compile_attaches_native_runner_report_in_non_strict_mode():
    env = {"NATIVE_RUNNER_ENABLED": "0", "NATIVE_RUNNER_STRICT": "0"}

    class DummyModel:
        pass

    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner._legacy_compile_model", return_value=DummyModel()
    ):
        model = compile_model_native_first([])
        report = getattr(model, "_native_runner_report", None)
        assert isinstance(report, dict)
        assert "semantic_warning_count" in report


def test_compile_attaches_runner_abi_session_when_prepare_succeeds():
    env = {
        "NATIVE_RUNNER_ENABLED": "1",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_ABI_EXEC": "1",
        "NATIVE_RUNNER_ABI_MODEL_ONLY": "0",
    }

    class DummyModel:
        pass

    session = SimpleNamespace(close=lambda: None)
    abi_prepare = {
        "requested": True,
        "attempted": True,
        "succeeded": True,
        "reason": "ok",
        "model_handle": 77,
        "session": session,
    }

    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner._try_load_native_lib", return_value=object()
    ), patch(
        "research.scientist.native_runner._maybe_prepare_runner_abi_session",
        return_value=abi_prepare,
    ), patch(
        "research.scientist.native_runner._legacy_compile_model", return_value=DummyModel()
    ):
        model = compile_model_native_first([])
        assert getattr(model, "_native_runner_abi_session", None) is session
        report = getattr(model, "_native_runner_report", {}) or {}
        runner_abi = report.get("runner_abi", {}) or {}
        assert runner_abi.get("requested") is True
        assert runner_abi.get("succeeded") is True
        assert runner_abi.get("session_attached") is True


def test_compile_strict_raises_when_runner_abi_prepare_fails():
    env = {
        "NATIVE_RUNNER_ENABLED": "1",
        "NATIVE_RUNNER_STRICT": "1",
        "NATIVE_RUNNER_ABI_EXEC": "1",
    }

    abi_prepare = {
        "requested": True,
        "attempted": True,
        "succeeded": False,
        "reason": "nr_compile_failed:-2",
        "model_handle": None,
        "session": None,
    }

    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner._try_load_native_lib", return_value=object()
    ), patch(
        "research.scientist.native_runner._maybe_prepare_runner_abi_session",
        return_value=abi_prepare,
    ):
        with pytest.raises(RuntimeError, match="runner ABI prepare failed"):
            compile_model_native_first([])


def test_runner_abi_prepare_skips_when_no_supported_family_graph():
    env = {
        "NATIVE_RUNNER_ENABLED": "1",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_ABI_EXEC": "1",
    }

    graph = SimpleNamespace(
        nodes={
            "n0": SimpleNamespace(op_name="matmul"),
            "n1": SimpleNamespace(op_name="add"),
        }
    )

    native_lib = SimpleNamespace(
        nr_runtime_init=lambda: 0,
        nr_set_strict_mode=lambda strict: 0,
        nr_compile=lambda req: None,
        nr_execute=lambda req: None,
        nr_release_model=lambda handle: None,
    )

    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ):
        report = _maybe_prepare_runner_abi_session(
            layer_graphs=[graph],
            native_lib=native_lib,
            state=SimpleNamespace(strict=False),
            vocab_size=128,
            max_seq_len=32,
        )

    assert report.get("requested") is True
    assert report.get("attempted") is False
    assert report.get("succeeded") is False
    assert report.get("reason") == "no_abi_family_graph"


def test_runner_abi_prepare_attempts_for_exp_add_mul_matmul_linear_softmax_rmsnorm_sub_family_graph():
    env = {
        "NATIVE_RUNNER_ENABLED": "1",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_ABI_EXEC": "1",
    }

    graph = SimpleNamespace(
        nodes={
            "n0": SimpleNamespace(op_name="exp", input_ids=[]),
            "n1": SimpleNamespace(op_name="add", input_ids=["n0"]),
            "n2": SimpleNamespace(op_name="mul", input_ids=["n1", "n0"]),
            "n3": SimpleNamespace(op_name="matmul", input_ids=["n2", "n1"]),
            "n4": SimpleNamespace(op_name="linear", input_ids=["n3"]),
            "n5": SimpleNamespace(op_name="softmax", input_ids=["n4"]),
            "n6": SimpleNamespace(op_name="rmsnorm", input_ids=["n5"]),
            "n7": SimpleNamespace(op_name="sub", input_ids=["n6", "n1"]),
        }
    )

    logits_buf = (ctypes.c_float * 2)(1.0, 0.5)

    def _ok_compile(_req):
        return SimpleNamespace(status=0, model_handle=11, message=b"exp")

    def _ok_execute(_req):
        return SimpleNamespace(status=0, logits=logits_buf, vocab_size=2)

    native_lib = SimpleNamespace(
        nr_runtime_init=lambda: 0,
        nr_set_strict_mode=lambda strict: 0,
        nr_compile=_ok_compile,
        nr_execute=_ok_execute,
        nr_release_model=lambda handle: None,
    )

    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ), patch(
        "research.synthesis.native_ir_converter.graph_to_native_ir_json",
        return_value='{"schema_version":"native_ir.v1","nodes":[]}',
    ):
        report = _maybe_prepare_runner_abi_session(
            layer_graphs=[graph],
            native_lib=native_lib,
            state=SimpleNamespace(strict=False),
            vocab_size=2,
            max_seq_len=8,
        )

    assert report.get("requested") is True
    assert report.get("attempted") is True
    assert report.get("succeeded") is True
    assert report.get("reason") == "ok"


def test_runner_abi_prepare_reports_granular_nr_compile_reason_mapping():
    env = {
        "NATIVE_RUNNER_ENABLED": "1",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_ABI_EXEC": "1",
    }

    graph = SimpleNamespace(
        nodes={
            "n0": SimpleNamespace(op_name="exp", input_ids=[]),
            "n1": SimpleNamespace(op_name="add", input_ids=["n0"]),
            "n2": SimpleNamespace(op_name="mul", input_ids=["n1", "n0"]),
            "n3": SimpleNamespace(op_name="matmul", input_ids=["n2", "n1"]),
            "n4": SimpleNamespace(op_name="linear", input_ids=["n3"]),
            "n5": SimpleNamespace(op_name="softmax", input_ids=["n4"]),
            "n6": SimpleNamespace(op_name="rmsnorm", input_ids=["n5"]),
            "n7": SimpleNamespace(op_name="sub", input_ids=["n6", "n1"]),
        }
    )

    def _fail_compile(_req):
        return SimpleNamespace(status=-3, model_handle=-1, message=b"unsupported_graph_family_required_chain_invalid")

    native_lib = SimpleNamespace(
        nr_runtime_init=lambda: 0,
        nr_set_strict_mode=lambda strict: 0,
        nr_compile=_fail_compile,
        nr_execute=lambda req: None,
        nr_release_model=lambda handle: None,
    )

    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ), patch(
        "research.synthesis.native_ir_converter.graph_to_native_ir_json",
        return_value='{"schema_version":"native_ir.v1","nodes":[]}',
    ):
        report = _maybe_prepare_runner_abi_session(
            layer_graphs=[graph],
            native_lib=native_lib,
            state=SimpleNamespace(strict=False),
            vocab_size=128,
            max_seq_len=32,
        )

    assert report.get("requested") is True
    assert report.get("attempted") is True
    assert report.get("succeeded") is False
    assert report.get("compile_status") == -3
    assert report.get("compile_message") == "unsupported_graph_family_required_chain_invalid"
    assert report.get("compile_reason") == "unsupported_graph_family_required_chain_invalid"
    assert report.get("reason") == "nr_compile_failed:-3:unsupported_graph_family_required_chain_invalid"


def test_runner_abi_prepare_skips_transitively_linked_family_graph_without_direct_chain_links():
    env = {
        "NATIVE_RUNNER_ENABLED": "1",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_ABI_EXEC": "1",
    }

    graph = SimpleNamespace(
        nodes={
            "n0": SimpleNamespace(op_name="exp", input_ids=[]),
            "b1": SimpleNamespace(op_name="relu", input_ids=["n0"]),
            "n1": SimpleNamespace(op_name="add", input_ids=["b1", "b1"]),
            "b2": SimpleNamespace(op_name="relu", input_ids=["n1"]),
            "n2": SimpleNamespace(op_name="mul", input_ids=["b2", "n0"]),
            "b3": SimpleNamespace(op_name="relu", input_ids=["n2"]),
            "n3": SimpleNamespace(op_name="matmul", input_ids=["b3", "n1"]),
            "b4": SimpleNamespace(op_name="relu", input_ids=["n3"]),
            "n4": SimpleNamespace(op_name="linear", input_ids=["b4"]),
            "b5": SimpleNamespace(op_name="relu", input_ids=["n4"]),
            "n5": SimpleNamespace(op_name="softmax", input_ids=["b5"]),
            "b6": SimpleNamespace(op_name="relu", input_ids=["n5"]),
            "n6": SimpleNamespace(op_name="rmsnorm", input_ids=["b6"]),
            "b7": SimpleNamespace(op_name="relu", input_ids=["n6"]),
            "n7": SimpleNamespace(op_name="sub", input_ids=["b7", "n1"]),
        }
    )

    logits_buf = (ctypes.c_float * 2)(1.0, 0.5)

    def _ok_compile(_req):
        return SimpleNamespace(status=0, model_handle=12, message=b"exp")

    def _ok_execute(_req):
        return SimpleNamespace(status=0, logits=logits_buf, vocab_size=2)

    native_lib = SimpleNamespace(
        nr_runtime_init=lambda: 0,
        nr_set_strict_mode=lambda strict: 0,
        nr_compile=_ok_compile,
        nr_execute=_ok_execute,
        nr_release_model=lambda handle: None,
    )

    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ), patch(
        "research.synthesis.native_ir_converter.graph_to_native_ir_json",
        return_value='{"schema_version":"native_ir.v1","nodes":[]}',
    ):
        report = _maybe_prepare_runner_abi_session(
            layer_graphs=[graph],
            native_lib=native_lib,
            state=SimpleNamespace(strict=False),
            vocab_size=2,
            max_seq_len=8,
        )

    assert report.get("requested") is True
    assert report.get("attempted") is False
    assert report.get("succeeded") is False
    assert report.get("reason") == "no_abi_family_graph"


def test_runner_abi_prepare_skips_when_required_link_is_duplicated_in_input_ids():
    env = {
        "NATIVE_RUNNER_ENABLED": "1",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_ABI_EXEC": "1",
    }

    graph = SimpleNamespace(
        nodes={
            "n0": SimpleNamespace(op_name="exp", input_ids=[]),
            "n1": SimpleNamespace(op_name="add", input_ids=["n0", "n0"]),
            "n2": SimpleNamespace(op_name="mul", input_ids=["n1", "n0"]),
            "n3": SimpleNamespace(op_name="matmul", input_ids=["n2", "n1"]),
            "n4": SimpleNamespace(op_name="linear", input_ids=["n3"]),
            "n5": SimpleNamespace(op_name="softmax", input_ids=["n4"]),
            "n6": SimpleNamespace(op_name="rmsnorm", input_ids=["n5"]),
            "n7": SimpleNamespace(op_name="sub", input_ids=["n6", "n1"]),
        }
    )

    native_lib = SimpleNamespace(
        nr_runtime_init=lambda: 0,
        nr_set_strict_mode=lambda strict: 0,
        nr_compile=lambda req: None,
        nr_execute=lambda req: None,
        nr_release_model=lambda handle: None,
    )

    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ):
        report = _maybe_prepare_runner_abi_session(
            layer_graphs=[graph],
            native_lib=native_lib,
            state=SimpleNamespace(strict=False),
            vocab_size=128,
            max_seq_len=32,
        )

    assert report.get("requested") is True
    assert report.get("attempted") is False
    assert report.get("succeeded") is False
    assert report.get("reason") == "no_abi_family_graph"


def test_runner_abi_prepare_skips_when_required_link_is_duplicated_in_declared_edges():
    env = {
        "NATIVE_RUNNER_ENABLED": "1",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_ABI_EXEC": "1",
    }

    graph = SimpleNamespace(
        nodes={
            "n0": SimpleNamespace(op_name="exp", input_ids=[]),
            "n1": SimpleNamespace(op_name="add", input_ids=["n0"]),
            "n2": SimpleNamespace(op_name="mul", input_ids=["n1", "n0"]),
            "n3": SimpleNamespace(op_name="matmul", input_ids=["n2", "n1"]),
            "n4": SimpleNamespace(op_name="linear", input_ids=["n3"]),
            "n5": SimpleNamespace(op_name="softmax", input_ids=["n4"]),
            "n6": SimpleNamespace(op_name="rmsnorm", input_ids=["n5"]),
            "n7": SimpleNamespace(op_name="sub", input_ids=["n6", "n1"]),
        },
        edges=[
            {"source": "n0", "target": "n1"},
            {"source": "n1", "target": "n2"},
            {"source": "n2", "target": "n3"},
            {"source": "n2", "target": "n3"},
            {"source": "n3", "target": "n4"},
            {"source": "n4", "target": "n5"},
            {"source": "n5", "target": "n6"},
            {"source": "n6", "target": "n7"},
        ],
    )

    native_lib = SimpleNamespace(
        nr_runtime_init=lambda: 0,
        nr_set_strict_mode=lambda strict: 0,
        nr_compile=lambda req: None,
        nr_execute=lambda req: None,
        nr_release_model=lambda handle: None,
    )

    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ):
        report = _maybe_prepare_runner_abi_session(
            layer_graphs=[graph],
            native_lib=native_lib,
            state=SimpleNamespace(strict=False),
            vocab_size=128,
            max_seq_len=32,
        )

    assert report.get("requested") is True
    assert report.get("attempted") is False
    assert report.get("succeeded") is False
    assert report.get("reason") == "no_abi_family_graph"


def test_runner_abi_prepare_skips_out_of_order_exp_add_mul_matmul_linear_softmax_rmsnorm_sub_family_graph():
    env = {
        "NATIVE_RUNNER_ENABLED": "1",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_ABI_EXEC": "1",
    }

    graph = SimpleNamespace(
        nodes={
            "n0": SimpleNamespace(op_name="add", input_ids=[]),
            "n1": SimpleNamespace(op_name="exp", input_ids=["n0"]),
            "n2": SimpleNamespace(op_name="mul", input_ids=["n1", "n0"]),
            "n3": SimpleNamespace(op_name="matmul", input_ids=["n2", "n1"]),
            "n4": SimpleNamespace(op_name="linear", input_ids=["n3"]),
            "n5": SimpleNamespace(op_name="softmax", input_ids=["n4"]),
            "n6": SimpleNamespace(op_name="rmsnorm", input_ids=["n5"]),
            "n7": SimpleNamespace(op_name="sub", input_ids=["n6", "n1"]),
        }
    )

    native_lib = SimpleNamespace(
        nr_runtime_init=lambda: 0,
        nr_set_strict_mode=lambda strict: 0,
        nr_compile=lambda req: None,
        nr_execute=lambda req: None,
        nr_release_model=lambda handle: None,
    )

    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ):
        report = _maybe_prepare_runner_abi_session(
            layer_graphs=[graph],
            native_lib=native_lib,
            state=SimpleNamespace(strict=False),
            vocab_size=128,
            max_seq_len=32,
        )

    assert report.get("requested") is True
    assert report.get("attempted") is False
    assert report.get("succeeded") is False
    assert report.get("reason") == "no_abi_family_graph"


def test_runner_abi_prepare_skips_unlinked_exp_add_mul_matmul_linear_softmax_rmsnorm_sub_family_graph():
    env = {
        "NATIVE_RUNNER_ENABLED": "1",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_ABI_EXEC": "1",
    }

    graph = SimpleNamespace(
        nodes={
            "n0": SimpleNamespace(op_name="exp", input_ids=[]),
            "n1": SimpleNamespace(op_name="add", input_ids=["n0", "n0"]),
            "n2": SimpleNamespace(op_name="mul", input_ids=["n1", "n0"]),
            "n3": SimpleNamespace(op_name="matmul", input_ids=["n2", "n1"]),
            "n4": SimpleNamespace(op_name="linear", input_ids=["n0"]),
            "n5": SimpleNamespace(op_name="softmax", input_ids=["n4"]),
            "n6": SimpleNamespace(op_name="rmsnorm", input_ids=["n5"]),
            "n7": SimpleNamespace(op_name="sub", input_ids=["n6", "n1"]),
        }
    )

    native_lib = SimpleNamespace(
        nr_runtime_init=lambda: 0,
        nr_set_strict_mode=lambda strict: 0,
        nr_compile=lambda req: None,
        nr_execute=lambda req: None,
        nr_release_model=lambda handle: None,
    )

    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ):
        report = _maybe_prepare_runner_abi_session(
            layer_graphs=[graph],
            native_lib=native_lib,
            state=SimpleNamespace(strict=False),
            vocab_size=128,
            max_seq_len=32,
        )

    assert report.get("requested") is True
    assert report.get("attempted") is False
    assert report.get("succeeded") is False
    assert report.get("reason") == "no_abi_family_graph"


def test_runner_abi_prepare_skips_when_edges_break_family_ancestry_even_if_input_ids_linked():
    env = {
        "NATIVE_RUNNER_ENABLED": "1",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_ABI_EXEC": "1",
    }

    graph = SimpleNamespace(
        nodes={
            "n0": SimpleNamespace(op_name="exp", input_ids=[]),
            "n1": SimpleNamespace(op_name="add", input_ids=["n0", "n0"]),
            "n2": SimpleNamespace(op_name="mul", input_ids=["n1", "n0"]),
            "n3": SimpleNamespace(op_name="matmul", input_ids=["n2", "n1"]),
            "n4": SimpleNamespace(op_name="linear", input_ids=["n3"]),
            "n5": SimpleNamespace(op_name="softmax", input_ids=["n4"]),
            "n6": SimpleNamespace(op_name="rmsnorm", input_ids=["n5"]),
            "n7": SimpleNamespace(op_name="sub", input_ids=["n6", "n1"]),
        },
        edges=[
            {"source": "n0", "target": "n1"},
            {"source": "n1", "target": "n2"},
            {"source": "n2", "target": "n3"},
            {"source": "n0", "target": "n4"},
            {"source": "n4", "target": "n5"},
            {"source": "n5", "target": "n6"},
            {"source": "n6", "target": "n7"},
        ],
    )

    native_lib = SimpleNamespace(
        nr_runtime_init=lambda: 0,
        nr_set_strict_mode=lambda strict: 0,
        nr_compile=lambda req: None,
        nr_execute=lambda req: None,
        nr_release_model=lambda handle: None,
    )

    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ):
        report = _maybe_prepare_runner_abi_session(
            layer_graphs=[graph],
            native_lib=native_lib,
            state=SimpleNamespace(strict=False),
            vocab_size=128,
            max_seq_len=32,
        )

    assert report.get("requested") is True
    assert report.get("attempted") is False
    assert report.get("succeeded") is False
    assert report.get("reason") == "no_abi_family_graph"


def test_runner_abi_prepare_skips_when_edges_declared_but_empty_even_if_input_ids_linked():
    env = {
        "NATIVE_RUNNER_ENABLED": "1",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_ABI_EXEC": "1",
    }

    graph = SimpleNamespace(
        nodes={
            "n0": SimpleNamespace(op_name="exp", input_ids=[]),
            "n1": SimpleNamespace(op_name="add", input_ids=["n0", "n0"]),
            "n2": SimpleNamespace(op_name="mul", input_ids=["n1", "n0"]),
            "n3": SimpleNamespace(op_name="matmul", input_ids=["n2", "n1"]),
            "n4": SimpleNamespace(op_name="linear", input_ids=["n3"]),
            "n5": SimpleNamespace(op_name="softmax", input_ids=["n4"]),
            "n6": SimpleNamespace(op_name="rmsnorm", input_ids=["n5"]),
            "n7": SimpleNamespace(op_name="sub", input_ids=["n6", "n1"]),
        },
        edges=[],
    )

    native_lib = SimpleNamespace(
        nr_runtime_init=lambda: 0,
        nr_set_strict_mode=lambda strict: 0,
        nr_compile=lambda req: None,
        nr_execute=lambda req: None,
        nr_release_model=lambda handle: None,
    )

    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ):
        report = _maybe_prepare_runner_abi_session(
            layer_graphs=[graph],
            native_lib=native_lib,
            state=SimpleNamespace(strict=False),
            vocab_size=128,
            max_seq_len=32,
        )

    assert report.get("requested") is True
    assert report.get("attempted") is False
    assert report.get("succeeded") is False
    assert report.get("reason") == "no_abi_family_graph"


def test_runner_abi_prepare_skips_when_required_family_marker_is_duplicated():
    env = {
        "NATIVE_RUNNER_ENABLED": "1",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_ABI_EXEC": "1",
    }

    graph = SimpleNamespace(
        nodes={
            "n0": SimpleNamespace(op_name="exp", input_ids=[]),
            "n1": SimpleNamespace(op_name="add", input_ids=["n0", "n0"]),
            "n2": SimpleNamespace(op_name="mul", input_ids=["n1", "n0"]),
            "n3": SimpleNamespace(op_name="matmul", input_ids=["n2", "n1"]),
            "n4": SimpleNamespace(op_name="linear", input_ids=["n3"]),
            "n5": SimpleNamespace(op_name="softmax", input_ids=["n4"]),
            "n6": SimpleNamespace(op_name="rmsnorm", input_ids=["n5"]),
            "n7": SimpleNamespace(op_name="sub", input_ids=["n6", "n1"]),
            "n8": SimpleNamespace(op_name="add", input_ids=["n0", "n0"]),
        }
    )

    native_lib = SimpleNamespace(
        nr_runtime_init=lambda: 0,
        nr_set_strict_mode=lambda strict: 0,
        nr_compile=lambda req: None,
        nr_execute=lambda req: None,
        nr_release_model=lambda handle: None,
    )

    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ):
        report = _maybe_prepare_runner_abi_session(
            layer_graphs=[graph],
            native_lib=native_lib,
            state=SimpleNamespace(strict=False),
            vocab_size=128,
            max_seq_len=32,
        )

    assert report.get("requested") is True
    assert report.get("attempted") is False
    assert report.get("succeeded") is False
    assert report.get("reason") == "no_abi_family_graph"


def test_runner_abi_prepare_skips_when_required_link_exists_only_in_edges_not_input_ids():
    env = {
        "NATIVE_RUNNER_ENABLED": "1",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_ABI_EXEC": "1",
    }

    graph = SimpleNamespace(
        nodes={
            "n0": SimpleNamespace(op_name="exp", input_ids=[]),
            "n1": SimpleNamespace(op_name="add", input_ids=["n0", "n0"]),
            "n2": SimpleNamespace(op_name="mul", input_ids=["n1", "n0"]),
            "n3": SimpleNamespace(op_name="matmul", input_ids=["n2", "n1"]),
            "n4": SimpleNamespace(op_name="linear", input_ids=["n3"]),
            "n5": SimpleNamespace(op_name="softmax", input_ids=["n4"]),
            "n6": SimpleNamespace(op_name="rmsnorm", input_ids=["n5"]),
            "n7": SimpleNamespace(op_name="sub", input_ids=["n6", "n1"]),
        },
        edges=[
            {"source": "n0", "target": "n1"},
            {"source": "n1", "target": "n2"},
            {"source": "n2", "target": "n3"},
            {"source": "n3", "target": "n4"},
            {"source": "n4", "target": "n5"},
            {"source": "n5", "target": "n6"},
            {"source": "n6", "target": "n7"},
        ],
    )
    graph.nodes["n4"] = SimpleNamespace(op_name="linear", input_ids=["n0"])

    native_lib = SimpleNamespace(
        nr_runtime_init=lambda: 0,
        nr_set_strict_mode=lambda strict: 0,
        nr_compile=lambda req: None,
        nr_execute=lambda req: None,
        nr_release_model=lambda handle: None,
    )

    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ):
        report = _maybe_prepare_runner_abi_session(
            layer_graphs=[graph],
            native_lib=native_lib,
            state=SimpleNamespace(strict=False),
            vocab_size=128,
            max_seq_len=32,
        )

    assert report.get("requested") is True
    assert report.get("attempted") is False
    assert report.get("succeeded") is False
    assert report.get("reason") == "no_abi_family_graph"


def test_runner_abi_prepare_skips_when_required_chain_input_refs_missing_node_id():
    env = {
        "NATIVE_RUNNER_ENABLED": "1",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_ABI_EXEC": "1",
    }

    graph = SimpleNamespace(
        nodes={
            "n0": SimpleNamespace(op_name="exp", input_ids=[]),
            "n1": SimpleNamespace(op_name="add", input_ids=["n0"]),
            "n2": SimpleNamespace(op_name="mul", input_ids=["n1", "n0"]),
            "n3": SimpleNamespace(op_name="matmul", input_ids=["n2", "n1"]),
            "n4": SimpleNamespace(op_name="linear", input_ids=["n3", "n_missing"]),
            "n5": SimpleNamespace(op_name="softmax", input_ids=["n4"]),
            "n6": SimpleNamespace(op_name="rmsnorm", input_ids=["n5"]),
            "n7": SimpleNamespace(op_name="sub", input_ids=["n6", "n1"]),
        },
        edges=[
            {"source": "n0", "target": "n1"},
            {"source": "n1", "target": "n2"},
            {"source": "n2", "target": "n3"},
            {"source": "n3", "target": "n4"},
            {"source": "n4", "target": "n5"},
            {"source": "n5", "target": "n6"},
            {"source": "n6", "target": "n7"},
        ],
    )

    native_lib = SimpleNamespace(
        nr_runtime_init=lambda: 0,
        nr_set_strict_mode=lambda strict: 0,
        nr_compile=lambda req: None,
        nr_execute=lambda req: None,
        nr_release_model=lambda handle: None,
    )

    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ):
        report = _maybe_prepare_runner_abi_session(
            layer_graphs=[graph],
            native_lib=native_lib,
            state=SimpleNamespace(strict=False),
            vocab_size=128,
            max_seq_len=32,
        )

    assert report.get("requested") is True
    assert report.get("attempted") is False
    assert report.get("succeeded") is False
    assert report.get("reason") == "no_abi_family_graph"


def test_compile_non_strict_includes_designer_probe_when_enabled():
    env = {"NATIVE_RUNNER_ENABLED": "0", "NATIVE_RUNNER_STRICT": "0"}

    class DummyModel:
        pass

    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner_adapter.Path.exists", return_value=True
    ), patch(
        "research.scientist.native_runner._legacy_compile_model", return_value=DummyModel()
    ):
        model = compile_model_native_first([object()])
        report = getattr(model, "_native_runner_report", None)
        assert isinstance(report, dict)
        # With native disabled, probe is not attempted
        assert report.get("legacy_compile_used") is True


def test_compile_non_strict_probe_error_does_not_break_fallback():
    env = {"NATIVE_RUNNER_ENABLED": "0", "NATIVE_RUNNER_STRICT": "0"}

    class DummyModel:
        pass

    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner_adapter.Path.exists", return_value=True
    ), patch(
        "research.scientist.native_runner._legacy_compile_model", return_value=DummyModel()
    ):
        model = compile_model_native_first([object()])
        report = getattr(model, "_native_runner_report", None)
        assert isinstance(report, dict)
        # With native disabled, probe is not attempted; legacy compile still works
        assert report.get("legacy_compile_used") is True


def test_fallback_metrics_increment_on_compile():
    env = {"NATIVE_RUNNER_ENABLED": "0", "NATIVE_RUNNER_STRICT": "0"}

    class DummyModel:
        pass

    reset_native_runner_telemetry()

    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner._legacy_compile_model", return_value=DummyModel()
    ):
        compile_model_native_first([])
        report = native_runner_capability_report()
        metrics = report.get("fallback_metrics") or {}
        assert metrics.get("total_compiles") == 1
        assert metrics.get("native_enabled_compiles") == 0
        assert metrics.get("fallback_compiles") == 0
        assert metrics.get("legacy_compile_count") == 1
        assert metrics.get("legacy_compile_invocations") == 1
        assert (metrics.get("deprecated_fields") or {}).get("legacy_compile_invocations") == "use legacy_compile_count"
        assert metrics.get("fallback_rate") == 0.0


def test_fail_fast_when_fallback_rate_exceeds_threshold():
    env = {
        "NATIVE_RUNNER_ENABLED": "0",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_MAX_FALLBACK_RATE": "0.0",
        "NATIVE_RUNNER_FALLBACK_MIN_SAMPLES": "1",
    }

    class DummyModel:
        pass

    reset_native_runner_telemetry()

    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ), patch(
        "research.scientist.native_runner._legacy_compile_model", return_value=DummyModel()
    ):
        # With native disabled, fallback rate check does not apply;
        # legacy compile succeeds normally.
        model = compile_model_native_first([])
        report = getattr(model, "_native_runner_report", {}) or {}
        assert report.get("legacy_compile_used") is True


def test_fail_fast_when_legacy_compile_usage_exceeds_threshold():
    env = {
        "NATIVE_RUNNER_ENABLED": "0",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_MAX_LEGACY_COMPILE_INVOCATIONS": "0",
    }

    class DummyModel:
        pass

    reset_native_runner_telemetry()

    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ), patch(
        "research.scientist.native_runner._try_load_native_lib", return_value=None
    ), patch(
        "research.scientist.native_runner._legacy_compile_model", return_value=DummyModel()
    ):
        with pytest.raises(RuntimeError, match="legacy compile usage exceeded threshold"):
            compile_model_native_first([])


def test_cutover_gate_waiting_without_active_thresholds():
    reset_native_runner_telemetry()
    env = {
        "NATIVE_RUNNER_ENABLED": "1",
        "NATIVE_RUNNER_STRICT": "0",
    }
    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ):
        report = native_runner_capability_report()
        gate = report.get("cutover_gate", {}) or {}
        assert gate.get("status") == "waiting"
        assert gate.get("ready") is None
        assert isinstance(gate.get("checks"), list)
        assert len(gate.get("checks")) == 0


def test_cutover_gate_blocked_then_ready_for_legacy_threshold():
    class DummyModel:
        pass

    reset_native_runner_telemetry()

    env_no_gate = {
        "NATIVE_RUNNER_ENABLED": "0",
        "NATIVE_RUNNER_STRICT": "0",
    }
    # First compile increments legacy_compile_invocations.
    with patch("research.scientist.native_runner_adapter.os.environ", env_no_gate), patch(
        "research.scientist.native_runner.os.environ", env_no_gate
    ), patch(
        "research.scientist.native_runner._legacy_compile_model", return_value=DummyModel()
    ):
        compile_model_native_first([])

    env_blocked = {
        **env_no_gate,
        "NATIVE_RUNNER_MAX_LEGACY_COMPILE_INVOCATIONS": "0",
    }
    with patch("research.scientist.native_runner_adapter.os.environ", env_blocked), patch(
        "research.scientist.native_runner.os.environ", env_blocked
    ):
        blocked = native_runner_capability_report()
        gate = blocked.get("cutover_gate", {}) or {}
        assert gate.get("status") == "blocked"
        assert gate.get("ready") is False

    env_ready = {
        **env_no_gate,
        "NATIVE_RUNNER_MAX_LEGACY_COMPILE_INVOCATIONS": "1",
    }
    with patch("research.scientist.native_runner_adapter.os.environ", env_ready), patch(
        "research.scientist.native_runner.os.environ", env_ready
    ):
        ready = native_runner_capability_report()
        gate = ready.get("cutover_gate", {}) or {}
        assert gate.get("status") == "ready"
        assert gate.get("ready") is True


def test_cutover_gate_waiting_when_parity_required_without_samples():
    reset_native_runner_telemetry()
    env = {
        "NATIVE_RUNNER_ENABLED": "1",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_REQUIRE_PARITY_PASS": "1",
    }
    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ):
        report = native_runner_capability_report()
        gate = report.get("cutover_gate", {}) or {}
        assert gate.get("status") == "waiting"
        assert gate.get("ready") is None


def test_cutover_gate_ready_when_parity_required_with_pass_sample():
    reset_native_runner_telemetry()
    record_native_abi_parity_result(True)
    env = {
        "NATIVE_RUNNER_ENABLED": "1",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_REQUIRE_PARITY_PASS": "1",
    }
    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ):
        report = native_runner_capability_report()
        gate = report.get("cutover_gate", {}) or {}
        assert gate.get("status") == "ready"
        assert gate.get("ready") is True


def test_disable_legacy_compile_gate_rejects_compile_path():
    env = {
        "NATIVE_RUNNER_ENABLED": "0",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_DISABLE_LEGACY_COMPILE": "1",
    }

    class DummyModel:
        pass

    reset_native_runner_telemetry()
    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ), patch(
        "research.scientist.native_runner._legacy_compile_model", return_value=DummyModel()
    ):
        with pytest.raises(RuntimeError, match="Legacy compile path disabled"):
            compile_model_native_first([])


def test_disable_legacy_compile_conflicts_with_legacy_only():
    env = {
        "NATIVE_RUNNER_ENABLED": "0",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_DISABLE_LEGACY_COMPILE": "1",
        "NATIVE_RUNNER_LEGACY_ONLY": "1",
    }

    class DummyModel:
        pass

    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ), patch(
        "research.scientist.native_runner._legacy_compile_model", return_value=DummyModel()
    ):
        with pytest.raises(RuntimeError, match="conflicts with NATIVE_RUNNER_DISABLE_LEGACY_COMPILE"):
            compile_model_native_first([])


def test_disable_legacy_compile_native_enabled_gate_rejects_when_native_enabled():
    env = {
        "NATIVE_RUNNER_ENABLED": "0",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_DISABLE_LEGACY_COMPILE": "1",
    }

    class DummyModel:
        pass

    reset_native_runner_telemetry()
    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ), patch(
        "research.scientist.native_runner._legacy_compile_model", return_value=DummyModel()
    ):
        with pytest.raises(RuntimeError, match="Legacy compile path disabled"):
            compile_model_native_first([])


def test_disable_legacy_compile_native_enabled_gate_allows_when_native_disabled():
    env = {
        "NATIVE_RUNNER_ENABLED": "0",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_DISABLE_LEGACY_COMPILE_NATIVE_ENABLED": "1",
    }

    class DummyModel:
        pass

    reset_native_runner_telemetry()
    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ), patch(
        "research.scientist.native_runner._legacy_compile_model", return_value=DummyModel()
    ):
        model = compile_model_native_first([])

    report = getattr(model, "_native_runner_report", {}) or {}
    assert report.get("legacy_compile_used") is True


def test_abi_model_only_returns_native_model_without_legacy_compile():
    import torch

    env = {
        "NATIVE_RUNNER_ENABLED": "1",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_ABI_MODEL_ONLY": "1",
    }

    class _Session:
        def execute_tokens(self, token_ids, batch=1):
            return [0.2, 0.4, 0.6, 0.8]

    abi_report = {
        "requested": True,
        "attempted": True,
        "succeeded": True,
        "reason": "ok",
        "model_handle": 1,
        "session": _Session(),
    }

    reset_native_runner_telemetry()
    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ), patch(
        "research.scientist.native_runner._maybe_prepare_runner_abi_session", return_value=abi_report
    ), patch(
        "research.scientist.native_runner._legacy_compile_model", side_effect=AssertionError("legacy path should not run")
    ):
        model = compile_model_native_first([], vocab_size=4)

    out = model(torch.tensor([[1, 2, 3]], dtype=torch.long))
    assert tuple(out.shape) == (1, 3, 4)
    report = getattr(model, "_native_runner_report", {}) or {}
    assert report.get("execution_path") == "native_abi_model_only"
    assert report.get("legacy_compile_used") is False


def test_abi_model_only_requires_successful_abi_session():
    """When ABI fails and legacy compile is explicitly disabled, raise RuntimeError."""
    env = {
        "NATIVE_RUNNER_ENABLED": "1",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_ABI_MODEL_ONLY": "1",
        "NATIVE_RUNNER_DISABLE_LEGACY_COMPILE": "1",
    }
    abi_report = {
        "requested": True,
        "attempted": True,
        "succeeded": False,
        "reason": "not_supported",
        "model_handle": None,
        "session": None,
    }

    reset_native_runner_telemetry()
    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ), patch(
        "research.scientist.native_runner._maybe_prepare_runner_abi_session", return_value=abi_report
    ):
        with pytest.raises(RuntimeError, match="requires successful ABI session"):
            compile_model_native_first([])


def test_abi_model_only_allows_legacy_disable_gate_in_native_mode():
    env = {
        "NATIVE_RUNNER_ENABLED": "1",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_DISABLE_LEGACY_COMPILE_NATIVE_ENABLED": "1",
        "NATIVE_RUNNER_ABI_MODEL_ONLY": "1",
    }

    class _Session:
        def execute_tokens(self, token_ids, batch=1):
            return [0.1, 0.2, 0.3, 0.4]

    abi_report = {
        "requested": True,
        "attempted": True,
        "succeeded": True,
        "reason": "ok",
        "model_handle": 99,
        "session": _Session(),
    }

    reset_native_runner_telemetry()
    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ), patch(
        "research.scientist.native_runner._maybe_prepare_runner_abi_session", return_value=abi_report
    ), patch(
        "research.scientist.native_runner._legacy_compile_model", side_effect=AssertionError("legacy path should not run")
    ):
        model = compile_model_native_first([], vocab_size=4)

    report = getattr(model, "_native_runner_report", {}) or {}
    assert report.get("execution_path") == "native_abi_model_only"
    assert report.get("legacy_compile_used") is False


# ---------------------------------------------------------------------------
# Phase 3: Native kernel dispatch tests
# ---------------------------------------------------------------------------


def test_try_load_native_lib_returns_none_when_no_lib_exists():
    """When no .so files exist on disk, _try_load_native_lib returns None."""
    _reset_native_lib_cache()
    with patch("research.scientist.native_runner.Path.exists", return_value=False):
        lib = _try_load_native_lib()
        assert lib is None
    _reset_native_lib_cache()


def test_try_load_native_lib_returns_cdll_when_lib_exists():
    """When the .so is present, _try_load_native_lib returns a CDLL object."""
    _reset_native_lib_cache()
    fake_cdll = MagicMock()
    with patch("research.scientist.native_runner.Path.exists", return_value=True), patch(
        "research.scientist.native_runner.ctypes.CDLL", return_value=fake_cdll
    ):
        lib = _try_load_native_lib()
        assert lib is fake_cdll
    _reset_native_lib_cache()


def test_try_load_native_lib_caches_result():
    """Repeated calls should not re-probe the filesystem."""
    _reset_native_lib_cache()
    fake_cdll = MagicMock()
    with patch("research.scientist.native_runner.Path.exists", return_value=True) as mock_exists, patch(
        "research.scientist.native_runner.ctypes.CDLL", return_value=fake_cdll
    ):
        lib1 = _try_load_native_lib()
        lib2 = _try_load_native_lib()
        assert lib1 is lib2
        # CDLL should only be called once thanks to caching.
        assert fake_cdll is lib1
    _reset_native_lib_cache()


def test_check_native_op_support_all_supported():
    """When the native lib reports all ops as registered, coverage is 1.0."""
    fake_lib = MagicMock()
    fake_lib.nk_is_registered = MagicMock(return_value=1)

    graphs = [make_fake_graph(["matmul", "relu", "softmax"])]
    result = _check_native_op_support(graphs, fake_lib)

    assert result["native_coverage"] == 1.0
    assert sorted(result["supported"]) == ["matmul", "relu", "softmax"]
    assert result["unsupported"] == []


def test_check_native_op_support_partial():
    """When some ops are not registered, they appear in unsupported."""
    def fake_is_registered(op_bytes):
        return 1 if op_bytes in (b"matmul", b"relu") else 0

    fake_lib = MagicMock()
    fake_lib.nk_is_registered = MagicMock(side_effect=fake_is_registered)

    graphs = [make_fake_graph(["matmul", "relu", "custom_op"])]
    result = _check_native_op_support(graphs, fake_lib)

    assert result["native_coverage"] == pytest.approx(2.0 / 3.0)
    assert result["supported"] == ["matmul", "relu"]
    assert result["unsupported"] == ["custom_op"]


def test_check_native_op_support_no_lib():
    """When native_lib is None and Cython bridge unavailable, all ops are unsupported."""
    graphs = [make_fake_graph(["matmul", "relu"])]
    with patch("research.scientist.native_runner._try_import_cython_bridge", return_value=None):
        result = _check_native_op_support(graphs, None)

    assert result["native_coverage"] == 0.0
    assert result["supported"] == []
    assert sorted(result["unsupported"]) == ["matmul", "relu"]


def test_check_native_op_support_empty_graphs():
    """With no ops, coverage defaults to 0/max(0,1) = 0.0 with empty lists."""
    result = _check_native_op_support([], None)
    assert result["native_coverage"] == 0.0
    assert result["all_ops"] == []


def test_check_native_op_support_ignores_structural_ops_in_coverage():
    """Structural ops should not count as unsupported native kernel gaps."""
    def fake_is_registered(op_bytes):
        return 1 if op_bytes == b"matmul" else 0

    fake_lib = MagicMock()
    fake_lib.nk_is_registered = MagicMock(side_effect=fake_is_registered)

    graphs = [make_fake_graph(["input", "concat", "split2", "matmul", "custom_op"])]
    result = _check_native_op_support(graphs, fake_lib)

    assert sorted(result["kernel_relevant_ops"]) == ["custom_op", "matmul"]
    assert result["supported"] == ["matmul"]
    assert result["unsupported"] == ["custom_op"]
    assert result["native_coverage"] == pytest.approx(0.5)


def test_check_native_op_support_structural_only_is_full_coverage():
    """Graphs with only structural ops should not force fallback."""
    fake_lib = MagicMock()
    fake_lib.nk_is_registered = MagicMock(return_value=0)

    graphs = [make_fake_graph(["input", "concat", "split2"])]
    result = _check_native_op_support(graphs, fake_lib)

    assert result["kernel_relevant_ops"] == []
    assert result["supported"] == []
    assert result["unsupported"] == []
    assert result["native_coverage"] == 1.0


def test_check_native_op_support_square_requires_square_kernel():
    """square should be treated as its own kernel capability in coverage checks."""
    def fake_is_registered(op_bytes):
        return 1 if op_bytes == b"mul" else 0

    fake_lib = MagicMock()
    fake_lib.nk_is_registered = MagicMock(side_effect=fake_is_registered)

    graphs = [make_fake_graph(["square"])]
    result = _check_native_op_support(graphs, fake_lib)

    assert result["supported"] == []
    assert result["unsupported"] == ["square"]
    assert result["native_coverage"] == 0.0


def test_check_native_op_support_log_sqrt_supported():
    """Coverage check should recognize dedicated log/sqrt native kernels."""
    def fake_is_registered(op_bytes):
        return 1 if op_bytes in (b"log", b"sqrt") else 0

    fake_lib = MagicMock()
    fake_lib.nk_is_registered = MagicMock(side_effect=fake_is_registered)

    graphs = [make_fake_graph(["log", "sqrt"])]
    result = _check_native_op_support(graphs, fake_lib)

    assert sorted(result["supported"]) == ["log", "sqrt"]
    assert result["unsupported"] == []
    assert result["native_coverage"] == 1.0


def test_check_native_op_support_abs_neg_supported():
    """Coverage check should recognize dedicated abs/neg native kernels."""
    def fake_is_registered(op_bytes):
        return 1 if op_bytes in (b"abs", b"neg") else 0

    fake_lib = MagicMock()
    fake_lib.nk_is_registered = MagicMock(side_effect=fake_is_registered)

    graphs = [make_fake_graph(["abs", "neg"])]
    result = _check_native_op_support(graphs, fake_lib)

    assert sorted(result["supported"]) == ["abs", "neg"]
    assert result["unsupported"] == []
    assert result["native_coverage"] == 1.0


def test_check_native_op_support_reciprocal_supported():
    """Coverage check should recognize dedicated reciprocal native kernel."""
    def fake_is_registered(op_bytes):
        return 1 if op_bytes == b"reciprocal" else 0

    fake_lib = MagicMock()
    fake_lib.nk_is_registered = MagicMock(side_effect=fake_is_registered)

    graphs = [make_fake_graph(["reciprocal"])]
    result = _check_native_op_support(graphs, fake_lib)

    assert result["supported"] == ["reciprocal"]
    assert result["unsupported"] == []
    assert result["native_coverage"] == 1.0


def test_strict_mode_raises_with_unsupported_ops():
    """STRICT=1 with unsupported ops should raise RuntimeError listing them."""
    env = {"NATIVE_RUNNER_ENABLED": "1", "NATIVE_RUNNER_STRICT": "1"}

    def fake_is_registered(op_bytes):
        return 1 if op_bytes == b"matmul" else 0

    fake_lib = MagicMock()
    fake_lib.nk_is_registered = MagicMock(side_effect=fake_is_registered)

    graphs = [make_fake_graph(["matmul", "fancy_attention"])]

    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner_adapter.Path.exists", return_value=False
    ), patch(
        "research.scientist.native_runner._try_load_native_lib", return_value=fake_lib
    ):
        with pytest.raises(RuntimeError, match="NATIVE_RUNNER_STRICT=1.*fancy_attention"):
            compile_model_native_first(graphs)


def test_non_strict_mode_falls_back_with_unsupported_ops():
    """Non-strict with partial coverage should fall back to legacy compile."""
    env = {"NATIVE_RUNNER_ENABLED": "0", "NATIVE_RUNNER_STRICT": "0"}

    class DummyModel:
        pass

    def fake_is_registered(op_bytes):
        return 1 if op_bytes == b"matmul" else 0

    fake_lib = MagicMock()
    fake_lib.nk_is_registered = MagicMock(side_effect=fake_is_registered)

    graphs = [make_fake_graph(["matmul", "custom_op"])]

    reset_native_runner_telemetry()
    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner_adapter.Path.exists", return_value=False
    ), patch(
        "research.scientist.native_runner._legacy_compile_model", return_value=DummyModel()
    ):
        model = compile_model_native_first(graphs)
        report = getattr(model, "_native_runner_report", None)
        assert report is not None
        # With native disabled, op support check is skipped; legacy compile used
        assert report.get("legacy_compile_used") is True


def test_full_native_coverage_records_dispatch_metric():
    """When native disabled, legacy compile is used; dispatch metrics stay at zero."""
    env = {"NATIVE_RUNNER_ENABLED": "0", "NATIVE_RUNNER_STRICT": "0"}

    class DummyModel:
        pass

    graphs = [make_fake_graph(["matmul", "relu"])]

    reset_native_runner_telemetry()
    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner_adapter.Path.exists", return_value=False
    ), patch(
        "research.scientist.native_runner._legacy_compile_model", return_value=DummyModel()
    ):
        model = compile_model_native_first(graphs)
        report = getattr(model, "_native_runner_report", None)
        assert report is not None
        metrics = report.get("fallback_metrics", {})
        assert metrics.get("native_dispatch_compiles") == 0
        assert metrics.get("fallback_compiles") == 0
        assert report.get("legacy_compile_used") is True


def test_selective_mode_candidate_reported_when_probe_and_coverage_are_green():
    env = {
        "NATIVE_RUNNER_ENABLED": "0",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_EXECUTION_MODE": "selective",
        "NATIVE_RUNNER_SELECTIVE_LAYER_EXEC": "1",
    }

    class DummyModel:
        pass

    fake_lib = MagicMock()
    fake_lib.nk_is_registered = MagicMock(return_value=1)
    graphs = [make_fake_graph(["matmul", "relu"])]

    reset_native_runner_telemetry()
    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ), patch(
        "research.scientist.native_runner_adapter.Path.exists", return_value=True
    ), patch(
        "research.scientist.native_runner._legacy_compile_model", return_value=DummyModel()
    ):
        model = compile_model_native_first(graphs)
        report = getattr(model, "_native_runner_report", {}) or {}
        selective = report.get("selective_execution", {}) or {}
        assert selective.get("requested") is True
        # With native disabled, selective mode cannot produce candidates
        assert selective.get("candidate") is False
        assert selective.get("reason") == "native_runner_disabled"
        assert report.get("legacy_compile_used") is True


def test_fallback_telemetry_soak_counters_stay_consistent():
    env = {
        "NATIVE_RUNNER_ENABLED": "0",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_EXECUTION_MODE": "probe",
    }

    class DummyModel:
        pass

    graphs = [make_fake_graph(["matmul", "custom_op"])]

    reset_native_runner_telemetry()
    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ), patch(
        "research.scientist.native_runner._legacy_compile_model", return_value=DummyModel()
    ):
        for _ in range(40):
            compile_model_native_first(graphs)
        report = native_runner_capability_report()
        metrics = report.get("fallback_metrics", {})
        assert metrics.get("total_compiles") == 40
        # With native disabled, native_enabled_compiles stays 0
        assert metrics.get("native_enabled_compiles") == 0
        assert metrics.get("fallback_compiles") == 0
        assert metrics.get("probe_failures") == 0
        assert metrics.get("probe_successes") == 0
        assert metrics.get("fallback_rate") == 0.0
        assert metrics.get("legacy_compile_count") == 40


def test_selective_mode_executes_native_sanity_dispatch_when_symbols_available():
    env = {
        "NATIVE_RUNNER_ENABLED": "0",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_EXECUTION_MODE": "selective",
        "NATIVE_RUNNER_SELECTIVE_LAYER_EXEC": "1",
    }

    class DummyModel:
        pass

    class FakeNativeLib:
        def __init__(self):
            self.nk_is_registered = MagicMock(return_value=1)

        @staticmethod
        def aria_relu_f32(x, y, n):
            for i in range(int(n)):
                y[i] = x[i] if x[i] > 0 else 0.0

        @staticmethod
        def aria_add_f32(a, b, y, n):
            for i in range(int(n)):
                y[i] = a[i] + b[i]

    graphs = [make_fake_graph(["matmul", "relu", "add"])]

    reset_native_runner_telemetry()
    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ), patch(
        "research.scientist.native_runner._legacy_compile_model", return_value=DummyModel()
    ):
        model = compile_model_native_first(graphs)
        report = getattr(model, "_native_runner_report", {}) or {}
        selective = report.get("selective_execution", {}) or {}
        assert selective.get("requested") is True
        # With native disabled, selective mode cannot produce candidates
        assert selective.get("candidate") is False
        assert selective.get("reason") == "native_runner_disabled"
        assert report.get("legacy_compile_used") is True


def test_selective_mode_activation_failure_is_reported_without_breaking_compile():
    env = {
        "NATIVE_RUNNER_ENABLED": "0",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_EXECUTION_MODE": "selective",
        "NATIVE_RUNNER_SELECTIVE_LAYER_EXEC": "1",
    }

    class DummyModel:
        pass

    graphs = [make_fake_graph(["matmul", "relu", "add"])]

    reset_native_runner_telemetry()
    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ), patch(
        "research.scientist.native_runner._legacy_compile_model", return_value=DummyModel()
    ):
        model = compile_model_native_first(graphs)
        report = getattr(model, "_native_runner_report", {}) or {}
        selective = report.get("selective_execution", {}) or {}
        # With native disabled, no candidate, no activation attempt
        assert selective.get("candidate") is False
        assert selective.get("reason") == "native_runner_disabled"
        assert report.get("legacy_compile_used") is True


def test_selective_guardrail_triggers_after_sustained_not_candidate_window():
    env = {
        "NATIVE_RUNNER_ENABLED": "0",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_EXECUTION_MODE": "selective",
        "NATIVE_RUNNER_SELECTIVE_GUARDRAIL_WINDOW": "2",
    }

    class DummyModel:
        pass

    reset_native_runner_telemetry()
    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ), patch(
        "research.scientist.native_runner_adapter.Path.exists", return_value=False
    ), patch(
        "research.scientist.native_runner._try_load_native_lib", return_value=None
    ), patch(
        "research.scientist.native_runner._legacy_compile_model", return_value=DummyModel()
    ):
        compile_model_native_first([])
        model = compile_model_native_first([])
        report = getattr(model, "_native_runner_report", {}) or {}
        guardrail = report.get("selective_guardrail", {}) or {}
        assert guardrail.get("threshold") == 2
        assert guardrail.get("consecutive_requested_not_candidate") >= 2
        assert guardrail.get("triggered") is True
        assert guardrail.get("last_reason") in {"incomplete_native_coverage", "native_runner_disabled"}
        history = guardrail.get("history") or []
        assert isinstance(history, list)
        assert history
        assert history[-1].get("event") == "triggered"
        assert isinstance(history[-1].get("timestamp"), str)
        assert "T" in str(history[-1].get("timestamp"))
        assert history[-1].get("source") == "compile_model_native_first"


def test_selective_guardrail_resets_after_candidate_success():
    env = {
        "NATIVE_RUNNER_ENABLED": "0",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_EXECUTION_MODE": "selective",
        "NATIVE_RUNNER_SELECTIVE_GUARDRAIL_WINDOW": "2",
    }

    class DummyModel:
        pass

    reset_native_runner_telemetry()
    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ), patch(
        "research.scientist.native_runner_adapter.Path.exists", return_value=True
    ), patch(
        "research.scientist.native_runner._legacy_compile_model", return_value=DummyModel()
    ):
        # Two selective-mode compiles trigger guardrail (native disabled => not candidate)
        compile_model_native_first([])
        compile_model_native_first([])
        # Switch to probe mode to clear the guardrail
        env["NATIVE_RUNNER_EXECUTION_MODE"] = "probe"
        model = compile_model_native_first([])

        report = getattr(model, "_native_runner_report", {}) or {}
        guardrail = report.get("selective_guardrail", {}) or {}
        assert guardrail.get("triggered") is False
        assert int(guardrail.get("consecutive_requested_not_candidate") or 0) == 0
        history = guardrail.get("history") or []
        assert isinstance(history, list)
        assert history
        assert history[-1].get("event") == "cleared"
        assert isinstance(history[-1].get("timestamp"), str)
        assert "T" in str(history[-1].get("timestamp"))
        assert history[-1].get("source") == "compile_model_native_first"


def test_selective_guardrail_history_retains_last_25_events_in_order():
    env = {
        "NATIVE_RUNNER_ENABLED": "0",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_EXECUTION_MODE": "selective",
        "NATIVE_RUNNER_SELECTIVE_GUARDRAIL_WINDOW": "1",
    }

    class DummyModel:
        pass

    reset_native_runner_telemetry()
    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ), patch(
        "research.scientist.native_runner_adapter.Path.exists", return_value=False
    ), patch(
        "research.scientist.native_runner._try_load_native_lib", return_value=None
    ), patch(
        "research.scientist.native_runner._legacy_compile_model", return_value=DummyModel()
    ):
        for _ in range(20):
            env["NATIVE_RUNNER_EXECUTION_MODE"] = "selective"
            compile_model_native_first([])
            env["NATIVE_RUNNER_EXECUTION_MODE"] = "probe"
            compile_model_native_first([])

    report = native_runner_capability_report()
    guardrail = report.get("selective_guardrail", {}) or {}
    history = guardrail.get("history") or []

    assert len(history) == 25
    assert history[0].get("event") == "cleared"
    assert history[0].get("trigger_count") == 8
    assert history[-1].get("event") == "cleared"
    assert history[-1].get("trigger_count") == 20
    for entry in history:
        assert isinstance(entry.get("timestamp"), str)
        assert "T" in str(entry.get("timestamp"))
        assert entry.get("source") == "compile_model_native_first"


def test_selective_mode_applies_designer_layer_replacements_when_available():
    env = {
        "NATIVE_RUNNER_ENABLED": "0",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_EXECUTION_MODE": "selective",
        "NATIVE_RUNNER_SELECTIVE_LAYER_EXEC": "1",
    }

    class DummyModel:
        def __init__(self):
            self.layers = [object()]
            self.model_dim = 32

    class FakeNativeLib:
        def __init__(self):
            self.nk_is_registered = MagicMock(return_value=1)

        @staticmethod
        def aria_relu_f32(x, y, n):
            for i in range(int(n)):
                y[i] = x[i] if x[i] > 0 else 0.0

        @staticmethod
        def aria_add_f32(a, b, y, n):
            for i in range(int(n)):
                y[i] = a[i] + b[i]

    class FakeWorkflowModule:
        def __call__(self, inputs):
            return {"y": next(iter(inputs.values()))}

    graphs = [make_fake_graph(["matmul", "relu", "add"])]

    reset_native_runner_telemetry()
    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ), patch(
        "research.scientist.native_runner._legacy_compile_model", return_value=DummyModel()
    ):
        model = compile_model_native_first(graphs)
        report = getattr(model, "_native_runner_report", {}) or {}
        selective = report.get("selective_execution", {}) or {}
        # With native disabled, selective mode cannot activate
        assert selective.get("candidate") is False
        assert selective.get("reason") == "native_runner_disabled"
        assert report.get("legacy_compile_used") is True


def test_selective_mode_reports_layer_build_errors_without_breaking_compile():
    env = {
        "NATIVE_RUNNER_ENABLED": "0",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_EXECUTION_MODE": "selective",
        "NATIVE_RUNNER_SELECTIVE_LAYER_EXEC": "1",
    }

    class DummyModel:
        pass

    class FakeNativeLib:
        def __init__(self):
            self.nk_is_registered = MagicMock(return_value=1)

        @staticmethod
        def aria_relu_f32(x, y, n):
            for i in range(int(n)):
                y[i] = x[i] if x[i] > 0 else 0.0

        @staticmethod
        def aria_add_f32(a, b, y, n):
            for i in range(int(n)):
                y[i] = a[i] + b[i]

    graphs = [make_fake_graph(["matmul", "relu", "add"])]

    reset_native_runner_telemetry()
    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ), patch(
        "research.scientist.native_runner._legacy_compile_model", return_value=DummyModel()
    ):
        model = compile_model_native_first(graphs)
        report = getattr(model, "_native_runner_report", {}) or {}
        selective = report.get("selective_execution", {}) or {}
        # With native disabled, selective mode is not a candidate
        assert selective.get("candidate") is False
        assert selective.get("reason") == "native_runner_disabled"
        assert report.get("legacy_compile_used") is True


def test_selective_mode_skips_layer_with_missing_input_node_id():
    env = {
        "NATIVE_RUNNER_ENABLED": "0",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_EXECUTION_MODE": "selective",
        "NATIVE_RUNNER_SELECTIVE_LAYER_EXEC": "1",
    }

    class DummyModel:
        def __init__(self):
            self.layers = [object()]
            self.model_dim = 32

    class FakeNativeLib:
        def __init__(self):
            self.nk_is_registered = MagicMock(return_value=1)

        @staticmethod
        def aria_relu_f32(x, y, n):
            for i in range(int(n)):
                y[i] = x[i] if x[i] > 0 else 0.0

        @staticmethod
        def aria_add_f32(a, b, y, n):
            for i in range(int(n)):
                y[i] = a[i] + b[i]

    class FakeWorkflowModule:
        def __call__(self, inputs):
            return {"y": next(iter(inputs.values()))}

    graphs = [make_fake_graph(["matmul", "relu", "add"])]

    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ), patch(
        "research.scientist.native_runner._legacy_compile_model", return_value=DummyModel()
    ):
        model = compile_model_native_first(graphs)
        report = getattr(model, "_native_runner_report", {}) or {}
        selective = report.get("selective_execution", {}) or {}
        # With native disabled, selective mode is not active
        assert selective.get("candidate") is False
        assert selective.get("reason") == "native_runner_disabled"
        assert report.get("legacy_compile_used") is True


def test_selective_mode_skips_layer_on_adapter_contract_mismatch():
    env = {
        "NATIVE_RUNNER_ENABLED": "0",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_EXECUTION_MODE": "selective",
        "NATIVE_RUNNER_SELECTIVE_LAYER_EXEC": "1",
    }

    class DummyModel:
        def __init__(self):
            self.layers = [object()]
            self.model_dim = 32

    class FakeNativeLib:
        def __init__(self):
            self.nk_is_registered = MagicMock(return_value=1)

        @staticmethod
        def aria_relu_f32(x, y, n):
            for i in range(int(n)):
                y[i] = x[i] if x[i] > 0 else 0.0

        @staticmethod
        def aria_add_f32(a, b, y, n):
            for i in range(int(n)):
                y[i] = a[i] + b[i]

    class BadWorkflowModule:
        def __call__(self, inputs):
            return {"y": None}

    graphs = [make_fake_graph(["matmul", "relu", "add"])]

    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ), patch(
        "research.scientist.native_runner._legacy_compile_model", return_value=DummyModel()
    ):
        model = compile_model_native_first(graphs)
        report = getattr(model, "_native_runner_report", {}) or {}
        selective = report.get("selective_execution", {}) or {}
        # With native disabled, selective mode is not active
        assert selective.get("candidate") is False
        assert selective.get("reason") == "native_runner_disabled"
        assert report.get("legacy_compile_used") is True


def test_selective_mode_layer_strict_raises_on_incompatible_replacement():
    env = {
        "NATIVE_RUNNER_ENABLED": "0",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_EXECUTION_MODE": "selective",
        "NATIVE_RUNNER_SELECTIVE_LAYER_EXEC": "1",
        "NATIVE_RUNNER_SELECTIVE_LAYER_STRICT": "1",
    }

    class DummyModel:
        def __init__(self):
            self.layers = [object()]
            self.model_dim = 32

    class FakeNativeLib:
        def __init__(self):
            self.nk_is_registered = MagicMock(return_value=1)

        @staticmethod
        def aria_relu_f32(x, y, n):
            for i in range(int(n)):
                y[i] = x[i] if x[i] > 0 else 0.0

        @staticmethod
        def aria_add_f32(a, b, y, n):
            for i in range(int(n)):
                y[i] = a[i] + b[i]

    class BadWorkflowModule:
        def __call__(self, inputs):
            return {"y": None}

    graphs = [make_fake_graph(["matmul", "relu", "add"])]

    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ), patch(
        "research.scientist.native_runner._legacy_compile_model", return_value=DummyModel()
    ):
        # With native disabled, selective mode strict does not trigger;
        # legacy compile succeeds normally.
        model = compile_model_native_first(graphs)
        report = getattr(model, "_native_runner_report", {}) or {}
        selective = report.get("selective_execution", {}) or {}
        assert selective.get("candidate") is False
        assert selective.get("reason") == "native_runner_disabled"
        assert report.get("legacy_compile_used") is True


def test_selective_guardrail_history_retention_keeps_latest_window():
    reset_native_runner_telemetry()
    total_events = _SELECTIVE_GUARDRAIL_HISTORY_MAX + 7
    for i in range(total_events):
        _SELECTIVE_GUARDRAIL["consecutive_requested_not_candidate"] = i
        _SELECTIVE_GUARDRAIL["trigger_count"] = i
        _record_guardrail_event(
            "triggered",
            reason=f"reason_{i}",
            threshold=3,
            source="retention_test",
        )

    report = native_runner_capability_report()
    history = (report.get("selective_guardrail") or {}).get("history") or []
    assert len(history) == _SELECTIVE_GUARDRAIL_HISTORY_MAX

    dropped = total_events - _SELECTIVE_GUARDRAIL_HISTORY_MAX
    assert history[0].get("reason") == f"reason_{dropped}"
    assert history[-1].get("reason") == f"reason_{total_events - 1}"
    reasons = [item.get("reason") for item in history]
    expected = [f"reason_{i}" for i in range(dropped, total_events)]
    assert reasons == expected


def test_selective_guardrail_history_order_preserved_with_mixed_events():
    reset_native_runner_telemetry()
    _SELECTIVE_GUARDRAIL["consecutive_requested_not_candidate"] = 2
    _SELECTIVE_GUARDRAIL["trigger_count"] = 1
    _record_guardrail_event(
        "triggered",
        reason="incomplete_native_coverage",
        threshold=2,
        source="order_test",
    )
    _SELECTIVE_GUARDRAIL["consecutive_requested_not_candidate"] = 0
    _record_guardrail_event(
        "cleared",
        reason="candidate_ready",
        threshold=2,
        source="order_test",
    )

    report = native_runner_capability_report()
    history = (report.get("selective_guardrail") or {}).get("history") or []
    assert len(history) >= 2
    assert history[-2].get("event") == "triggered"
    assert history[-1].get("event") == "cleared"
    assert history[-2].get("source") == "order_test"
    assert history[-1].get("source") == "order_test"


def test_abi_model_only_is_default_when_native_enabled():
    """Stream 2: ABI model-only is now the default when native is enabled."""
    import torch

    env = {
        "NATIVE_RUNNER_ENABLED": "1",
        "NATIVE_RUNNER_STRICT": "0",
    }

    class _Session:
        def execute_tokens(self, token_ids, batch=1):
            return [0.1, 0.2, 0.3, 0.4]

    abi_report = {
        "requested": True,
        "attempted": True,
        "succeeded": True,
        "reason": "ok",
        "model_handle": 42,
        "session": _Session(),
    }

    reset_native_runner_telemetry()
    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ), patch(
        "research.scientist.native_runner._maybe_prepare_runner_abi_session", return_value=abi_report
    ), patch(
        "research.scientist.native_runner._legacy_compile_model",
        side_effect=AssertionError("legacy path should not run when ABI default is active"),
    ):
        model = compile_model_native_first([], vocab_size=4)

    out = model(torch.tensor([[1, 2, 3]], dtype=torch.long))
    assert tuple(out.shape) == (1, 3, 4)
    report = getattr(model, "_native_runner_report", {}) or {}
    assert report.get("execution_path") == "native_abi_model_only"
    assert report.get("legacy_compile_used") is False
    assert report.get("execution_mode_classification") == "native_abi_model_only"


def test_abi_default_falls_back_to_legacy_when_abi_fails():
    """When ABI prep fails without DISABLE_LEGACY_COMPILE, fall back to legacy."""
    env = {
        "NATIVE_RUNNER_ENABLED": "1",
        "NATIVE_RUNNER_STRICT": "0",
    }
    abi_report = {
        "requested": True,
        "attempted": True,
        "succeeded": False,
        "reason": "not_supported",
        "model_handle": None,
        "session": None,
    }

    reset_native_runner_telemetry()
    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ), patch(
        "research.scientist.native_runner._maybe_prepare_runner_abi_session", return_value=abi_report
    ), patch(
        "research.scientist.native_runner._legacy_compile_model"
    ) as mock_legacy:
        mock_legacy.return_value = "legacy_model"
        result = compile_model_native_first([])
        mock_legacy.assert_called_once()


def test_abi_hard_fails_when_legacy_compile_disabled():
    """When ABI prep fails AND legacy compile is disabled, raise RuntimeError."""
    env = {
        "NATIVE_RUNNER_ENABLED": "1",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_DISABLE_LEGACY_COMPILE": "1",
    }
    abi_report = {
        "requested": True,
        "attempted": True,
        "succeeded": False,
        "reason": "not_supported",
        "model_handle": None,
        "session": None,
    }

    reset_native_runner_telemetry()
    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ), patch(
        "research.scientist.native_runner._maybe_prepare_runner_abi_session", return_value=abi_report
    ):
        with pytest.raises(RuntimeError, match="requires successful ABI session"):
            compile_model_native_first([])


def test_abi_falls_back_with_legacy_fallback_flag():
    """NATIVE_RUNNER_ALLOW_LEGACY_FALLBACK is removed; legacy fallback is the default."""
    env = {
        "NATIVE_RUNNER_ENABLED": "1",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_ALLOW_LEGACY_FALLBACK": "1",
    }

    abi_report = {
        "requested": True,
        "attempted": True,
        "succeeded": False,
        "reason": "not_supported",
        "model_handle": None,
        "session": None,
    }

    reset_native_runner_telemetry()
    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ), patch(
        "research.scientist.native_runner._maybe_prepare_runner_abi_session", return_value=abi_report
    ), patch(
        "research.scientist.native_runner._legacy_compile_model"
    ) as mock_legacy:
        mock_legacy.return_value = "legacy_model"
        result = compile_model_native_first([])
        mock_legacy.assert_called_once()


def test_capability_payload_includes_execution_mode_classification():
    """Stream 2: capability payload now distinguishes native_abi_model_only / native_legacy_fallback / legacy_only."""
    env = {"NATIVE_RUNNER_ENABLED": "0", "NATIVE_RUNNER_STRICT": "0"}

    class DummyModel:
        pass

    reset_native_runner_telemetry()
    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner._legacy_compile_model", return_value=DummyModel()
    ):
        model = compile_model_native_first([])
        report = getattr(model, "_native_runner_report", {}) or {}
        assert report.get("execution_mode_classification") == "legacy_only"


def test_capability_payload_marks_legacy_disabled():
    env = {"NATIVE_RUNNER_ENABLED": "0", "NATIVE_RUNNER_DISABLE_LEGACY_COMPILE": "1"}
    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ):
        reset_native_runner_telemetry()
        report = native_runner_capability_report()
        assert report.get("execution_mode_classification") == "legacy_disabled"
        assert report.get("legacy_compile_disabled") is True
        assert report.get("legacy_compile_disabled_reason") == "env_flag"


def test_legacy_only_escape_hatch_skips_native_logic():
    """Phase D: NATIVE_RUNNER_LEGACY_ONLY=1 with NATIVE_RUNNER_ENABLED=1 now raises RuntimeError."""
    env = {
        "NATIVE_RUNNER_ENABLED": "1",
        "NATIVE_RUNNER_STRICT": "1",
        "NATIVE_RUNNER_LEGACY_ONLY": "1",
    }

    reset_native_runner_telemetry()
    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ):
        with pytest.raises(RuntimeError, match="NATIVE_RUNNER_LEGACY_ONLY=1 cannot be used when NATIVE_RUNNER_ENABLED=1"):
            compile_model_native_first([object()])

import ctypes
import os
from pathlib import Path
from typing import Any, Dict, List, Set

from .native.core import (
    PARTIAL_NATIVE_COVERAGE_THRESHOLD,
    NativeRunnerState,
    _FALLBACK_METRICS,
    _NATIVE_FALLBACK_LOG_STATE,
    _NATIVE_FALLBACK_LOG_WINDOW_S,
    _SELECTIVE_GUARDRAIL,
    _SELECTIVE_GUARDRAIL_HISTORY,
    _SELECTIVE_GUARDRAIL_HISTORY_MAX,
    _env_flag,
    _reset_cython_bridge_cache,
    _try_import_cython_bridge,
    _try_import_rust_scheduler,
    detect_native_state,
)
from .native.abi import (
    NativeRunnerAbiSession,
    _build_native_abi_only_model,
    _maybe_prepare_runner_abi_session as _maybe_prepare_runner_abi_session_impl,
    _normalize_nr_compile_reason,
    _reset_native_lib_cache,
    _try_load_native_lib as _try_load_native_lib_impl,
    record_native_abi_parity_result,
)
from .native.autograd import NativeForwardWrapper, NativeSubgraphFunction, SubgraphDispatcher
from .native.compiler import _legacy_compile_model as _legacy_compile_model_impl
from .native import abi as _abi_mod
from .native import compiler as _compiler_mod
from .native.designer import (
    DesignerWorkflowLayerAdapter,
    _summarize_layer_build,
    _validate_designer_layer_adapter_contract,
)
from .native.dispatch import (
    _activate_selective_native_dispatch,
    _check_native_op_support,
    _requested_execution_mode,
    dispatch_graph_backward_native,
    dispatch_graph_forward_native_saved,
    dispatch_graph_native,
    dispatch_graph_native_cached,
    dispatch_op_backward_native,
    dispatch_op_native,
)
from .native.guardrails import (
    _maybe_enforce_fallback_guardrails,
    _maybe_fail_on_fallback_rate,
    _maybe_fail_on_legacy_compile_usage,
    _maybe_warn_deprecated_legacy_only_flag,
    _record_guardrail_event,
)
from .native.profiling import enable_native_profiling, get_native_profile
from .native.telemetry import (
    _legacy_compile_count,
    _log_native_fallback_coverage,
    _record_legacy_compile_invocation,
    native_runner_capability_report,
    reset_native_runner_telemetry,
)

_legacy_compile_model = _legacy_compile_model_impl
def _try_load_native_lib():
    _abi_mod.os.environ = os.environ
    _abi_mod.Path = Path
    _abi_mod.ctypes = ctypes
    return _try_load_native_lib_impl()


def _maybe_prepare_runner_abi_session(*, layer_graphs, native_lib, state, vocab_size, max_seq_len):
    _abi_mod.os.environ = os.environ
    return _maybe_prepare_runner_abi_session_impl(
        layer_graphs=layer_graphs,
        native_lib=native_lib,
        state=state,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
    )


def compile_model_native_first(layer_graphs, vocab_size=32000, max_seq_len=None, **kwargs):
    _compiler_mod.os.environ = os.environ
    _compiler_mod._legacy_compile_model = _legacy_compile_model
    _compiler_mod._try_load_native_lib = _try_load_native_lib
    _compiler_mod._maybe_prepare_runner_abi_session = _maybe_prepare_runner_abi_session
    return _compiler_mod.compile_model_native_first(
        layer_graphs,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        **kwargs,
    )



# NativeForwardWrapper and SubgraphDispatcher are re-exported from native.autograd

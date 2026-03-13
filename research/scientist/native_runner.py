"""Re-export facade for the native/ subpackage.

Only the 10 symbols actually consumed by the rest of the codebase are
re-exported here.  Everything else should be imported directly from
the ``scientist.native.*`` submodules.
"""

import ctypes
import os
from pathlib import Path

from .native import abi as _abi_mod
from .native import autograd as _autograd_mod
from .native import compiler as _compiler_mod
from .native.abi import (
    _maybe_prepare_runner_abi_session as _maybe_prepare_runner_abi_session_impl,
    _reset_native_lib_cache,
    _try_load_native_lib as _try_load_native_lib_impl,
    record_native_abi_parity_result,
)
from .native.compiler import _legacy_compile_model as _legacy_compile_model_impl
from .native.core import (
    _FALLBACK_METRICS,
    _SELECTIVE_GUARDRAIL,
    _SELECTIVE_GUARDRAIL_HISTORY_MAX,
    _reset_cython_bridge_cache,
    _try_import_cython_bridge,
    _try_import_rust_scheduler,
    detect_native_state,
)
from .native.dispatch import (
    _activate_selective_native_dispatch,
    _check_native_op_support,
    dispatch_graph_backward_native,
    dispatch_graph_forward_native_saved,
    dispatch_graph_native,
    dispatch_graph_native_cached,
    dispatch_op_backward_native,
    dispatch_op_native,
)
from .native.autograd import NativeSubgraphFunction, SubgraphDispatcher
from .native.guardrails import _maybe_fail_on_fallback_rate, _record_guardrail_event
from .native.profiling import enable_native_profiling, get_native_profile
from .native.telemetry import (
    native_runner_capability_report,
    reset_native_runner_telemetry,
)

_legacy_compile_model = _legacy_compile_model_impl


class NativeForwardWrapper(_autograd_mod.NativeForwardWrapper):
    """Facade wrapper that keeps dispatch patching aligned with legacy tests."""

    def dispatch(self, op_name, *tensors):
        _autograd_mod.dispatch_op_native = dispatch_op_native
        return super().dispatch(op_name, *tensors)


def _try_load_native_lib():
    _abi_mod.os.environ = os.environ
    _abi_mod.Path = Path
    _abi_mod.ctypes = ctypes
    return _try_load_native_lib_impl()


def _maybe_prepare_runner_abi_session(*, layer_graphs, native_lib, state, vocab_size, max_seq_len):
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

"""Re-export facade for the native/ subpackage.

Only the 10 symbols actually consumed by the rest of the codebase are
re-exported here.  Everything else should be imported directly from
the ``scientist.native.*`` submodules.
"""

import ctypes
import os
from pathlib import Path

from research.defaults import VOCAB_SIZE
from .native import abi as _abi_mod
from .native import autograd as _autograd_mod
from .native import compiler as _compiler_mod
from .native.abi import (
    _maybe_prepare_runner_abi_session as _maybe_prepare_runner_abi_session_impl,
    _try_load_native_lib as _try_load_native_lib_impl,
)
from .native.compiler import _legacy_compile_model as _legacy_compile_model_impl
from .native.dispatch import (
    dispatch_op_native,
)

_legacy_compile_model = _legacy_compile_model_impl


class NativeForwardWrapper(_autograd_mod.NativeForwardWrapper):
    """Facade wrapper that keeps dispatch patching aligned with legacy tests."""

    def dispatch(self, op_name, *tensors):
        _autograd_mod.dispatch_op_native = dispatch_op_native
        return super().dispatch(op_name, *tensors)


def _try_load_native_lib():
    _abi_mod.os = os
    _abi_mod.Path = Path
    _abi_mod.ctypes = ctypes
    return _try_load_native_lib_impl()


def _maybe_prepare_runner_abi_session(
    *, layer_graphs, native_lib, state, vocab_size, max_seq_len
):
    return _maybe_prepare_runner_abi_session_impl(
        layer_graphs=layer_graphs,
        native_lib=native_lib,
        state=state,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
    )


def compile_model_native_first(
    layer_graphs, vocab_size=VOCAB_SIZE, max_seq_len=None, **kwargs
):
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

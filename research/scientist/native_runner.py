"""Re-export facade for the native/ subpackage."""

import os  # noqa: F401 - compatibility surface for tests and legacy monkeypatches.

from research.defaults import VOCAB_SIZE
from .native import autograd as _autograd_mod
from .native import compiler as _compiler_mod
from .native import telemetry as _telemetry_mod
from .native.abi import (
    _maybe_prepare_runner_abi_session,
    _try_load_native_lib,
)
from .native.compiler import _legacy_compile_model
from .native.dispatch import (
    dispatch_graph_native as dispatch_graph_native,  # noqa: F401
    dispatch_op_native as dispatch_op_native,  # noqa: F401
)


class NativeForwardWrapper(_autograd_mod.NativeForwardWrapper):
    """Compatibility alias for legacy imports."""


def compile_model_native_first(
    layer_graphs, vocab_size=VOCAB_SIZE, max_seq_len=None, **kwargs
):
    # Reject graphs with byte-unsafe ops before native compilation —
    # token_merge/mod_topk break tensor layout in native execution.
    from research.synthesis.context_rules import find_byte_safety_violations

    for g in layer_graphs:
        if not hasattr(g, "nodes"):
            continue
        try:
            violations = find_byte_safety_violations(g)
        except AttributeError:
            continue
        if violations:
            raise ValueError(f"Cannot compile for native execution: {violations[0]}")
    _compiler_mod._legacy_compile_model = _legacy_compile_model
    _compiler_mod._try_load_native_lib = _try_load_native_lib
    _compiler_mod._maybe_prepare_runner_abi_session = _maybe_prepare_runner_abi_session
    return _compiler_mod.compile_model_native_first(
        layer_graphs,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        **kwargs,
    )


def native_runner_capability_report():
    return _telemetry_mod.native_runner_capability_report()


def reset_native_runner_telemetry():
    return _telemetry_mod.reset_native_runner_telemetry()

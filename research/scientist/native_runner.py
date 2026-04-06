"""Re-export facade for the native/ subpackage."""

from research.defaults import VOCAB_SIZE
from .native import autograd as _autograd_mod
from .native import telemetry as _telemetry_mod
from .native.compiler import compile_model_native_first as _compile_model_native_first


class NativeForwardWrapper(_autograd_mod.NativeForwardWrapper):
    """Compatibility alias for legacy imports."""


def compile_model_native_first(
    layer_graphs, vocab_size=VOCAB_SIZE, max_seq_len=None, **kwargs
):
    # Reject graphs with byte-unsafe ops before native compilation —
    # token_merge/mod_topk break tensor layout in native execution.
    from research.synthesis.context_rules import find_byte_safety_violations

    for g in layer_graphs:
        violations = find_byte_safety_violations(g)
        if violations:
            raise ValueError(f"Cannot compile for native execution: {violations[0]}")
    return _compile_model_native_first(
        layer_graphs,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        **kwargs,
    )


def native_runner_capability_report():
    return _telemetry_mod.native_runner_capability_report()


def reset_native_runner_telemetry():
    return _telemetry_mod.reset_native_runner_telemetry()

"""Re-export facade for the native/ subpackage."""

import logging
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
from .native.core import (
    _env_flag,
    _try_import_rust_scheduler as _try_import_rust_scheduler,
)
from .native.dispatch import (
    dispatch_graph_native as dispatch_graph_native,  # noqa: F401
    dispatch_op_native as dispatch_op_native,  # noqa: F401
)
from .native.profiling import (
    enable_native_profiling as enable_native_profiling,
    get_native_profile as get_native_profile,
)

logger = logging.getLogger(__name__)


class NativeForwardWrapper(_autograd_mod.NativeForwardWrapper):
    """Compatibility alias for legacy imports."""


def compile_model_native_first(
    layer_graphs, vocab_size=VOCAB_SIZE, max_seq_len=None, **kwargs
):
    # token_merge/mod_topk break tensor layout in native execution, but the
    # legacy PyTorch compiler runs them correctly. Route byte-unsafe graphs
    # there instead of failing — the same decision runner/screening.py makes,
    # so replay/backfill tools can rebuild these rows. Honour an explicit
    # legacy-disable gate by failing loud rather than silently.
    from research.synthesis.context_rules import find_byte_safety_violations

    for g in layer_graphs:
        if not hasattr(g, "nodes"):
            continue
        try:
            violations = find_byte_safety_violations(g)
        except AttributeError:
            continue
        if violations:
            if _env_flag("NATIVE_RUNNER_DISABLE_LEGACY_COMPILE", False):
                raise ValueError(
                    f"Cannot compile for native execution: {violations[0]}"
                )
            logger.info(
                "Routing byte-unsafe graph to legacy compile "
                "(native execution unsupported): %s",
                violations[0],
            )
            return _legacy_compile_model(
                layer_graphs,
                vocab_size=vocab_size,
                max_seq_len=max_seq_len,
                **kwargs,
            )
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

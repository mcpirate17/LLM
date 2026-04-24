"""Execution mixin: validation + scale-up threads."""

from __future__ import annotations


import logging

logger = logging.getLogger(__name__)


def _fail_loud(phase: str, message: str, exc: BaseException) -> None:
    logger.exception("%s: %s", phase, message)
    raise RuntimeError(f"{phase}: {message}") from exc


# ── Mixin composition ─────────────────────────────────────────────
# Method bodies live in three split modules to stay under the
# 1250-line file cap. _ExecutionValidationMixin composes them.

from .execution_validation_thread import _ExecutionValidationThreadMixin  # noqa: E402
from .execution_validation_candidate import _ExecutionValidationCandidateMixin  # noqa: E402
from .execution_validation_scale import _ExecutionValidationScaleMixin  # noqa: E402
from ._lifecycle import _LifecycleMixin  # noqa: E402


class _ExecutionValidationMixin(
    _ExecutionValidationThreadMixin,
    _ExecutionValidationCandidateMixin,
    _ExecutionValidationScaleMixin,
):
    """Execution-validation mixin (composed)."""

    __slots__ = ()
    _publish_terminal_event = _LifecycleMixin._publish_terminal_event
    _publish_validation_terminal_event = _LifecycleMixin._publish_terminal_event
    _fail_experiment_compat = _LifecycleMixin._fail_experiment_compat
    _complete_experiment_compat = _LifecycleMixin._complete_experiment_compat
    _log_learning_event_compat = _LifecycleMixin._log_learning_event_compat

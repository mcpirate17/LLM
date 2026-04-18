"""Execution mixin: validation + scale-up threads."""

from __future__ import annotations

import json
import sqlite3
import time
import traceback
from typing import List

from ..json_utils import json_safe
from ..runtime_events import publish_lifecycle_event


from ..native_runner import compile_model_native_first as compile_model
from ...synthesis.serializer import graph_to_json, graph_from_json
from ...eval.metrics import novelty_score
from ...eval.fingerprint import compute_fingerprint
from ...eval.diagnostic_tasks import run_diagnostic_suite
from ...training.checkpointing import CheckpointManager
from ..shared_utils import resolve_device
from ._helpers import (
    clear_gpu_memory,
    compute_seed_metrics,
    run_baseline_comparison,
    build_validation_entry,
    promote_validation_candidate,
    run_trajectory_probe,
    handle_breakthrough,
    screening_probe_fields,
    screening_wikitext_fields,
)

import logging

logger = logging.getLogger(__name__)

from ._types import RunConfig


def _fail_loud(phase: str, message: str, exc: BaseException) -> None:
    logger.exception("%s: %s", phase, message)
    raise RuntimeError(f"{phase}: {message}") from exc




# ── Mixin composition ─────────────────────────────────────────────
# Method bodies live in three split modules to stay under the
# 1250-line file cap. _ExecutionValidationMixin composes them.

from .execution_validation_thread import _ExecutionValidationThreadMixin  # noqa: E402
from .execution_validation_candidate import _ExecutionValidationCandidateMixin  # noqa: E402
from .execution_validation_scale import _ExecutionValidationScaleMixin  # noqa: E402


class _ExecutionValidationMixin(
    _ExecutionValidationThreadMixin,
    _ExecutionValidationCandidateMixin,
    _ExecutionValidationScaleMixin,
):
    """Execution-validation mixin (composed)."""

    __slots__ = ()

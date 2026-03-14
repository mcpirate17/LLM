from __future__ import annotations

import pytest

from research.scientist.runner._types import RunConfig
from research.scientist.runner.execution_training import (
    _allow_synthesized_training,
    _training_phase,
)

pytestmark = pytest.mark.unit


class _Owner:
    __slots__ = ("_live_training_context",)

    def __init__(self, phase: str):
        self._live_training_context = {"phase": phase}


def test_training_phase_reads_runner_context():
    assert _training_phase(_Owner("synthesis")) == "synthesis"
    assert _training_phase(object()) == ""


def test_synthesized_training_is_screening_only():
    config = RunConfig(loss_type="synthesized", optimizer_type="synthesized")

    assert _allow_synthesized_training(_Owner("synthesis"), config)
    assert _allow_synthesized_training(_Owner("candidate_screening"), config)
    assert not _allow_synthesized_training(_Owner("investigation"), config)
    assert not _allow_synthesized_training(_Owner("validation"), config)

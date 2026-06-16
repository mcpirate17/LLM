"""Shared builders + fixtures for component_fab tests.

``make_spec`` / ``make_candidate_spec`` replace the per-file ``_spec``
helpers; ``grade_record`` / ``promote_record`` mirror exactly the record
schema ``Ledger.record_grade`` / ``Ledger.record_promotion`` emit (keep
them in lock-step — a drifted key here silently invalidates analyzer
tests); ``write_ledger_jsonl`` goes through the real ``JsonlWriter``.
"""

from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path
from typing import Any, Iterable

# ── CPU thread hygiene under xdist ───────────────────────────────────
# Same fix as research/tests/conftest.py: without this every worker
# initializes an all-core OpenMP/BLAS pool and the nano-training tests
# thrash (measured 1233s → 3s per test once pinned). Must run before
# torch import; explicit env settings win.
_XDIST_WORKERS = os.environ.get("PYTEST_XDIST_WORKER_COUNT")
if _XDIST_WORKERS:
    _threads = str(max(1, (os.cpu_count() or 1) // int(_XDIST_WORKERS)))
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(_var, _threads)

import pytest

from component_fab.proposer.property_miner import CandidateTuple
from component_fab.proposer.spec_generator import (
    ProposalSpec,
    category_from_axes,
    spec_from_candidate,
    synthetic_axis_lift,
)
from component_fab.state.ledger import JsonlWriter, Ledger


def make_spec(
    axes: dict[str, Any] | None = None,
    pid: str = "cand",
    **overrides: Any,
) -> ProposalSpec:
    """Direct ProposalSpec builder with test-friendly defaults.

    ``overrides`` map onto ProposalSpec fields; ``anchor_witnesses_all``
    follows ``anchor_witness_op`` unless overridden explicitly.
    """
    axes = {} if axes is None else axes
    witness = overrides.pop("anchor_witness_op", "")
    fields: dict[str, Any] = {
        "proposal_id": pid,
        "name": pid,
        "category": category_from_axes(axes),
        "synthesis_kind": "novel_hybrid",
        "math_axes": axes,
        "anchor_witness_op": witness,
        "anchor_witnesses_all": (witness,) if witness else (),
        "declared_property_row": dict(axes),
        "predicted_lift": 0.5,
        "rationale": "test",
    }
    fields.update(overrides)
    return ProposalSpec(**fields)


def make_candidate_spec(
    axes: dict[str, Any], *, witness: str = "anchor"
) -> ProposalSpec:
    """Spec built through the real ``spec_from_candidate`` pipeline path."""
    tuple_values = tuple(axes.items())
    candidate = CandidateTuple(
        tuple_values=tuple_values,
        predicted_lift=0.5,
        per_axis_lift=tuple(
            synthetic_axis_lift(axis, value) for axis, value in tuple_values
        ),
        witness_ops=(witness,),
    )
    return spec_from_candidate(candidate)


def grade_record(
    pid: str,
    *,
    cycle: int = 1,
    knobs: tuple[str, ...] = (),
    eliminated_by: str | None = None,
    composite: float = 0.0,
    learned: bool = False,
    erf: float | None = None,
    nb: float | None = None,
    can_bind: bool | None = None,
    math_axes: dict[str, Any] | None = None,
    synthesis_kind: str = "semiring_swap",
    category: str = "lane",
    name: str | None = None,
) -> dict[str, Any]:
    """A ledger ``grade`` event dict — same keys as ``Ledger.record_grade``."""
    meta: dict[str, Any] = {"math_knobs": list(knobs)}
    if eliminated_by is not None:
        meta["eliminated_by"] = eliminated_by
    if erf is not None:
        meta["erf_density"] = erf
    if nb is not None:
        meta["nb_max_accuracy"] = nb
    if can_bind is not None:
        meta["can_bind"] = can_bind
    if math_axes is not None:
        meta["math_axes"] = math_axes
    return {
        "event": "grade",
        "proposal_id": pid,
        "name": name or pid,
        "category": category,
        "synthesis_kind": synthesis_kind,
        "cycle": cycle,
        "composite_score": composite,
        "smoke_pass": eliminated_by != "smoke",
        "learned_signal": learned,
        "metadata": meta,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }


def promote_record(pid: str, status: str = "promoted") -> dict[str, Any]:
    """A ledger ``promote`` event dict — same keys as ``record_promotion``."""
    return {
        "event": "promote",
        "proposal_id": pid,
        "status": status,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }


def write_ledger_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> Path:
    """Write records through the real ledger writer (one JSONL line each)."""
    with JsonlWriter(path) as writer:
        for record in records:
            writer.write(record)
    return path


@pytest.fixture
def tmp_ledger(tmp_path: Path) -> Ledger:
    """Fresh empty Ledger backed by a tmp file."""
    return Ledger(tmp_path / "ledger.jsonl")

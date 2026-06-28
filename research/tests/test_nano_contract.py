"""The nano contract: one floor for all nano models, with proof it can learn."""

from __future__ import annotations

import logging

import pytest

from research.eval.nano_contract import (
    NANO,
    measure_reference_induction_auc,
    meets_param_floor,
    reference_model,
)
from research.scientist.notebook.program_writes import _enforce_nano_floor

_LOG = logging.getLogger("test_nano_contract")


def _kw(**over):
    base = {"stage1_passed": True, "param_count": NANO.min_params}
    base.update(over)
    return base


# ── the floor predicate ──────────────────────────────────────────────────────


def test_param_floor_predicate() -> None:
    assert meets_param_floor(NANO.min_params) is True
    assert meets_param_floor(NANO.min_params + 1) is True
    assert meets_param_floor(NANO.min_params - 1) is False
    assert meets_param_floor(0) is False
    assert meets_param_floor(None) is False


def test_reference_model_is_above_floor() -> None:
    params = sum(p.numel() for p in reference_model().parameters())
    assert params >= NANO.min_params  # the positive control itself meets the floor


# ── hard-reject enforcement at the write gate ─────────────────────────────────


def test_sub_floor_stage1_write_is_rejected() -> None:
    with pytest.raises(ValueError, match="sub-floor nano"):
        _enforce_nano_floor(
            graph_fingerprint="abc123", kwargs=_kw(param_count=50_000), logger=_LOG
        )


def test_at_floor_write_is_allowed() -> None:
    _enforce_nano_floor(
        graph_fingerprint="abc123", kwargs=_kw(param_count=NANO.min_params), logger=_LOG
    )  # no raise


def test_sub_floor_allowed_when_not_stage1_passed() -> None:
    _enforce_nano_floor(
        graph_fingerprint="abc",
        kwargs=_kw(stage1_passed=False, param_count=1000),
        logger=_LOG,
    )  # a sub-floor model is fine as long as it does not CLAIM a screening pass


def test_missing_param_count_is_not_blocked_by_floor() -> None:
    _enforce_nano_floor(
        graph_fingerprint="abc", kwargs=_kw(param_count=None), logger=_LOG
    )  # floor cannot be enforced without a count; other gates cover that case


def test_backfill_trust_label_bypasses_floor() -> None:
    _enforce_nano_floor(
        graph_fingerprint="abc",
        kwargs=_kw(param_count=1000, trust_label="replay_backfill_x"),
        logger=_LOG,
    )  # replay/backfill paths re-fill in place; not re-gated


# ── the demonstrated evidence: the floor can learn ────────────────────────────


@pytest.mark.slow
def test_reference_at_floor_demonstrably_learns() -> None:
    """The positive control at the nano floor must clear the learnability gate.

    This is the committed evidence that the minimum nano size can learn: if the
    floor is ever shrunk below where a capable architecture learns, this fails.
    """
    auc = measure_reference_induction_auc(n_train_steps=800, seed=0)
    assert auc > NANO.learnability_threshold, (
        f"reference induction AUC {auc:.3f} <= floor "
        f"{NANO.learnability_threshold} — the nano floor no longer learns"
    )

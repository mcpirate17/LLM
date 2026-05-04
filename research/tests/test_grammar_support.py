"""Unit tests for the Phase 3.4 sa_factor blender extension.

Covers:
  - _sa_factor: floors at 0.1, returns 1.0 below n=20, scales linearly above.
  - _template_dynamic_weight: returns 4-tuple, applies sa_factor multiplicatively,
    clamps final weight to [0.5, 8.0].
"""

from __future__ import annotations

import pytest

from research.synthesis.grammar_support import (
    _sa_factor,
    _template_dynamic_weight,
    _SA_FACTOR_MIN_N,
)


pytestmark = [pytest.mark.unit]


def test_sa_factor_below_min_n_returns_one() -> None:
    assert _sa_factor(0, 0) == 1.0
    assert _sa_factor(5, _SA_FACTOR_MIN_N - 1) == 1.0
    assert _sa_factor(10, _SA_FACTOR_MIN_N - 1) == 1.0


def test_sa_factor_at_normalizer_passes_to_one() -> None:
    # 0.40 pass rate / 0.40 normalizer = 1.0
    assert _sa_factor(40, 100) == pytest.approx(1.0)


def test_sa_factor_high_pass_boosts() -> None:
    # 0.80 / 0.40 = 2.0
    assert _sa_factor(80, 100) == pytest.approx(2.0)


def test_sa_factor_zero_pass_floors() -> None:
    assert _sa_factor(0, 100) == pytest.approx(0.1)


def test_sa_factor_low_pass_floors() -> None:
    # 0.01 / 0.40 = 0.025 → clamped to 0.1 floor
    assert _sa_factor(1, 100) == pytest.approx(0.1)


def _row(
    tpl_name: str = "tpl_x",
    sa_pass: int = 80,
    sa_n: int = 100,
    eval_count: int = 50,
    s1_count: int = 25,
    mean_loss: float = 0.5,
):
    """Construct a synthetic row matching _fetch_template_weight_rows shape."""
    return (
        tpl_name,
        eval_count,
        s1_count,
        mean_loss,
        # _DB_METRIC_COLUMNS — 9 fields, all None for unit-test simplicity
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        sa_pass,
        sa_n,
    )


def test_template_dynamic_weight_returns_four_tuple() -> None:
    out = _template_dynamic_weight(
        _row(),
        template_slot_context={},
        default_template_weights={"tpl_x": 2.0},
        k=2.0,
    )
    assert len(out) == 4
    assert out[0] == "tpl_x"
    assert isinstance(out[1], int)
    assert isinstance(out[2], float)
    assert isinstance(out[3], float)


def test_template_dynamic_weight_sa_factor_propagates() -> None:
    # 80/100 = 0.80 / 0.40 = 2.0
    _, _, _, sa_factor = _template_dynamic_weight(
        _row(sa_pass=80, sa_n=100),
        template_slot_context={},
        default_template_weights={"tpl_x": 2.0},
        k=2.0,
    )
    assert sa_factor == pytest.approx(2.0)


def test_template_dynamic_weight_clamped_to_ceiling() -> None:
    # Force everything high; final weight must clamp at 8.0
    _, _, weight, _ = _template_dynamic_weight(
        _row(sa_pass=99, sa_n=100, mean_loss=0.0, s1_count=50, eval_count=50),
        template_slot_context={},
        default_template_weights={"tpl_x": 8.0},
        k=2.0,
    )
    assert weight <= 8.0


def test_template_dynamic_weight_clamped_to_floor() -> None:
    # Force everything low; final weight must clamp at 0.5
    _, _, weight, sa_factor = _template_dynamic_weight(
        _row(sa_pass=0, sa_n=100, mean_loss=5.0, s1_count=0, eval_count=50),
        template_slot_context={},
        default_template_weights={"tpl_x": 0.5},
        k=2.0,
    )
    assert sa_factor == pytest.approx(0.1)
    assert weight >= 0.5


def test_template_dynamic_weight_sub_threshold_n_no_penalty() -> None:
    # n_sa < 20 → sa_factor = 1.0 even if pass count is zero
    _, _, _, sa_factor = _template_dynamic_weight(
        _row(sa_pass=0, sa_n=10),
        template_slot_context={},
        default_template_weights={"tpl_x": 2.0},
        k=2.0,
    )
    assert sa_factor == pytest.approx(1.0)

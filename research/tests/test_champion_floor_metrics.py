from __future__ import annotations

import math

import pytest

from research.eval.champion_floor_metrics import (
    CHAMPION_FLOOR_PROTOCOL_VERSION,
    CHAMPION_PLATEAU_WINDOW_STEPS,
    extract_champion_floor_metrics,
    lookup_gpt2_champion_baseline,
)


def test_fast_plateau_back_counts_floor_entry() -> None:
    curve = []
    for step in range(0, 2_001, 10):
        if step < 300:
            loss = 8.0 - 0.01 * step
        else:
            loss = 5.0 + (0.01 if (step // 10) % 2 else -0.01)
        curve.append((step, loss))

    metrics = extract_champion_floor_metrics(curve)

    assert metrics.champion_floor_protocol_version == CHAMPION_FLOOR_PROTOCOL_VERSION
    assert metrics.champion_plateau_window == CHAMPION_PLATEAU_WINDOW_STEPS
    assert 250 <= metrics.champion_steps_to_floor <= 350
    assert metrics.champion_plateau_detected_step >= (
        metrics.champion_steps_to_floor + CHAMPION_PLATEAU_WINDOW_STEPS
    )
    assert metrics.champion_floor_loss == pytest.approx(5.0, abs=0.02)
    assert metrics.champion_floor_ppl == pytest.approx(
        math.exp(metrics.champion_floor_loss)
    )


def test_no_plateau_returns_protocol_fields_without_floor_values() -> None:
    curve = [(step, 8.0 - 0.001 * step) for step in range(0, 2_001, 10)]

    metrics = extract_champion_floor_metrics(curve)

    assert metrics.champion_steps_to_floor is None
    assert metrics.champion_floor_loss is None
    assert metrics.champion_floor_ppl is None
    assert metrics.champion_floor_loss_std is None
    assert metrics.champion_plateau_detected_step is None
    assert metrics.champion_plateau_window == CHAMPION_PLATEAU_WINDOW_STEPS
    assert metrics.champion_floor_protocol_version == CHAMPION_FLOOR_PROTOCOL_VERSION


def test_noisy_stable_floor_uses_tail_floor_and_std() -> None:
    curve = []
    for step in range(0, 2_501, 10):
        if step < 700:
            loss = 7.4 - 0.0035 * step
        else:
            loss = 4.92 + ((step // 10) % 5 - 2) * 0.015
        curve.append({"step": step, "loss": loss})

    metrics = extract_champion_floor_metrics(curve)

    assert 650 <= metrics.champion_steps_to_floor <= 750
    assert metrics.champion_plateau_detected_step >= 1_150
    assert metrics.champion_floor_loss == pytest.approx(4.92, abs=0.03)
    assert 0.015 <= metrics.champion_floor_loss_std <= 0.025


def test_missing_and_short_curves_do_not_raise() -> None:
    curve = [
        {"step": 0, "loss": "bad"},
        {"step": 10, "loss": float("nan")},
        {"step": 20, "loss": 7.0},
        (30, None),
    ]

    metrics = extract_champion_floor_metrics(curve)

    assert metrics.champion_steps_to_floor is None
    assert metrics.to_dict()["champion_floor_loss"] is None


def test_gpt2_baseline_lookup_by_layers_and_protocol() -> None:
    baseline_4l = lookup_gpt2_champion_baseline(4)
    baseline_6l = lookup_gpt2_champion_baseline(6)

    assert baseline_4l.result_id == "gpt2cal490d5"
    assert baseline_4l.layers == 4
    assert baseline_4l.protocol_version == CHAMPION_FLOOR_PROTOCOL_VERSION
    assert baseline_4l.champion_plateau_window == CHAMPION_PLATEAU_WINDOW_STEPS
    assert baseline_4l.champion_steps_to_floor == 11_742
    assert baseline_6l.result_id == "gpt2cal87a29"
    assert baseline_6l.layers == 6


def test_gpt2_baseline_lookup_explicit_fallback_and_unsupported_layers() -> None:
    fallback = lookup_gpt2_champion_baseline(5, default_layers=4)

    assert fallback.result_id == "gpt2cal490d5"
    with pytest.raises(KeyError):
        lookup_gpt2_champion_baseline(5)

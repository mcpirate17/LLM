# pyright: reportPrivateImportUsage=false
"""Tests for the mixing + learning-speed ranking axes (2026-07-01).

Locks in the wiring of the two user-named objectives — "maximum mixing" and
"minimum steps" — into both the scalar composite (stability-gated bonuses) and
the Pareto objective vector. Uses synthetic scorecard dicts so the subscore
arithmetic is checked without running lanes.
"""

from __future__ import annotations

from component_fab.improver.ranking import (
    OBJECTIVE_KEYS,
    _LEARN_SPEED_BONUS,
    _LEARN_SPEED_REF_STEPS,
    _MIXING_BONUS,
    composite_score,
    learning_speed_subscore,
    mixing_subscore,
    objective_vector,
)


def _solo() -> dict:
    return {
        "proposal_id": "p",
        "name": "n",
        "category": "c",
        "synthesis_kind": "k",
        "smoke": {
            "forward_passed": True,
            "backward_passed": True,
            "output_finite": True,
            "param_grad_finite": True,
        },
        "property_cross_check": {"a_consistent": True},
        "metadata": {"orthogonality_radius": 0.0},
    }


def _cap(
    *, mixing: float = 0.0, mean_steps: float | None = None, ind: float = 0.0
) -> dict:
    return {
        "mixing_subscore": mixing,
        "mean_steps_to_threshold": mean_steps,
        "ind_max_accuracy": ind,
        "binds_per_probe": {},
        "relative_recall_per_probe": {},
    }


def test_mixing_subscore_reads_and_clamps() -> None:
    assert mixing_subscore(_cap(mixing=0.7)) == 0.7
    assert mixing_subscore(None) == 0.0
    assert mixing_subscore(_cap(mixing=5.0)) == 1.0
    assert mixing_subscore(_cap(mixing=-1.0)) == 0.0


def test_learning_speed_subscore_maps_steps() -> None:
    assert learning_speed_subscore(_cap(mean_steps=0)) == 1.0
    assert learning_speed_subscore(_cap(mean_steps=_LEARN_SPEED_REF_STEPS)) == 0.0
    assert abs(learning_speed_subscore(_cap(mean_steps=30)) - 0.5) < 1e-9
    assert learning_speed_subscore(_cap(mean_steps=None)) == 0.0
    assert learning_speed_subscore(None) == 0.0


def test_objective_keys_include_new_axes() -> None:
    assert "mixing" in OBJECTIVE_KEYS
    assert "learning_speed" in OBJECTIVE_KEYS


def test_objective_vector_gates_mixing_and_speed_behind_floor() -> None:
    stable = objective_vector(None, _cap(mixing=0.8, mean_steps=12, ind=0.5))
    assert stable["mixing"] == 0.8
    assert stable["learning_speed"] == 1.0 - 12.0 / _LEARN_SPEED_REF_STEPS

    unstable = objective_vector(None, _cap(mixing=0.8, mean_steps=12, ind=0.0))
    assert unstable["mixing"] == 0.0
    assert unstable["learning_speed"] == 0.0


def test_composite_mixing_and_speed_bonus_only_when_stable() -> None:
    solo = _solo()
    base = composite_score(solo, None, _cap(ind=0.5))[0]
    rich = composite_score(solo, None, _cap(mixing=0.8, mean_steps=12, ind=0.5))[0]
    expected_lift = _MIXING_BONUS * 0.8 + _LEARN_SPEED_BONUS * (
        1.0 - 12.0 / _LEARN_SPEED_REF_STEPS
    )
    assert abs((rich - base) - expected_lift) < 1e-9

    unstable_base = composite_score(solo, None, _cap(ind=0.0))[0]
    unstable_rich = composite_score(
        solo, None, _cap(mixing=0.8, mean_steps=12, ind=0.0)
    )[0]
    assert abs(unstable_rich - unstable_base) < 1e-9


def test_components_dict_carries_new_subscores() -> None:
    solo = _solo()
    _, comps = composite_score(solo, None, _cap(mixing=0.4, mean_steps=20, ind=0.5))
    assert comps["mixing"] == 0.4
    assert comps["learning_speed"] == 1.0 - 20.0 / _LEARN_SPEED_REF_STEPS

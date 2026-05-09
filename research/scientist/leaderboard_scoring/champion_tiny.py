"""Champion tiny-model 50-point protocol score and v12 hard-failure gate.

Imported by both v11 (trust ceiling) and v12 (champion eligibility gate).
The ``_V12_CHAMPION_ELIGIBILITY_CEILING`` it uses lives in ``_config`` to
break the champion_tiny ↔ v12 cycle.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ._config import _V12_CHAMPION_ELIGIBILITY_CEILING
from ._utils import (
    _clamp01,
    _falsey_flag,
    _finite_metric,
    _first_present,
    _truthy_flag,
)


CHAMPION_TINY_MODEL_SCORE_V1 = "champion_tiny_model_score_v1"
CHAMPION_INDUCTION_V3_PROTOCOLS = {
    "induction_validation_full_counterfactual_2k",
    "induction_validation_full_counterfactual_5k",
    "induction_validation_full_counterfactual_10k",
}


def compute_champion_tiny_model_score_v1(**metrics: Any) -> Dict[str, Any]:
    """Compute the versioned 50-point tiny-model champion score.

    This helper is pure arithmetic. It treats final CE/loss as informational:
    champion hard failures are limited to divergence, missing checkpoints,
    persistence failure, or corrupt/missing required champion-protocol metrics.
    """
    zero = {
        "protocol_version": CHAMPION_TINY_MODEL_SCORE_V1,
        "steps_to_floor": 0.0,
        "floor_quality": 0.0,
        "floor_stability": 0.0,
        "induction_validation": 0.0,
        "binding_long_context": 0.0,
        "ar_validation": 0.0,
        "total": 0.0,
        "hard_failure_reason": None,
    }

    diverged = any(
        _truthy_flag(metrics.get(name))
        for name in (
            "diverged",
            "training_diverged",
            "divergence_detected",
            "champion_diverged",
        )
    )
    status = (
        str(metrics.get("training_status") or metrics.get("champion_status") or "")
        .strip()
        .lower()
    )
    if diverged or status in {"diverged", "nan", "nonfinite", "non_finite"}:
        return {**zero, "hard_failure_reason": "divergence"}

    persistence_failed = any(
        _truthy_flag(metrics.get(name))
        for name in (
            "persistence_failed",
            "champion_persistence_failed",
            "result_persistence_failed",
        )
    )
    if persistence_failed:
        return {**zero, "hard_failure_reason": "persistence_failure"}

    checkpoint_available = metrics.get("checkpoint_available")
    if checkpoint_available is None:
        checkpoint_available = metrics.get("champion_checkpoint_available")
    missing_checkpoint = any(
        _truthy_flag(metrics.get(name))
        for name in ("missing_checkpoint", "champion_missing_checkpoint")
    )
    checkpoint_path = metrics.get("checkpoint_path") or metrics.get(
        "champion_checkpoint_path"
    )
    if missing_checkpoint or _falsey_flag(checkpoint_available):
        return {**zero, "hard_failure_reason": "missing_checkpoint"}
    if checkpoint_available is None and not checkpoint_path:
        return {**zero, "hard_failure_reason": "missing_checkpoint"}

    normalized: Dict[str, Any] = dict(metrics)
    aliases = {
        "champion_baseline_steps_to_floor": (
            "gpt2_steps_to_floor",
            "baseline_steps_to_floor",
        ),
        "champion_baseline_floor_ppl": ("gpt2_floor_ppl", "baseline_floor_ppl"),
        "champion_baseline_floor_loss": ("gpt2_floor_loss", "baseline_floor_loss"),
        "champion_baseline_floor_loss_std": (
            "gpt2_floor_loss_std",
            "baseline_floor_loss_std",
        ),
        "champion_baseline_long_ctx_combined_score": (
            "gpt2_long_context_baseline",
            "baseline_long_ctx_combined_score",
        ),
        "champion_baseline_ar_validation_steps_to_floor": (
            "gpt2_ar_validation_steps_to_floor",
            "baseline_ar_validation_steps_to_floor",
        ),
        "binding_intermediate_auc": (
            "binding_intermediate_auc",
            "binding_screening_auc",
        ),
        "robustness_long_ctx_combined_score": (
            "long_ctx_combined_score",
            "champion_long_ctx_combined_score",
        ),
    }
    for canonical, fallback_names in aliases.items():
        if normalized.get(canonical) is None:
            _, value = _first_present(normalized, *fallback_names)
            if value is not None:
                normalized[canonical] = value

    errors: Dict[str, list[str]] = {"missing": [], "corrupt": []}
    induction_validation_protocol = str(
        normalized.get("induction_validation_protocol_version") or ""
    ).strip()
    if not induction_validation_protocol:
        errors["missing"].append("induction_validation_protocol_version")
    elif induction_validation_protocol not in CHAMPION_INDUCTION_V3_PROTOCOLS:
        errors["corrupt"].append("induction_validation_protocol_version")

    steps = _finite_metric(normalized, errors, "champion_steps_to_floor", positive=True)
    baseline_steps = _finite_metric(
        normalized, errors, "champion_baseline_steps_to_floor", positive=True
    )

    floor_ppl_name, floor_ppl_value = _first_present(
        normalized, "champion_floor_ppl", "floor_ppl"
    )
    baseline_ppl_name, baseline_ppl_value = _first_present(
        normalized, "champion_baseline_floor_ppl", "baseline_floor_ppl"
    )
    use_ppl_floor = floor_ppl_value is not None or baseline_ppl_value is not None
    if use_ppl_floor:
        normalized[floor_ppl_name] = floor_ppl_value
        normalized[baseline_ppl_name] = baseline_ppl_value
        floor_quality_value = _finite_metric(
            normalized, errors, floor_ppl_name, positive=True
        )
        baseline_floor_quality = _finite_metric(
            normalized, errors, baseline_ppl_name, positive=True
        )
    else:
        floor_loss_name, floor_loss_value = _first_present(
            normalized, "champion_floor_loss", "floor_loss"
        )
        baseline_loss_name, baseline_loss_value = _first_present(
            normalized, "champion_baseline_floor_loss", "baseline_floor_loss"
        )
        normalized[floor_loss_name] = floor_loss_value
        normalized[baseline_loss_name] = baseline_loss_value
        floor_quality_value = _finite_metric(
            normalized, errors, floor_loss_name, positive=True
        )
        baseline_floor_quality = _finite_metric(
            normalized, errors, baseline_loss_name, positive=True
        )

    floor_std = _finite_metric(
        normalized, errors, "champion_floor_loss_std", min_value=0.0
    )
    baseline_floor_std = _finite_metric(
        normalized,
        errors,
        "champion_baseline_floor_loss_std",
        positive=True,
    )

    induction_screening_auc = _finite_metric(
        normalized, errors, "induction_validation_auc", min_value=0.0, max_value=1.0
    )
    induction_gap_cv = _finite_metric(
        normalized, errors, "induction_validation_gap_accuracy_cv", min_value=0.0
    )

    binding_screening_auc = _finite_metric(
        normalized,
        errors,
        "binding_intermediate_auc",
        min_value=0.0,
        max_value=1.0,
    )
    long_ctx = _finite_metric(
        normalized, errors, "robustness_long_ctx_combined_score", min_value=0.0
    )
    baseline_long_ctx = _finite_metric(
        normalized,
        errors,
        "champion_baseline_long_ctx_combined_score",
        positive=True,
    )

    held_pair = _finite_metric(
        normalized,
        errors,
        "ar_validation_held_pair_acc",
        min_value=0.0,
        max_value=1.0,
    )
    held_class = _finite_metric(
        normalized,
        errors,
        "ar_validation_held_class_acc",
        min_value=0.0,
        max_value=1.0,
    )
    learning_speed_score = normalized.get("ar_validation_learning_speed_score")
    if learning_speed_score is not None:
        ar_validation_speed = _finite_metric(
            normalized,
            errors,
            "ar_validation_learning_speed_score",
            min_value=0.0,
            max_value=1.0,
        )
    else:
        if normalized.get("ar_validation_steps_to_floor") is None:
            ar_validation_speed = 0.0
        else:
            ar_validation_steps = _finite_metric(
                normalized,
                errors,
                "ar_validation_steps_to_floor",
                positive=True,
            )
            baseline_ar_validation_steps = _finite_metric(
                normalized,
                errors,
                "champion_baseline_ar_validation_steps_to_floor",
                positive=True,
            )
            ar_validation_speed = (
                _clamp01(
                    (baseline_ar_validation_steps - ar_validation_steps)
                    / baseline_ar_validation_steps
                )
                if ar_validation_steps is not None
                and baseline_ar_validation_steps is not None
                else None
            )

    if errors["corrupt"]:
        fields = ",".join(sorted(set(errors["corrupt"])))
        return {
            **zero,
            "hard_failure_reason": f"corrupt_required_champion_metrics:{fields}",
        }
    if errors["missing"]:
        fields = ",".join(sorted(set(errors["missing"])))
        return {
            **zero,
            "hard_failure_reason": f"missing_required_champion_metrics:{fields}",
        }

    assert steps is not None
    assert baseline_steps is not None
    assert floor_quality_value is not None
    assert baseline_floor_quality is not None
    assert floor_std is not None
    assert baseline_floor_std is not None
    assert induction_screening_auc is not None
    assert induction_gap_cv is not None
    assert binding_screening_auc is not None
    assert long_ctx is not None
    assert baseline_long_ctx is not None
    assert held_pair is not None
    assert held_class is not None
    assert ar_validation_speed is not None

    steps_to_floor = _clamp01((baseline_steps - steps) / baseline_steps) * 10.0
    floor_quality = (
        _clamp01(
            (baseline_floor_quality - floor_quality_value) / baseline_floor_quality
        )
        * 10.0
    )
    floor_stability = _clamp01(1.0 - (floor_std / baseline_floor_std)) * 5.0
    induction_validation = (
        _clamp01((induction_screening_auc - 0.20) / 0.75) * 8.0
        + _clamp01(1.0 - induction_gap_cv) * 2.0
    )
    binding_long_context = 3.0 * _clamp01(binding_screening_auc) + 2.0 * _clamp01(
        long_ctx / baseline_long_ctx
    )
    ar_validation = (
        6.0 * _clamp01(held_pair)
        + 2.0 * _clamp01(held_class)
        + 2.0 * _clamp01(ar_validation_speed)
    )
    total = (
        steps_to_floor
        + floor_quality
        + floor_stability
        + induction_validation
        + binding_long_context
        + ar_validation
    )
    return {
        **zero,
        "steps_to_floor": steps_to_floor,
        "floor_quality": floor_quality,
        "floor_stability": floor_stability,
        "induction_validation": induction_validation,
        "binding_long_context": binding_long_context,
        "ar_validation": ar_validation,
        "total": total,
    }


def _champion_tiny_model_protocol_requested(kw: Dict[str, Any]) -> bool:
    version = str(kw.get("champion_tiny_model_protocol_version") or "").strip()
    return (
        bool(kw.get("champion_tiny_model_protocol"))
        or bool(kw.get("use_champion_tiny_model_score"))
        or version in {CHAMPION_TINY_MODEL_SCORE_V1, "v1", "tiny_model_v1"}
    )


def _apply_champion_tiny_model_hard_failure_gate(
    score: float,
    bd: Dict[str, Any],
    kw: Dict[str, Any],
) -> Optional[float]:
    if not _champion_tiny_model_protocol_requested(kw):
        return None

    champion = compute_champion_tiny_model_score_v1(**kw)
    bd["champion_steps_to_floor_score"] = champion["steps_to_floor"]
    bd["champion_floor_quality_score"] = champion["floor_quality"]
    bd["champion_floor_stability_score"] = champion["floor_stability"]
    bd["champion_induction_validation_score"] = champion["induction_validation"]
    bd["champion_binding_long_context_score"] = champion["binding_long_context"]
    bd["champion_ar_validation_score"] = champion["ar_validation"]
    bd["champion_tiny_model_score"] = champion["total"]
    bd["champion_tiny_model_protocol_version"] = champion["protocol_version"]
    bd["champion_hard_failure_reason"] = champion["hard_failure_reason"]

    if champion["hard_failure_reason"]:
        bd["_champion_tiny_model_hard_failure_gate"] = True
        bd["_v12_champion_eligibility_ceiling"] = _V12_CHAMPION_ELIGIBILITY_CEILING
        return min(score, _V12_CHAMPION_ELIGIBILITY_CEILING)
    bd["_champion_tiny_model_hard_failure_gate"] = False
    return score

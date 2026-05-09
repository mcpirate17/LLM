"""Shared capability-ranking metric fields and config helpers."""

from __future__ import annotations

from typing import Any, Mapping

AR_INTERMEDIATE_SCORE_FIELDS = (
    "ar_intermediate_train_pair_acc",
    "ar_intermediate_held_pair_acc",
    "ar_intermediate_held_class_acc",
    "ar_intermediate_pair_chance_acc",
    "ar_intermediate_class_chance_acc",
    "ar_intermediate_held_pair_lift",
    "ar_intermediate_held_class_lift",
    "ar_intermediate_early_held_pair_acc",
    "ar_intermediate_final_held_pair_acc",
    "ar_intermediate_best_held_pair_acc",
    "ar_intermediate_improvement",
    "ar_intermediate_slope_per_100_steps",
    "ar_intermediate_auc",
    "ar_intermediate_auc_lift",
    "ar_intermediate_steps_to_threshold",
    "ar_intermediate_diagnostic_score",
    "ar_intermediate_steps_trained",
)

BINDING_MULTISLOT_SCORE_FIELDS = (
    "binding_multislot_train_slot_acc",
    "binding_multislot_held_entity_slot_acc",
    "binding_multislot_held_entity_class_acc",
    "binding_multislot_two_plus_slots_acc",
    "binding_multislot_all_slots_acc",
    "binding_multislot_mixed_query_acc",
    "binding_multislot_mixed_two_plus_slots_acc",
    "binding_multislot_mixed_all_slots_acc",
    "binding_multislot_slot_chance_acc",
    "binding_multislot_class_chance_acc",
    "binding_multislot_two_plus_slots_chance_acc",
    "binding_multislot_all_slots_chance_acc",
    "binding_multislot_held_slot_lift",
    "binding_multislot_held_class_lift",
    "binding_multislot_two_plus_slots_lift",
    "binding_multislot_all_slots_lift",
    "binding_multislot_mixed_query_lift",
    "binding_multislot_mixed_two_plus_slots_lift",
    "binding_multislot_mixed_all_slots_lift",
    "binding_multislot_early_slot_acc",
    "binding_multislot_final_slot_acc",
    "binding_multislot_best_slot_acc",
    "binding_multislot_improvement",
    "binding_multislot_slope_per_100_steps",
    "binding_multislot_auc",
    "binding_multislot_auc_lift",
    "binding_multislot_steps_to_threshold",
    "binding_multislot_diagnostic_score",
    "binding_multislot_steps_trained",
)

CAPABILITY_RANKER_FIELDS = (
    "induction_intermediate_auc",
    "induction_intermediate_max_gap_acc",
    "induction_intermediate_gap_accuracies_json",
    "induction_intermediate_steps_trained",
    "induction_intermediate_status",
    "induction_intermediate_elapsed_ms",
    "induction_intermediate_protocol_version",
    "binding_intermediate_auc",
    "binding_intermediate_max_distance_acc",
    "binding_intermediate_distance_accuracies_json",
    "binding_intermediate_train_steps",
    "binding_intermediate_status",
    "binding_intermediate_elapsed_ms",
    "binding_intermediate_protocol_version",
    "induction_validation_auc",
    "induction_validation_max_gap_acc",
    "induction_validation_gap_accuracy_cv",
    "induction_validation_gap_accuracies_json",
    "induction_validation_steps_trained",
    "induction_validation_status",
    "induction_validation_elapsed_ms",
    "induction_validation_protocol_version",
    "ar_validation_metric_version",
    "ar_validation_final_acc",
    "ar_validation_held_pair_acc",
    "ar_validation_held_class_acc",
    "ar_validation_learning_curve_json",
    "ar_validation_steps_to_floor",
    "ar_validation_rank_score",
    "ar_validation_status",
    "ar_validation_elapsed_ms",
    "ar_intermediate_metric_version",
    *AR_INTERMEDIATE_SCORE_FIELDS,
    "ar_intermediate_learning_curve_json",
    "ar_intermediate_status",
    "ar_intermediate_elapsed_ms",
    "ar_intermediate_error",
    "binding_multislot_metric_version",
    *BINDING_MULTISLOT_SCORE_FIELDS,
    "binding_multislot_learning_curve_json",
    "binding_multislot_status",
    "binding_multislot_elapsed_ms",
    "binding_multislot_error",
)

CAPABILITY_RANKER_EVIDENCE_FIELDS = (
    "induction_intermediate_auc",
    "binding_intermediate_auc",
    "ar_intermediate_diagnostic_score",
    "binding_multislot_diagnostic_score",
    "induction_validation_auc",
    "ar_validation_rank_score",
)

CAPABILITY_RANKER_ENABLE_FLAGS = (
    "capability_ranking_run_intermediate",
    "capability_ranking_run_ar_intermediate",
    "capability_ranking_run_binding_multislot",
)


def capability_ranker_fields(data: Mapping[str, Any]) -> dict[str, Any]:
    """Extract columns owned by the capability-ranking tier."""
    return {key: data.get(key) for key in CAPABILITY_RANKER_FIELDS}


def has_capability_ranker_evidence(data: Mapping[str, Any]) -> bool:
    """Return true when a ranker produced an actual scoring metric."""
    return any(data.get(key) is not None for key in CAPABILITY_RANKER_EVIDENCE_FIELDS)


def available_capability_ranker_evidence_fields(columns: set[str]) -> tuple[str, ...]:
    return tuple(
        field for field in CAPABILITY_RANKER_EVIDENCE_FIELDS if field in columns
    )


def enable_capability_rankers(config: Any) -> Any:
    for attr in CAPABILITY_RANKER_ENABLE_FLAGS:
        if hasattr(config, attr):
            setattr(config, attr, True)
    return config

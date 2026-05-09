"""Canonical cascade probe metric names and physical DB rename catalog."""

from __future__ import annotations

from typing import Final


PROBE_METRIC_RENAMES: Final[dict[str, str]] = {
    # Legacy associative-recall probe kept only for historical display.
    "ar_auc": "ar_legacy_auc",
    "ar_final_acc": "ar_legacy_final_acc",
    "ar_timed_out": "ar_legacy_timed_out",
    "ar_above_chance": "ar_legacy_above_chance",
    # AR gate, formerly nano_ar_inv.
    "nano_ar_inv_metric_version": "ar_gate_metric_version",
    "nano_ar_inv_in_dist_pair_match_acc": "ar_gate_in_dist_pair_acc",
    "nano_ar_inv_in_dist_class_acc": "ar_gate_in_dist_class_acc",
    "nano_ar_inv_held_pair_match_acc": "ar_gate_held_pair_acc",
    "nano_ar_inv_held_class_acc": "ar_gate_held_class_acc",
    "nano_ar_inv_score": "ar_gate_score",
    "nano_ar_inv_status": "ar_gate_status",
    "nano_ar_inv_elapsed_ms": "ar_gate_elapsed_ms",
    "nano_ar_inv_train_steps_done": "ar_gate_train_steps_done",
    "nano_ar_inv_no_go": "ar_gate_no_go",
    # Screening induction/binding probes.
    "induction_auc": "induction_screening_auc",
    "induction_gap_accuracies_json": "induction_screening_gap_accuracies_json",
    "induction_probe_train_steps": "induction_screening_train_steps",
    "induction_probe_eval_examples": "induction_screening_eval_examples",
    "induction_probe_batch_size": "induction_screening_batch_size",
    "induction_probe_gaps_json": "induction_screening_gaps_json",
    "induction_probe_elapsed_ms": "induction_screening_elapsed_ms",
    "induction_probe_metric_version": "induction_screening_metric_version",
    "induction_probe_speed_mode": "induction_screening_speed_mode",
    "induction_probe_pool_size": "induction_screening_pool_size",
    "binding_auc": "binding_screening_auc",
    "binding_distance_accuracies_json": "binding_screening_distance_accuracies_json",
    "binding_probe_eval_examples": "binding_screening_eval_examples",
    "binding_probe_distances_json": "binding_screening_distances_json",
    "binding_probe_elapsed_ms": "binding_screening_elapsed_ms",
    "binding_composite": "binding_screening_composite",
    # Binding curriculum probe.
    "binding_auc_curriculum": "binding_curriculum_auc",
    "binding_distance_accuracies_curriculum_json": (
        "binding_curriculum_distance_accuracies_json"
    ),
    "binding_probe_curriculum_steps": "binding_curriculum_steps",
    "binding_probe_curriculum_elapsed_ms": "binding_curriculum_elapsed_ms",
    "binding_probe_curriculum_protocol_version": (
        "binding_curriculum_protocol_version"
    ),
    # Intermediate induction/binding probes, formerly v2 investigation.
    "induction_v2_investigation_auc": "induction_intermediate_auc",
    "induction_v2_investigation_max_gap_acc": ("induction_intermediate_max_gap_acc"),
    "induction_v2_investigation_gap_accuracies_json": (
        "induction_intermediate_gap_accuracies_json"
    ),
    "induction_v2_investigation_steps_trained": (
        "induction_intermediate_steps_trained"
    ),
    "induction_v2_investigation_status": "induction_intermediate_status",
    "induction_v2_investigation_elapsed_ms": ("induction_intermediate_elapsed_ms"),
    "induction_v2_investigation_protocol_version": (
        "induction_intermediate_protocol_version"
    ),
    "binding_v2_investigation_auc": "binding_intermediate_auc",
    "binding_v2_investigation_max_distance_acc": (
        "binding_intermediate_max_distance_acc"
    ),
    "binding_v2_investigation_distance_accuracies_json": (
        "binding_intermediate_distance_accuracies_json"
    ),
    "binding_v2_investigation_train_steps": "binding_intermediate_train_steps",
    "binding_v2_investigation_status": "binding_intermediate_status",
    "binding_v2_investigation_elapsed_ms": "binding_intermediate_elapsed_ms",
    "binding_v2_investigation_protocol_version": (
        "binding_intermediate_protocol_version"
    ),
    # Validation induction and AR probes.
    "champion_induction_v3_score": "champion_induction_validation_score",
    "induction_v3_auc": "induction_validation_auc",
    "induction_v3_max_gap_acc": "induction_validation_max_gap_acc",
    "induction_v3_gap_accuracy_cv": "induction_validation_gap_accuracy_cv",
    "induction_v3_gap_accuracies_json": ("induction_validation_gap_accuracies_json"),
    "induction_v3_steps_trained": "induction_validation_steps_trained",
    "induction_v3_status": "induction_validation_status",
    "induction_v3_elapsed_ms": "induction_validation_elapsed_ms",
    "induction_v3_protocol_version": "induction_validation_protocol_version",
    "champion_small_ar_score": "champion_ar_validation_score",
    "small_ar_champion_metric_version": "ar_validation_metric_version",
    "small_ar_champion_final_acc": "ar_validation_final_acc",
    "small_ar_champion_held_pair_match_acc": "ar_validation_held_pair_acc",
    "small_ar_champion_held_class_acc": "ar_validation_held_class_acc",
    "small_ar_champion_learning_curve_json": ("ar_validation_learning_curve_json"),
    "small_ar_champion_steps_to_floor": "ar_validation_steps_to_floor",
    "small_ar_champion_score": "ar_validation_rank_score",
    "small_ar_champion_status": "ar_validation_status",
    "small_ar_champion_elapsed_ms": "ar_validation_elapsed_ms",
    # Language-control probe ladder, formerly controlled_lang.
    "controlled_lang_metric_version": "language_control_metric_version",
    "controlled_lang_s05_sa_score": ("language_control_s05_sentence_assoc_score"),
    "controlled_lang_s05_nb_order_acc": ("language_control_s05_binding_order_acc"),
    "controlled_lang_s05_nb_score": "language_control_s05_binding_score",
    "controlled_lang_s10_sa_score": ("language_control_s10_sentence_assoc_score"),
    "controlled_lang_s10_nb_order_acc": ("language_control_s10_binding_order_acc"),
    "controlled_lang_s10_nb_score": "language_control_s10_binding_score",
    "controlled_lang_s10_checkpoints_json": ("language_control_s10_checkpoints_json"),
    "controlled_lang_inv_sa_score": (
        "language_control_investigation_sentence_assoc_score"
    ),
    "controlled_lang_inv_nb_order_acc": (
        "language_control_investigation_binding_order_acc"
    ),
    "controlled_lang_inv_nb_score": ("language_control_investigation_binding_score"),
    "controlled_lang_inv_checkpoints_json": (
        "language_control_investigation_checkpoints_json"
    ),
}


STATS_TABLE_RENAMES: Final[dict[str, str]] = {
    "avg_ar_auc": "avg_ar_legacy_auc",
    "avg_induction_auc": "avg_induction_screening_auc",
    "avg_binding_auc": "avg_binding_screening_auc",
    "avg_binding_composite": "avg_binding_screening_composite",
    "avg_induction_v2_investigation_auc": "avg_induction_intermediate_auc",
    "avg_binding_v2_investigation_auc": "avg_binding_intermediate_auc",
}


TABLE_RENAMES: Final[dict[str, dict[str, str]]] = {
    "program_results": PROBE_METRIC_RENAMES,
    "leaderboard": PROBE_METRIC_RENAMES,
    "template_stats": STATS_TABLE_RENAMES,
    "op_stats": STATS_TABLE_RENAMES,
    "motif_stats": STATS_TABLE_RENAMES,
    "slot_stats": STATS_TABLE_RENAMES,
}


def canonical_metric_name(name: str) -> str:
    """Return the cascade name for an old metric, or ``name`` if canonical."""

    return PROBE_METRIC_RENAMES.get(name, STATS_TABLE_RENAMES.get(name, name))

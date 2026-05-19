"""Shared leaderboard dashboard field groups."""

from __future__ import annotations

from typing import Final

from research.scientist.probe_metric_columns import MODERN_AR_BINDING_COLUMNS


def _prefixed_modern_probe_columns(*prefixes: str) -> tuple[str, ...]:
    return tuple(
        column
        for prefix in prefixes
        for column in MODERN_AR_BINDING_COLUMNS
        if column.startswith(prefix)
    )


CHAMPION_DASHBOARD_FIELDS: Final[tuple[str, ...]] = _prefixed_modern_probe_columns(
    "champion_",
    "induction_validation_",
    "ar_validation_",
)

V2_INVESTIGATION_DASHBOARD_FIELDS: Final[tuple[str, ...]] = (
    "induction_intermediate_auc",
    "induction_intermediate_max_gap_acc",
    "induction_intermediate_protocol_version",
    "binding_intermediate_auc",
    "binding_intermediate_max_distance_acc",
    "binding_intermediate_protocol_version",
)

_AR_INTERMEDIATE_DASHBOARD_FIELDS: Final[tuple[str, ...]] = (
    "ar_intermediate_metric_version",
    "ar_intermediate_diagnostic_score",
    "ar_intermediate_held_pair_acc",
    "ar_intermediate_held_pair_lift",
    "ar_intermediate_held_class_acc",
    "ar_intermediate_auc_lift",
    "ar_intermediate_best_held_pair_acc",
    "ar_intermediate_improvement",
    "ar_intermediate_status",
    "ar_intermediate_elapsed_ms",
)

_BINDING_MULTISLOT_DASHBOARD_FIELDS: Final[tuple[str, ...]] = (
    "binding_multislot_metric_version",
    "binding_multislot_diagnostic_score",
    "binding_multislot_held_entity_slot_acc",
    "binding_multislot_held_slot_lift",
    "binding_multislot_two_plus_slots_acc",
    "binding_multislot_two_plus_slots_lift",
    "binding_multislot_mixed_two_plus_slots_acc",
    "binding_multislot_mixed_two_plus_slots_lift",
    "binding_multislot_all_slots_acc",
    "binding_multislot_auc_lift",
    "binding_multislot_status",
    "binding_multislot_elapsed_ms",
)

INTERMEDIATE_SCREEN_DASHBOARD_FIELDS: Final[tuple[str, ...]] = (
    *_AR_INTERMEDIATE_DASHBOARD_FIELDS,
    *_BINDING_MULTISLOT_DASHBOARD_FIELDS,
    *_prefixed_modern_probe_columns("ar_curriculum_"),
)

PROGRAM_RESULT_DASHBOARD_ALIAS_FIELDS: Final[tuple[str, ...]] = (
    *CHAMPION_DASHBOARD_FIELDS,
    *V2_INVESTIGATION_DASHBOARD_FIELDS,
    *INTERMEDIATE_SCREEN_DASHBOARD_FIELDS,
)

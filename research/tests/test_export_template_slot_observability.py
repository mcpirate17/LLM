from __future__ import annotations

from research.tools.export_template_slot_observability import (
    build_slot_export_rows,
    build_template_export_rows,
)


def test_template_export_rows_include_strength_alignment() -> None:
    observability = {
        "all_templates": [
            {
                "name": "normalized_matmul",
                "actions": ["exploit"],
                "diagnosis": ["good induction"],
                "failure_reasons": ["stage1"],
            }
        ]
    }
    strength = {
        "normalized_matmul": {
            "adjusted_effect": 6.7,
            "confidence_tier": "medium",
            "support_graphs": 33,
            "matched_template_controls": 3,
            "artifact_flags": ["single_protocol"],
        }
    }

    rows = build_template_export_rows(observability, strength)

    assert rows[0]["strength_alignment"] == "aligned_positive_signal"
    assert rows[0]["strength_template_confidence_tier"] == "medium"
    assert rows[0]["actions"] == "exploit"
    assert rows[0]["strength_template_artifact_flags"] == "single_protocol"


def test_slot_export_rows_include_negative_pattern_alignment() -> None:
    observability = {
        "all_slots": [
            {
                "slot_key": "routed_bottleneck[0].slot1",
                "slot_classes": ["gate_core"],
                "top_selected_motif": "gate_progressive",
            }
        ]
    }
    strength = {
        "routed_bottleneck[0].slot1:gate_progressive": {
            "adjusted_effect": -5.4,
            "confidence_tier": "low",
            "support_graphs": 16,
            "artifact_flags": ["template_coupled"],
        }
    }

    rows = build_slot_export_rows(observability, strength)

    assert (
        rows[0]["strength_slot_component_key"]
        == "routed_bottleneck[0].slot1:gate_progressive"
    )
    assert rows[0]["strength_alignment"] == "likely_negative_pattern"
    assert rows[0]["slot_classes"] == "gate_core"

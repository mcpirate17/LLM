from __future__ import annotations

import json

from research.tools.audit_cheap_probe_predictors import (
    HEAD_SPECS,
    audit_heads,
    build_head_dataset,
    format_markdown,
)


def _graph(op_name: str, idx: int) -> str:
    return json.dumps(
        {
            "nodes": {
                "0": {"op_name": "input", "input_ids": []},
                "1": {"op_name": op_name, "input_ids": ["0"]},
                "2": {"op_name": "rmsnorm", "input_ids": ["1"]},
                "3": {"op_name": "linear_proj", "input_ids": ["2"]},
                "4": {"op_name": "add", "input_ids": ["0", "3"]},
            },
            "metadata": {"templates_used": [f"template_{idx % 3}"]},
        }
    )


def _synthetic_rows(n: int = 45) -> list[dict]:
    families = ("softmax_attention", "selective_scan", "rwkv_time_mixing")
    rows = []
    for i in range(n):
        strength = i / max(n - 1, 1)
        family_op = families[i % len(families)]
        rows.append(
            {
                "canonical_fingerprint": f"fp-{i}",
                "graph_json": _graph(family_op, i),
                "family": family_op,
                "latest_timestamp": float(i),
                "ar_gate_score": strength,
                "nano_induction_nearest_max_accuracy": max(0.0, strength - 0.1),
                "language_control_s05_sentence_assoc_score": min(1.0, strength + 0.05),
                "language_control_s05_binding_score": min(1.0, strength + 0.02),
                "language_control_s10_sentence_assoc_score": min(1.0, strength + 0.08),
                "language_control_s10_binding_score": min(1.0, strength + 0.04),
                "large_induction_intermediate_auc": strength,
                "large_binding_screening_auc": min(1.0, strength + 0.03),
                "large_binding_screening_composite": min(1.0, strength + 0.01),
                "large_blimp_overall_accuracy": 0.45 + 0.2 * strength,
                "large_ar_curriculum_auc_pair_final": max(0.0, strength - 0.05),
                "failure_target": 1.0 if strength < 0.35 else 0.0,
            }
        )
    return rows


def test_audit_heads_reports_each_required_head() -> None:
    report = audit_heads(
        _synthetic_rows(),
        min_samples=12,
        min_eval=4,
        min_family_holdout=5,
        n_estimators=8,
        top_features=5,
    )

    heads = {head["head"]: head for head in report["heads"]}
    assert set(heads) == {spec.name for spec in HEAD_SPECS}

    ar_head = heads["predict_ar_gate_from_graph"]
    assert ar_head["feature_mode"] == "graph"
    assert ar_head["sample_count"] == 45
    assert "temporal_holdout" in ar_head
    assert "binary_at_threshold" in ar_head["temporal_holdout"]
    assert "top_feature_importances" in ar_head["temporal_holdout"]
    assert "calibration_bins" in ar_head["temporal_holdout"]
    assert "stability_flags" in ar_head["temporal_holdout"]
    assert "stratified_diagnostics" in ar_head["temporal_holdout"]
    assert "model_comparison" in ar_head["temporal_holdout"]
    assert "operating_points" in ar_head["temporal_holdout"]
    assert {"balanced", "f1", "high_ppv", "high_npv"} <= set(
        ar_head["temporal_holdout"]["operating_points"]
    )

    nb10_binding = heads["predict_nb10_binding_from_graph"]
    assert nb10_binding["model_kind"] == "classifier"
    assert nb10_binding["target_columns"] == ["language_control_s10_binding_score"]
    assert nb10_binding["temporal_holdout"]["selected_model"]

    cheap_head = heads["predict_large_induction_from_cheap"]
    assert cheap_head["feature_mode"] == "cheap"
    assert cheap_head["leave_family_out"]["families_evaluated"] == 3

    markdown = format_markdown(report)
    assert "predict_ar_gate_from_graph" in markdown
    assert "PPV/NPV/accuracy" in markdown


def test_nb05_binding_and_joint_targets_do_not_use_sentence_assoc_as_binding() -> None:
    binding_spec = next(
        s for s in HEAD_SPECS if s.name == "predict_nb05_binding_from_graph"
    )
    joint_spec = next(
        s for s in HEAD_SPECS if s.name == "predict_nb05_joint_from_graph"
    )
    rows = [
        {
            "canonical_fingerprint": "fp-a",
            "graph_json": _graph("softmax_attention", 0),
            "family": "softmax_attention",
            "latest_timestamp": 1.0,
            "language_control_s05_binding_score": 0.25,
            "language_control_s05_sentence_assoc_score": 0.75,
        }
    ]

    features, y, kept = build_head_dataset(rows, binding_spec)

    assert len(features) == 1
    assert kept[0]["canonical_fingerprint"] == "fp-a"
    assert y.tolist() == [0.25]
    assert all(not name.startswith("language_control_") for name in features[0])

    _features, joint_y, _kept = build_head_dataset(rows, joint_spec)
    assert joint_y.tolist() == [0.25]

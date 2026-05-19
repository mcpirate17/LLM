from __future__ import annotations

import json
from pathlib import Path

from research.tools.dynamic_component_controlled_screening import (
    build_controlled_run_config,
    filtered_candidate_artifact,
    run_lowering_batch,
)


def _row(
    component_id: str,
    lowering: str,
    *,
    backward_passed: bool = True,
) -> dict:
    return {
        "proposed_template_name": component_id,
        "component_descriptor": {
            "component_id": component_id,
            "lowering": lowering,
        },
        "validation": {"backward_passed": backward_passed},
    }


def _write_candidates(path: Path) -> Path:
    payload = {
        "schema_version": "dynamic_component_candidates_v1",
        "metadata": {"source": "pytest"},
        "candidates": [
            _row("restore_a", "mixer_sidecar_restore_v1"),
            _row("router_a", "router_lane_blend_v1"),
            _row("restore_failed", "mixer_sidecar_restore_v1", backward_passed=False),
        ],
        "ready_for_registration": [
            _row("restore_a", "mixer_sidecar_restore_v1"),
            _row("router_a", "router_lane_blend_v1"),
            _row("restore_failed", "mixer_sidecar_restore_v1", backward_passed=False),
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_filtered_candidate_artifact_keeps_only_ready_lowering_rows(tmp_path: Path):
    source = _write_candidates(tmp_path / "candidates.json")
    output = tmp_path / "filtered.json"

    summary = filtered_candidate_artifact(
        source_path=source,
        output_path=output,
        lowering="mixer_sidecar_restore_v1",
    )

    artifact = json.loads(output.read_text(encoding="utf-8"))
    ready = artifact["ready_for_registration"]
    assert summary["ready_count"] == 1
    assert summary["example_components"] == ["restore_a"]
    assert artifact["metadata"]["controlled_lowering"] == "mixer_sidecar_restore_v1"
    assert [row["component_descriptor"]["component_id"] for row in ready] == [
        "restore_a"
    ]


def test_build_controlled_run_config_forces_dynamic_candidate_path(tmp_path: Path):
    config = build_controlled_run_config(
        n_programs=7,
        device="cpu",
        candidate_path=tmp_path / "filtered.json",
        dynamic_prob=1.0,
        dynamic_strength=0.0,
        max_candidates=4,
        composition_depth=2,
        stage1_steps=25,
        max_depth=28,
        max_ops=36,
    )

    assert config.n_programs == 7
    assert config.device == "cpu"
    assert config.use_dynamic_template_candidates is True
    assert config.dynamic_template_candidate_prob == 1.0
    assert config.dynamic_template_candidate_strength == 0.0
    assert config.dynamic_template_max_candidates == 4
    assert config.composition_depth == 2
    assert config.template_weights == {}
    assert config.op_weights == {}
    assert config.routing_mandatory is False
    assert config.gbm_prescreener_enabled is False
    assert config.stage1_steps == 25
    assert config.max_depth == 28
    assert config.max_ops == 36


def test_run_lowering_batch_dry_run_writes_artifact_and_config(tmp_path: Path):
    source = _write_candidates(tmp_path / "candidates.json")

    summary = run_lowering_batch(
        lowering="router_lane_blend_v1",
        n_programs=3,
        device="cpu",
        source_candidates=source,
        artifact_dir=tmp_path / "artifacts",
        dry_run=True,
    )

    artifact_path = Path(summary["artifact"]["output_path"])
    assert summary["dry_run"] is True
    assert artifact_path.exists()
    assert summary["artifact"]["ready_count"] == 1
    assert summary["config"]["controlled_dynamic_lowering"] == "router_lane_blend_v1"
    assert summary["config"]["dynamic_template_candidate_path"] == str(artifact_path)
    assert summary["config"]["dynamic_template_candidate_prob"] == 1.0

import json
import sqlite3
from pathlib import Path

from research.tools.dynamic_component_lowering_report import (
    build_dynamic_component_lowering_report,
)


def _write_db(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE program_results (
            result_id TEXT PRIMARY KEY,
            graph_json TEXT,
            stage1_passed INTEGER,
            loss_ratio REAL,
            timestamp REAL
        )
        """
    )
    rows = [
        (
            "linear-good",
            {
                "metadata": {
                    "dynamic_components_used": [
                        {
                            "component_id": "linear-a",
                            "lowering": "rmsnorm_chain_with_binary_skip",
                        }
                    ],
                    "dynamic_template_attempts": [{"status": "ok"}],
                }
            },
            1,
            0.4,
            2.0,
        ),
        (
            "branch-bad",
            {
                "metadata": {
                    "dynamic_components_used": [
                        {
                            "component_id": "branch-a",
                            "lowering": "mixer_sidecar_restore_v1",
                        }
                    ],
                    "dynamic_template_attempts": [{"status": "rolled_back"}],
                }
            },
            0,
            0.9,
            1.0,
        ),
        (
            "legacy",
            {
                "metadata": {
                    "dynamic_templates_used": [
                        {
                            "template_id": "legacy-template",
                            "component_descriptor": {
                                "lowering": "trunk_sidecar_merge_v1",
                                "component_id": "legacy-branch",
                            },
                        }
                    ]
                }
            },
            1,
            0.5,
            0.5,
        ),
    ]
    conn.executemany(
        """
        INSERT INTO program_results
            (result_id, graph_json, stage1_passed, loss_ratio, timestamp)
        VALUES (?, ?, ?, ?, ?)
        """,
        [(rid, json.dumps(graph), s1, loss, ts) for rid, graph, s1, loss, ts in rows],
    )
    conn.commit()
    conn.close()
    return path


def _write_candidates(path: Path) -> Path:
    payload = {
        "ready_for_registration": [
            {
                "promotion_score": 2.0,
                "component_descriptor": {
                    "component_id": "router-a",
                    "lowering": "router_lane_blend_v1",
                },
                "validation": {"backward_passed": True},
            },
            {
                "promotion_score": 3.0,
                "component_descriptor": {
                    "component_id": "restore-a",
                    "lowering": "mixer_sidecar_restore_v1",
                },
                "validation": {"backward_passed": True},
            },
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_dynamic_component_lowering_report_groups_history_and_candidates(
    tmp_path: Path,
) -> None:
    db = _write_db(tmp_path / "runs.db")
    candidates = _write_candidates(tmp_path / "candidates.json")
    output = tmp_path / "report.json"

    report = build_dynamic_component_lowering_report(
        db_path=db,
        candidate_path=candidates,
        output_path=output,
    )

    assert output.exists()
    historical = {row["lowering"]: row for row in report["historical"]["by_lowering"]}
    assert historical["rmsnorm_chain_with_binary_skip"]["stage1_rate"] == 1.0
    assert historical["mixer_sidecar_restore_v1"]["stage1_rate"] == 0.0
    assert historical["trunk_sidecar_merge_v1"]["n_components"] == 1
    assert report["dynamic_attempts"]["rollback_count"] == 1

    artifact = {
        row["lowering"]: row for row in report["candidate_artifact"]["by_lowering"]
    }
    assert artifact["router_lane_blend_v1"]["ready_count"] == 1
    assert artifact["mixer_sidecar_restore_v1"]["backward_validated_count"] == 1

    recommendations = {
        row["lowering"]: row for row in report["selection_recommendations"]
    }
    assert recommendations["router_lane_blend_v1"]["confidence"] == "prior"
    assert (
        recommendations["router_lane_blend_v1"]["recommended_selection_multiplier"]
        == 0.75
    )

import json
import sqlite3
from pathlib import Path

from research.tools.mine_component_rules import mine_component_rules


def _graph(ops: list[str], metadata: dict | None = None) -> str:
    nodes = {
        "0": {
            "id": 0,
            "op_name": "input",
            "input_ids": [],
            "is_input": True,
            "is_output": False,
        }
    }
    prev = 0
    for idx, op_name in enumerate(ops, start=1):
        nodes[str(idx)] = {
            "id": idx,
            "op_name": op_name,
            "input_ids": [prev],
            "is_input": False,
            "is_output": False,
        }
        prev = idx
    return json.dumps({"nodes": nodes, "metadata": metadata or {}})


def test_mine_component_rules_reads_compat_view_shape(tmp_path: Path) -> None:
    db_path = tmp_path / "runs.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE program_results_compat (
            result_id TEXT,
            graph_json TEXT,
            stage0_passed INTEGER,
            stage1_passed INTEGER,
            loss_ratio REAL,
            timestamp REAL
        )
        """
    )
    ops = [
        "rmsnorm",
        "selective_scan",
        "linear_proj",
        "gelu",
        "linear_proj",
        "add",
        "rmsnorm",
        "softmax_attention",
    ]
    conn.execute(
        "INSERT INTO program_results_compat VALUES (?, ?, ?, ?, ?, ?)",
        (
            "r1",
            _graph(
                ops,
                {
                    "templates_used": ["unit_component"],
                    "dynamic_components_used": [
                        {"component_id": "component_chain_test"}
                    ],
                },
            ),
            1,
            1,
            0.4,
            1.0,
        ),
    )
    conn.execute(
        "INSERT INTO program_results_compat VALUES (?, ?, ?, ?, ?, ?)",
        ("r2", _graph(ops), 1, 0, 0.9, 2.0),
    )
    conn.commit()
    conn.close()

    report = mine_component_rules(
        db_path=db_path,
        limit=10,
        min_window_ops=8,
        min_support=1,
    )

    assert report["summary"]["graphs_parsed"] == 2
    assert report["summary"]["stage1_graphs"] == 1
    assert report["summary"]["dynamic_template_rows"] == 1
    assert report["template_counts"]["unit_component"] == 1
    assert report["candidate_windows"]
    assert report["candidate_windows"][0]["lowered_op_count"] >= 8

from __future__ import annotations

import sqlite3
from pathlib import Path

from research.tests._ml_corpus_test_support import create_test_db


def _linear_graph(*ops: str) -> str:
    nodes = {
        "0": {
            "id": 0,
            "op_name": "input",
            "input_ids": [],
            "config": {},
            "is_input": True,
            "is_output": False,
        }
    }
    prev_id = "0"
    for idx, op in enumerate(ops, start=1):
        node_id = str(idx)
        nodes[node_id] = {
            "id": idx,
            "op_name": op,
            "input_ids": [int(prev_id)],
            "config": {},
            "is_input": False,
            "is_output": False,
        }
        prev_id = node_id
    nodes[prev_id]["is_output"] = True
    return (
        "{"
        f'"nodes":{__import__("json").dumps(nodes, sort_keys=True)},'
        '"input_node_id":0,'
        f'"output_node_id":{prev_id},'
        '"metadata":{"templates_used":["unit_test"]}'
        "}"
    )


def _branch_graph() -> str:
    return (
        "{"
        '"nodes":{'
        '"0":{"id":0,"op_name":"input","input_ids":[],"config":{},"is_input":true,"is_output":false},'
        '"1":{"id":1,"op_name":"layernorm","input_ids":[0],"config":{},"is_input":false,"is_output":false},'
        '"2":{"id":2,"op_name":"gelu","input_ids":[1],"config":{},"is_input":false,"is_output":false},'
        '"3":{"id":3,"op_name":"swiglu","input_ids":[1],"config":{},"is_input":false,"is_output":false},'
        '"4":{"id":4,"op_name":"add","input_ids":[2,3],"config":{},"is_input":false,"is_output":true}'
        "},"
        '"input_node_id":0,'
        '"output_node_id":4,'
        '"metadata":{"templates_used":["branch_test"]}'
        "}"
    )


def test_extract_graph_segments_linear_paths() -> None:
    from research.scientist.intelligence.graph_segments import extract_graph_segments

    extraction = extract_graph_segments(
        _linear_graph("layernorm", "gelu", "linear_proj", "add"),
        min_len=3,
        max_len=6,
    )
    assert extraction.count_map == {
        "seg_p3:gelu>linear_proj>add": 1,
        "seg_p3:layernorm>gelu>linear_proj": 1,
        "seg_p4:layernorm>gelu>linear_proj>add": 1,
    }
    assert extraction.presence_set == frozenset(extraction.count_map)


def test_extract_graph_segments_branching_paths() -> None:
    from research.scientist.intelligence.graph_segments import extract_graph_segments

    extraction = extract_graph_segments(_branch_graph(), min_len=3, max_len=6)
    assert extraction.count_map == {
        "seg_p3:layernorm>gelu>add": 1,
        "seg_p3:layernorm>swiglu>add": 1,
    }


def test_load_stage05_native_segment_corpus_aggregates_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    from research.scientist.intelligence import graph_segments

    db_path = tmp_path / "segment_corpus.sqlite3"
    create_test_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute("ALTER TABLE program_results ADD COLUMN binding_auc REAL")
    conn.execute("ALTER TABLE program_results ADD COLUMN induction_auc REAL")
    conn.execute("ALTER TABLE program_results ADD COLUMN hellaswag_acc REAL")
    shared_graph = _linear_graph("layernorm", "gelu", "linear_proj", "add")
    conn.execute(
        """
        INSERT INTO program_results (
            result_id, graph_json, graph_fingerprint, fingerprint_json,
            novelty_score, structural_novelty, loss_ratio, wikitext_perplexity,
            stage0_passed, stage05_passed, stage1_passed, timestamp,
            trust_label, comparability_label, binding_auc, induction_auc, hellaswag_acc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "r1",
            shared_graph,
            "fp1",
            '{"isotropy": 0.1}',
            0.0,
            0.0,
            0.9,
            11.0,
            1,
            1,
            0,
            1.0,
            "candidate_grade",
            "candidate_comparable",
            None,
            0.0,
            None,
        ),
    )
    conn.execute(
        """
        INSERT INTO program_results (
            result_id, graph_json, graph_fingerprint, fingerprint_json,
            novelty_score, structural_novelty, loss_ratio, wikitext_perplexity,
            stage0_passed, stage05_passed, stage1_passed, timestamp,
            trust_label, comparability_label, binding_auc, induction_auc, hellaswag_acc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "r2",
            shared_graph,
            "fp2",
            '{"isotropy": 0.2}',
            0.0,
            0.0,
            0.4,
            7.0,
            1,
            1,
            1,
            2.0,
            "candidate_grade",
            "candidate_comparable",
            0.11,
            0.02,
            0.3,
        ),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(
        graph_segments, "_is_native_safe_graph", lambda _graph_json: True
    )

    rows = graph_segments.load_stage05_native_segment_corpus(db_path)
    assert len(rows) == 1
    row = rows[0]
    assert row.n_rows == 2
    assert row.stage1_any_passed is True
    assert row.stage1_pass_rate == 0.5
    assert row.loss_ratio_best == 0.4
    assert row.wikitext_perplexity_best == 7.0
    assert row.binding_auc == 0.11
    assert row.induction_auc == 0.02
    assert row.hellaswag_acc == 0.3
    assert row.all_three_positive is True


def test_evaluate_feature_families_smoke() -> None:
    from research.scientist.intelligence.graph_segments import (
        SegmentCorpusRow,
        evaluate_feature_families,
    )

    rows = [
        SegmentCorpusRow(
            canonical_fingerprint=f"fp{i}",
            graph_json=_linear_graph(
                "layernorm",
                "gelu" if i % 2 == 0 else "swiglu",
                "linear_proj",
                "add",
            ),
            n_rows=1,
            latest_timestamp=float(i),
            stage1_any_passed=bool(i % 2 == 0),
            stage1_pass_rate=1.0 if i % 2 == 0 else 0.0,
            loss_ratio_best=0.2 + (0.05 * i),
            wikitext_perplexity_best=5.0 + i,
            binding_auc=0.1 if i % 2 == 0 else 0.0,
            induction_auc=0.05 if i % 3 == 0 else 0.0,
            hellaswag_acc=0.3 if i % 2 == 0 else 0.0,
            binding_positive=bool(i % 2 == 0),
            induction_positive=bool(i % 3 == 0),
            hellaswag_positive=bool(i % 2 == 0),
            all_three_positive=bool(i % 6 == 0),
        )
        for i in range(24)
    ]

    report = evaluate_feature_families(rows, min_support=2, seed=0)
    assert report["n_graphs"] == 24
    assert report["n_fragment_features"] > 0
    assert "stage1_any_passed" in report["binary"]
    assert "loss_ratio_best" in report["continuous"]

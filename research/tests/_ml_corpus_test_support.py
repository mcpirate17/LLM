from __future__ import annotations

import sqlite3
from pathlib import Path


def create_test_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE program_results (
            result_id TEXT,
            graph_json TEXT,
            graph_fingerprint TEXT,
            fingerprint_json TEXT,
            novelty_score REAL,
            structural_novelty REAL,
            loss_ratio REAL,
            wikitext_perplexity REAL,
            stage0_passed INTEGER,
            stage05_passed INTEGER,
            stage1_passed INTEGER,
            timestamp REAL
        );

        CREATE TABLE leaderboard (
            result_id TEXT,
            investigation_loss_ratio REAL,
            tier TEXT
        );
        """
    )
    conn.close()


def graph_json(metadata: str, *, middle_op: str = "layernorm") -> str:
    return (
        "{"
        '"model_dim":256,'
        '"nodes":{'
        '"0":{"id":0,"op_name":"input","input_ids":[],"output_shape":{"batch":"B","seq":"S","dim":256},"config":{},"is_input":true,"is_output":false},'
        f'"1":{{"id":1,"op_name":"{middle_op}","input_ids":[0],"output_shape":{{"batch":"B","seq":"S","dim":256}},"config":{{}},"is_input":false,"is_output":false}},'
        '"2":{"id":2,"op_name":"add","input_ids":[0,1],"output_shape":{"batch":"B","seq":"S","dim":256},"config":{},"is_input":false,"is_output":true}'
        "},"
        '"input_node_id":0,'
        '"output_node_id":2,'
        f'"metadata":{metadata}'
        "}"
    )

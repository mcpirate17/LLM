import sqlite3

from research.tools.exact_graph_replay import _expand_replays, _fetch_source_rows


def test_expand_replays_repeats_each_source_in_order():
    rows = [
        {"result_id": "a"},
        {"result_id": "b"},
    ]

    expanded = _expand_replays(rows, 2)

    assert [row["result_id"] for row in expanded] == ["a", "a", "b", "b"]
    assert [row["replay_index"] for row in expanded] == [0, 1, 0, 1]


def test_fetch_source_rows_skips_malformed_graph_json(tmp_path):
    db_path = tmp_path / "replay.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """CREATE TABLE program_results (
            result_id TEXT PRIMARY KEY,
            graph_json TEXT,
            graph_fingerprint TEXT,
            loss_ratio REAL,
            stage1_passed INTEGER,
            stage05_passed INTEGER,
            timestamp REAL
        )"""
    )
    conn.execute(
        """INSERT INTO program_results
           (result_id, graph_json, graph_fingerprint, loss_ratio, stage1_passed, stage05_passed, timestamp)
           VALUES ('good', '{"nodes":{"0":{"op_name":"input","input_ids":[]}}}', 'fp_good', 0.5, 1, 1, 1.0)"""
    )
    conn.execute(
        """INSERT INTO program_results
           (result_id, graph_json, graph_fingerprint, loss_ratio, stage1_passed, stage05_passed, timestamp)
           VALUES ('bad', '{"nodes":{"0":{"op_name":"input"}', 'fp_bad', 0.9, 0, 1, 2.0)"""
    )
    conn.commit()
    conn.close()

    rows = _fetch_source_rows(db_path, ["good", "bad"])

    assert [row["result_id"] for row in rows] == ["good"]


def test_fetch_source_rows_dedupes_duplicate_fingerprints(tmp_path):
    db_path = tmp_path / "replay_dups.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """CREATE TABLE program_results (
            result_id TEXT PRIMARY KEY,
            graph_json TEXT,
            graph_fingerprint TEXT,
            loss_ratio REAL,
            stage1_passed INTEGER,
            stage05_passed INTEGER,
            timestamp REAL
        )"""
    )
    conn.execute(
        """INSERT INTO program_results
           (result_id, graph_json, graph_fingerprint, loss_ratio, stage1_passed, stage05_passed, timestamp)
           VALUES ('first', '{"nodes":{"0":{"op_name":"input","input_ids":[]}}}', 'fp_same', 0.5, 1, 1, 1.0)"""
    )
    conn.execute(
        """INSERT INTO program_results
           (result_id, graph_json, graph_fingerprint, loss_ratio, stage1_passed, stage05_passed, timestamp)
           VALUES ('second', '{"nodes":{"0":{"op_name":"input","input_ids":[]}}}', 'fp_same', 0.4, 1, 1, 2.0)"""
    )
    conn.commit()
    conn.close()

    rows = _fetch_source_rows(db_path, ["first", "second"])

    assert [row["result_id"] for row in rows] == ["first"]

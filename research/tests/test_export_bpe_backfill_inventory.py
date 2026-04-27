import json
import sqlite3

from research.tools.export_bpe_backfill_inventory import export_inventory


def _init_db(path):
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE program_results(
                result_id TEXT PRIMARY KEY,
                timestamp REAL,
                graph_fingerprint TEXT,
                graph_json TEXT,
                loss_ratio REAL,
                screening_wikitext_metric_version TEXT,
                wikitext_perplexity REAL,
                tinystories_perplexity REAL,
                hellaswag_acc REAL,
                blimp_overall_accuracy REAL,
                trust_label TEXT,
                comparability_label TEXT
            );
            CREATE TABLE leaderboard(
                entry_id TEXT PRIMARY KEY,
                result_id TEXT,
                tier TEXT,
                composite_score REAL
            );
            """
        )
        conn.execute(
            """
            INSERT INTO program_results VALUES
            ('stale-off', 10, 'fp-stale-off', '{"ops":["x"]}', 0.5, NULL, 12, NULL, 0.2, NULL, 'runtime_observation', 'partial'),
            ('fresh-off', 11, 'fp-fresh-off', '{"ops":["x"]}', 0.4, 'bpe_eval_v1', 10, 20, 0.25, 0.52, 'candidate_grade', 'candidate_comparable'),
            ('stale-on', 12, 'fp-stale-on', '{"ops":["x"]}', 0.3, NULL, 12, NULL, 0.2, NULL, 'candidate_grade', 'candidate_comparable')
            """
        )
        conn.execute(
            "INSERT INTO leaderboard VALUES ('entry-stale-on', 'stale-on', 'validation', 100.0)"
        )


def test_export_inventory_defaults_to_off_leaderboard_stale_fingerprints(tmp_path):
    db_path = tmp_path / "inventory.db"
    output_path = tmp_path / "inventory.json"
    _init_db(db_path)

    result = export_inventory(
        db_path,
        output_path=output_path,
        scope="off_leaderboard",
        limit=None,
    )

    payload = json.loads(output_path.read_text())
    assert result["fingerprint_count"] == 1
    assert payload["rows"][0]["graph_fingerprint"] == "fp-stale-off"


def test_export_inventory_can_select_leaderboard_scope(tmp_path):
    db_path = tmp_path / "inventory.db"
    output_path = tmp_path / "inventory.json"
    _init_db(db_path)

    export_inventory(
        db_path,
        output_path=output_path,
        scope="leaderboard",
        limit=None,
    )

    payload = json.loads(output_path.read_text())
    assert [row["graph_fingerprint"] for row in payload["rows"]] == ["fp-stale-on"]

from __future__ import annotations

import sqlite3

from research.tools.audit_component_scaffold_coverage import build_audit_report


def test_build_audit_report_flags_missing_scaffold_ops(tmp_path):
    db_path = tmp_path / "component_profiles.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE op_profiles (op_name TEXT PRIMARY KEY)")
        conn.executemany(
            "INSERT INTO op_profiles(op_name) VALUES (?)",
            [
                ("cascade",),
                ("route_lanes",),
                ("entropy_score",),
                ("relu",),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    report = build_audit_report(db_path)

    assert report["counts"]["profiled_ops_raw"] == 4
    assert "arch_router" in report["missing_canonical_ops"]
    assert report["scaffoldable_missing_ops"]["arch_router"] == "gpt2_replace"
    assert "learned_token_gate" not in report["missing_canonical_ops"]

import sqlite3

from research.tools.db_split_readiness import (
    build_report,
    db_pointer_summary,
    graph_json_stats,
    legacy_db_references,
    local_split_bundles,
)


def _make_runs_db(path):
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE program_results(
                result_id TEXT PRIMARY KEY,
                graph_json TEXT,
                data_provenance_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE experiments(
                experiment_id TEXT PRIMARY KEY,
                config_json TEXT,
                results_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE entries(
                entry_id TEXT PRIMARY KEY,
                metadata_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE healer_tasks(
                task_id TEXT PRIMARY KEY,
                trigger_payload_json TEXT,
                result_json TEXT
            )
            """
        )
        conn.execute("CREATE TABLE notebook_artifacts(artifact_id TEXT PRIMARY KEY)")
        conn.execute(
            """
            INSERT INTO program_results
            VALUES ('r1', '{"nodes":[1]}', '{"source":"inline"}')
            """
        )
        conn.execute(
            """
            INSERT INTO program_results
            VALUES ('r2', '{"nodes":[1,2]}', '{"source":"inline"}')
            """
        )
        conn.execute("INSERT INTO experiments VALUES ('e1', '{}', '{}')")
        conn.execute(
            """
            INSERT INTO entries
            VALUES ('entry1', '{"_notebook_artifact":"a1"}')
            """
        )
        conn.execute(
            """
            INSERT INTO healer_tasks
            VALUES ('task1', '{"_notebook_artifact":"a2"}', '{"ok":true}')
            """
        )
        conn.execute("INSERT INTO notebook_artifacts VALUES ('a1')")


def test_graph_json_stats_counts_inline_payloads(tmp_path):
    db_path = tmp_path / "runs.db"
    _make_runs_db(db_path)

    stats = graph_json_stats(db_path)

    assert stats["available"] is True
    assert stats["non_empty_rows"] == 2
    assert stats["total_bytes"] == len('{"nodes":[1]}') + len('{"nodes":[1,2]}')
    assert stats["max_bytes"] == len('{"nodes":[1,2]}')


def test_pointer_summary_separates_sensitive_and_expected_pointers(tmp_path):
    db_path = tmp_path / "runs.db"
    _make_runs_db(db_path)

    summary = db_pointer_summary(db_path)

    assert summary["notebook_artifact_rows"] == 1
    assert summary["sensitive_pointer_counts"] == {
        "experiments.config_json": 0,
        "experiments.results_json": 0,
        "program_results.data_provenance_json": 0,
    }
    assert summary["expected_pointer_counts"]["entries.metadata_json"] == 1
    assert summary["expected_pointer_counts"]["healer_tasks.trigger_payload_json"] == 1


def test_legacy_reference_scan_flags_unapproved_tool_defaults(tmp_path):
    project = tmp_path / "repo"
    tool_dir = project / "research" / "tools"
    tool_dir.mkdir(parents=True)
    (tool_dir / "active_tool.py").write_text(
        'DEFAULT_DB = "research/lab_notebook.db"\n', encoding="utf-8"
    )
    (tool_dir / "restore_lab_notebook.py").write_text(
        'DEFAULT_DB = "research/lab_notebook.db"\n', encoding="utf-8"
    )

    refs = legacy_db_references(project)

    assert refs["total"] == 2
    assert refs["unapproved_total"] == 1
    assert [ref["path"] for ref in refs["references"]] == [
        "research/tools/active_tool.py",
        "research/tools/restore_lab_notebook.py",
    ]
    assert refs["references"][1]["category"] == "legacy_restore"


def test_local_split_bundles_reports_without_deleting(tmp_path):
    project = tmp_path / "repo"
    upload = project / "research" / "tmp" / "db-backup-upload" / "stamp"
    upload.mkdir(parents=True)
    bundle = upload / "db-backups.tar.zst"
    bundle.write_bytes(b"bundle")

    found = local_split_bundles(project / "research", project)

    assert found == [
        {
            "path": "research/tmp/db-backup-upload/stamp/db-backups.tar.zst",
            "bytes": 6,
        }
    ]
    assert bundle.exists()


def test_build_report_surfaces_retirement_blockers(tmp_path):
    project = tmp_path / "repo"
    research = project / "research"
    tools = research / "tools"
    tools.mkdir(parents=True)
    runs_db = research / "runs.db"
    lab_db = research / "lab_notebook.db"
    _make_runs_db(runs_db)
    lab_db.write_bytes(b"legacy")
    (tools / "active_tool.py").write_text(
        'DEFAULT_DB = "research/lab_notebook.db"\n', encoding="utf-8"
    )

    report = build_report(
        project_root=project,
        runs_db=runs_db,
        lab_db=lab_db,
        artifact_dir=research / "artifacts" / "notebook",
        runtime_events_dir=research / "runtime_events",
    )

    assert report["databases"]["runs"]["exists"] is True
    assert report["databases"]["legacy_lab"]["exists"] is True
    assert "unapproved_lab_notebook_db_tool_references" in report["retirement_blockers"]

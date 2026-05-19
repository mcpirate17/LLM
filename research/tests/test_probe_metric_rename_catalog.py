import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

from research.scientist.probe_metric_names import (
    PROBE_METRIC_RENAMES,
    TABLE_RENAMES,
    canonical_metric_name,
)
from research.scientist.probe_metric_columns import MODERN_AR_BINDING_COLUMNS
from research.meta_analysis import metadata_db
from research.tools.cross_exp_probe_merge import MERGE_COLUMNS_BY_FAMILY
from research.tools.rename_probe_metrics_cascade import (
    apply_renames,
    plan_renames,
    verify_no_pending_renames,
)


def test_probe_metric_rename_catalog_is_one_to_one():
    assert canonical_metric_name("nano_ar_inv_score") == "ar_gate_score"
    assert (
        canonical_metric_name("small_ar_champion_score") == "ar_validation_rank_score"
    )
    assert canonical_metric_name("unknown_metric") == "unknown_metric"
    assert len(set(PROBE_METRIC_RENAMES.values())) == len(PROBE_METRIC_RENAMES)


def test_physical_probe_metric_rename_preserves_values():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE program_results (
            result_id TEXT PRIMARY KEY,
            nano_ar_inv_score REAL,
            small_ar_champion_score REAL,
            induction_v2_investigation_auc REAL,
            binding_composite REAL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO program_results (
            result_id,
            nano_ar_inv_score,
            small_ar_champion_score,
            induction_v2_investigation_auc,
            binding_composite
        ) VALUES ('r1', 0.75, 4.2, 0.31, 0.12)
        """
    )

    actions = plan_renames(conn)
    assert {action.old for action in actions} == {
        "nano_ar_inv_score",
        "small_ar_champion_score",
        "induction_v2_investigation_auc",
        "binding_composite",
    }

    apply_renames(conn, actions)
    verify_no_pending_renames(conn)
    row = conn.execute(
        """
        SELECT ar_gate_score,
               ar_validation_rank_score,
               induction_intermediate_auc,
               binding_screening_composite
        FROM program_results
        WHERE result_id = 'r1'
        """
    ).fetchone()
    assert row["ar_gate_score"] == pytest.approx(0.75)
    assert row["ar_validation_rank_score"] == pytest.approx(4.2)
    assert row["induction_intermediate_auc"] == pytest.approx(0.31)
    assert row["binding_screening_composite"] == pytest.approx(0.12)


def test_physical_probe_metric_rename_blocks_collisions():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE leaderboard (
            entry_id TEXT PRIMARY KEY,
            nano_ar_inv_score REAL,
            ar_gate_score REAL
        )
        """
    )

    with pytest.raises(RuntimeError, match="both nano_ar_inv_score and ar_gate_score"):
        plan_renames(conn)


def test_stats_table_rename_catalog_covers_feedback_tables():
    expected_tables = {"template_stats", "op_stats", "motif_stats", "slot_stats"}
    assert expected_tables.issubset(TABLE_RENAMES)
    for table in expected_tables:
        assert (
            TABLE_RENAMES[table]["avg_induction_auc"] == "avg_induction_screening_auc"
        )
        assert TABLE_RENAMES[table]["avg_binding_v2_investigation_auc"] == (
            "avg_binding_intermediate_auc"
        )


def test_modern_ar_binding_column_group_stays_shared():
    start = metadata_db._OUTCOME_COLUMNS.index("ar_gate_metric_version")
    end = metadata_db._OUTCOME_COLUMNS.index("hellaswag_acc")

    assert metadata_db._OUTCOME_COLUMNS[start:end] == MODERN_AR_BINDING_COLUMNS
    assert MERGE_COLUMNS_BY_FAMILY["modern_ar_binding"] == MODERN_AR_BINDING_COLUMNS
    assert MODERN_AR_BINDING_COLUMNS[-1] == "champion_hard_failure_reason"


def test_legacy_probe_names_are_allowlisted_to_migration_files():
    repo = Path(__file__).resolve().parents[2]
    allowlist = {
        "research/scientist/probe_metric_names.py",
        "research/tests/test_probe_metric_rename_catalog.py",
    }
    stale_terms = (
        "nano_ar_inv",
        "small_ar_champion",
        "medium_ar_",
        "multi_blank_binding",
        "induction_v2_investigation",
        "binding_v2_investigation",
        "controlled_lang",
        "champion_small_ar",
        "binding_composite",
        "binding_auc",
        "induction_auc",
    )
    search_paths = (
        "research/scientist",
        "research/eval",
        "research/tools",
        "research/dashboard/src",
        "research/tests",
    )
    suffixes = ("py", "js", "jsx", "md", "yaml", "yml")
    pathspecs = [f"{root}/**/*.{ext}" for root in search_paths for ext in suffixes]

    if shutil.which("git") is None:
        pytest.skip("git not available")

    pattern = "|".join(stale_terms)
    proc = subprocess.run(
        ["git", "grep", "-l", "--untracked", "-E", pattern, "--", *pathspecs],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    # git grep exits 1 when no matches found — treat as success.
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"git grep failed: {proc.stderr}")

    offenders = sorted(
        line for line in proc.stdout.splitlines() if line and line not in allowlist
    )
    assert offenders == []

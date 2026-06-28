from __future__ import annotations

from pathlib import Path

from research.eval.ar_validation import (
    DEFAULT_STABLE_SEED_COUNT,
    STABLE_AR_VALIDATION_PROTOCOL,
    make_ar_validation_config_from_args,
)
from research.stats import wilson_score_interval
from research.tools.db_backup import backup_database


def test_wilson_score_interval_handles_empty_and_bounds() -> None:
    assert wilson_score_interval(0, 0) == (0.0, 0.0)

    lo, hi = wilson_score_interval(6, 10)

    assert 0.31 < lo < 0.32
    assert 0.83 < hi < 0.84


def test_ar_validation_config_from_args_preserves_legacy_mode() -> None:
    cfg = make_ar_validation_config_from_args(
        legacy_v2=True,
        timeout_s=12.5,
        train_steps=345,
    )

    assert cfg.protocol == "integer_v2"
    assert cfg.copy_model is False
    assert cfg.auto_size_budget is False
    assert cfg.deterministic_episode_bank is False
    assert cfg.seed_count == 1
    assert cfg.timeout_s == 12.5
    assert cfg.train_steps == 345


def test_ar_validation_config_from_args_uses_stable_defaults() -> None:
    cfg = make_ar_validation_config_from_args(legacy_v2=False, timeout_s=30.0)

    assert cfg.protocol == STABLE_AR_VALIDATION_PROTOCOL
    assert cfg.copy_model is True
    assert cfg.auto_size_budget is True
    assert cfg.deterministic_episode_bank is True
    assert cfg.seed_count == DEFAULT_STABLE_SEED_COUNT
    assert cfg.timeout_s == 30.0


def test_backup_database_dry_run_reports_targets(tmp_path: Path) -> None:
    db_path = tmp_path / "runs.db"
    db_path.write_text("db", encoding="utf-8")
    project_root = tmp_path / "repo"
    google_root = tmp_path / "google"

    targets = backup_database(
        db_path,
        "pre_test",
        project_root=project_root,
        google_backup_root=google_root,
        dry_run=True,
    )

    assert targets == {
        "local": str(project_root / "research/db_backups/pre_test/runs.db"),
        "google_drive": str(google_root / "pre_test/runs.db"),
    }
    assert not (project_root / "research/db_backups/pre_test").exists()
    assert not (google_root / "pre_test").exists()


def test_backup_database_copies_to_local_and_google(tmp_path: Path) -> None:
    db_path = tmp_path / "runs.db"
    db_path.write_text("db", encoding="utf-8")
    project_root = tmp_path / "repo"
    google_root = tmp_path / "google"

    targets = backup_database(
        db_path,
        "pre_test",
        project_root=project_root,
        google_backup_root=google_root,
    )

    assert Path(targets["local"]).read_text(encoding="utf-8") == "db"
    assert Path(targets["google_drive"]).read_text(encoding="utf-8") == "db"

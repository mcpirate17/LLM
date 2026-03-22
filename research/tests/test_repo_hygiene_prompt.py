from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = REPO_ROOT / "research"


def test_cutover_plan_exists_for_readme_reference() -> None:
    plan_path = RESEARCH_ROOT / "CUTOVER_REMOVAL_PLAN.md"
    assert plan_path.is_file(), f"Missing README-linked doc: {plan_path}"


def test_archive_is_quarantined_outside_research_package() -> None:
    active_archive = RESEARCH_ROOT / "tools" / "archive"
    quarantined_archive = REPO_ROOT / "archive" / "research-tools" / "README.md"
    assert not active_archive.exists(), (
        f"Archive scripts leaked back into active package: {active_archive}"
    )
    assert quarantined_archive.is_file(), (
        f"Missing archive quarantine README: {quarantined_archive}"
    )


def test_generated_runtime_artifacts_are_not_present_in_research_tree() -> None:
    forbidden_paths = [
        RESEARCH_ROOT / ":memory:",
        RESEARCH_ROOT / ":memory:-shm",
        RESEARCH_ROOT / ":memory:-wal",
        RESEARCH_ROOT
        / "runtime"
        / "native"
        / "cython"
        / "aria_bridge.cpython-312-x86_64-linux-gnu.so",
    ]
    present = [str(path) for path in forbidden_paths if path.exists()]
    assert not present, (
        f"Generated runtime artifacts should not live in research/: {present}"
    )


def test_readme_mentions_quarantined_archive_location() -> None:
    readme = (RESEARCH_ROOT / "README.md").read_text(encoding="utf-8")
    assert "../archive/research-tools/" in readme

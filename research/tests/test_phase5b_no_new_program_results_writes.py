"""Phase 5b Stage 4 lock-door test.

Counts every direct write to the legacy ``program_results`` table across
production code and asserts the count matches the snapshot captured at
the time the triggers were installed (2026-05-10 cycle 3). The intent is
to make further accumulation of legacy writes impossible without
deliberate sign-off:

- **Bumping a count** = a new ``UPDATE/INSERT/DELETE program_results``
  was added. Either retarget it to ``graph_runs`` (the canonical
  storage) or expand the whitelist with reviewer approval.
- **Dropping a count** = a retarget happened. Decrement the whitelist
  and the test re-passes.

When the whitelist reaches zero across the board, Phase 5b Stage 3 (drop
the legacy table) becomes safe to execute. Until then, the
``_gn_sync_pr_update_to_runs`` / ``_gn_sync_pr_delete_to_runs`` /
``_gn_sync_pr_to_runs`` triggers keep ``graph_runs`` in sync, but the
goal is to retire the legacy write surface entirely.

Tests in ``research/tests/`` and notebook bootstrap DDL in
``notebook_core.py`` are excluded — they're either fixtures or
trigger-definition strings, not live writers.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCAN_DIRS = (REPO / "research/scientist", REPO / "research/tools")

# Files excluded from scanning (each has a doc/DDL-string reason, not a real
# writer). Add new entries only when a match is a docstring or trigger-body
# reference, never to hide a real write.
_EXCLUDED_FILES = frozenset(
    {
        # Trigger DDL + SQL mirror regex literals.
        "research/scientist/notebook/notebook_core.py",
        # Builds trigger SQL strings on the live db.
        "research/tools/install_phase5b_propagation_triggers.py",
        # One-shot migration that reads from the legacy table by design.
        "research/tools/backfill_graph_runs_from_program_results.py",
        # The Stage 3 dropper itself (operates on the table being dropped).
        "research/tools/drop_legacy_program_results.py",
        # Schema migration tool that references the table by name.
        "research/tools/migrate_graph_normalization.py",
        # Docstring describes the AFTER-INSERT propagation behavior. Body
        # writes to graphs + graph_runs only.
        "research/scientist/notebook/program_writes.py",
        # Comment explains why it reads from program_results (truthful
        # mirror); only SELECTs, no writes.
        "research/tools/diff_leaderboard_snapshot.py",
    }
)

# Snapshot of per-file legacy-write counts as of 2026-05-10 cycle 3.
# Decrement when a file's writes are retargeted; bump only with reviewer
# approval (and prefer to retarget instead).
_LEGACY_WRITE_COUNTS: dict[str, int] = {
    "research/scientist/api_routes/programs_routes/program_actions.py": 0,
    "research/scientist/language_control_gates.py": 0,
    "research/scientist/notebook/notebook_entries.py": 0,
    "research/scientist/notebook/notebook_programs.py": 0,
    "research/scientist/notebook/program_result_merge.py": 0,
    "research/scientist/runner/_helpers_benchmark.py": 0,
    "research/scientist/runner/_helpers_metrics.py": 0,
    "research/scientist/runner/dashboard_orchestrator.py": 0,
    "research/tools/apply_language_control_s10_comparison.py": 0,
    "research/tools/ar_gate_no_go_flag.py": 0,
    "research/tools/backfill.py": 0,
    "research/tools/backfill_ar_gate.py": 0,
    "research/tools/backfill_ar_validation.py": 0,
    "research/tools/backfill_bpe_evals.py": 0,
    "research/tools/backfill_champion_reference_tests.py": 0,
    "research/tools/backfill_spec_norm.py": 0,
    "research/tools/backfill_sticky_probe_columns.py": 0,
    "research/tools/backfill_trajectory_metrics.py": 0,
    "research/tools/backfill_trajectory_metrics_parallel.py": 0,
    "research/tools/backfill_understanding_metrics.py": 0,
    "research/tools/dedup_within_experiment.py": 0,
    "research/tools/import_ar_validation_fingerprint_sweep.py": 0,
    "research/tools/language_control_backfill.py": 0,
    "research/tools/nano_bind_backfill.py": 0,
    "research/tools/promote_backlog_batch.py": 0,
    "research/tools/rescore_champion_tiny_model.py": 0,
}

_WRITE_RE = re.compile(
    r"\b(UPDATE|INSERT\s+INTO|DELETE\s+FROM|REPLACE\s+INTO|"
    r"INSERT\s+OR\s+(?:IGNORE|REPLACE)\s+INTO)\s+program_results\b",
    re.IGNORECASE,
)
# Skip matches against the compat view / merge-backup tables (read-only mirrors).
_DENY_TAIL_RE = re.compile(r"program_results_(compat|cross_exp_merge_backup)")


def _count_writes_in_file(path: Path) -> int:
    count = 0
    text = path.read_text(encoding="utf-8", errors="replace")
    for match in _WRITE_RE.finditer(text):
        # The regex anchors on the table name; reject the view/backup variants.
        line_start = text.rfind("\n", 0, match.start()) + 1
        line_end = text.find("\n", match.end())
        if line_end == -1:
            line_end = len(text)
        line = text[line_start:line_end]
        if _DENY_TAIL_RE.search(line):
            continue
        count += 1
    return count


def _scan_all() -> dict[str, int]:
    counts: dict[str, int] = {}
    for root in SCAN_DIRS:
        for py_path in root.rglob("*.py"):
            rel = str(py_path.relative_to(REPO))
            if rel in _EXCLUDED_FILES:
                continue
            n = _count_writes_in_file(py_path)
            if n:
                counts[rel] = n
    return counts


def test_no_unknown_program_results_writers():
    """Every file with a legacy-table write must appear in the whitelist."""
    actual = _scan_all()
    unknown = sorted(set(actual) - set(_LEGACY_WRITE_COUNTS))
    assert not unknown, (
        f"New program_results writers detected: {unknown}. "
        "Retarget to graph_runs (preferred) or add to "
        "_LEGACY_WRITE_COUNTS in this test with reviewer approval."
    )


def test_program_results_write_counts_match_snapshot():
    """Per-file write counts must match the 2026-05-10 cycle-3 snapshot.

    A bump means a new legacy write was added — retarget it.
    A drop means a retarget happened — decrement the whitelist entry
    (or remove it entirely when the count hits zero).
    """
    actual = _scan_all()
    diffs = []
    for rel, expected in _LEGACY_WRITE_COUNTS.items():
        observed = actual.get(rel, 0)
        if observed != expected:
            diffs.append(f"  {rel}: expected {expected}, found {observed}")
    if diffs:
        msg = (
            "program_results write counts diverged from the 2026-05-10 cycle-3 "
            "snapshot:\n"
            + "\n".join(diffs)
            + "\n\nDecrement the count in _LEGACY_WRITE_COUNTS if a retarget "
            "happened, or revert the new write if you added one."
        )
        raise AssertionError(msg)


def test_stage3_readiness_signal():
    """Skip-converting sentinel for Stage 3 readiness.

    Passes silently while legacy writers remain (work-in-progress state).
    Skips with a clear message when the whitelist sums to zero — the
    skip text is visible in test output so a maintainer notices that
    ``research/tools/drop_legacy_program_results.py`` is now safe to
    apply. Skip (not fail) so CI stays green when the work lands; the
    drop itself is a separate, deliberate action.
    """
    import pytest

    total = sum(_LEGACY_WRITE_COUNTS.values())
    if total == 0:
        pytest.skip(
            "Stage 3 readiness reached: zero remaining program_results writers. "
            "Run research/tools/drop_legacy_program_results.py to drop the "
            "legacy table, then clean up this sentinel."
        )
    # Implicit pass while legacy writers remain.

from __future__ import annotations

from pathlib import Path
import re

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[1]
_SCAN_ROOTS = [
    _ROOT / "scientist" / "runner",
    _ROOT / "scientist" / "api_routes",
]

_FORBIDDEN_PATTERNS = {
    "nb.start_experiment": re.compile(r"\bnb\.start_experiment\("),
    "nb.complete_experiment": re.compile(r"\bnb\.complete_experiment\("),
    "nb.fail_experiment": re.compile(r"\bnb\.fail_experiment\("),
    "nb.log_learning_event": re.compile(r"\bnb\.log_learning_event\("),
    "retry_nb.log_learning_event": re.compile(r"\bretry_nb\.log_learning_event\("),
    "getattr.start_experiment": re.compile(
        r"""getattr\(\s*nb\s*,\s*["']start_experiment["']\s*\)"""
    ),
    "getattr.complete_experiment": re.compile(
        r"""getattr\(\s*nb\s*,\s*["']complete_experiment["']\s*\)"""
    ),
    "getattr.fail_experiment": re.compile(
        r"""getattr\(\s*nb\s*,\s*["']fail_experiment["']\s*\)"""
    ),
    "getattr.log_learning_event": re.compile(
        r"""getattr\(\s*nb\s*,\s*["']log_learning_event["']\s*\)"""
    ),
    "sql.status.running": re.compile(r"UPDATE experiments SET status = 'running'"),
    "sql.status.interrupted": re.compile(
        r"UPDATE experiments SET status = 'interrupted'"
    ),
    "nb._maybe_commit": re.compile(r"\bnb\._maybe_commit\("),
}

# Snapshot of reviewed exceptions during migration. New occurrences require
# explicit review and matrix/audit updates. The guard intentionally ignores line
# numbers so unrelated edits do not force a baseline churn.
#
# Categories:
# - compatibility sinks: runner-owned helper methods that deliberately keep
#   notebook persistence for backward compatibility after publishing events
# - telemetry bridges: best-effort logging paths not yet fully migrated
# - admin/runtime direct writes: explicitly reviewed non-lifecycle writes
_EXPECTED_BASELINE = {
    # API telemetry bridge
    "scientist/api_routes/analytics_bp.py:nb.log_learning_event": 1,
    # API admin task cancellation write
    "scientist/api_routes/programs_routes/validation_rerun.py:nb._maybe_commit": 1,
    # Runtime benchmark/metric direct-write bridge
    "scientist/runner/_helpers_benchmark.py:nb._maybe_commit": 2,
    # Centralized lifecycle compatibility sinks
    "scientist/runner/_lifecycle.py:getattr.complete_experiment": 1,
    "scientist/runner/_lifecycle.py:getattr.fail_experiment": 1,
    "scientist/runner/_lifecycle.py:getattr.log_learning_event": 1,
    # Admin maintenance writes
    "scientist/runner/control_actions.py:nb._maybe_commit": 2,
    # Automation telemetry bridges
    "scientist/runner/results_auto_escalate_phase7.py:nb.log_learning_event": 1,
}


def _scan_contract_violations() -> dict[str, int]:
    matches: dict[str, int] = {}
    for root in _SCAN_ROOTS:
        for path in sorted(root.rglob("*.py")):
            rel_path = path.relative_to(_ROOT).as_posix()
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                for name, pattern in _FORBIDDEN_PATTERNS.items():
                    if pattern.search(line):
                        key = f"{rel_path}:{name}"
                        matches[key] = matches.get(key, 0) + 1
    return matches


def test_runtime_event_contract_guard_baseline():
    found = _scan_contract_violations()
    assert found == _EXPECTED_BASELINE, (
        "Runtime event contract drift detected.\n"
        "New direct notebook/runtime lifecycle writes in runner/api_routes "
        "must be reviewed and classified in the migration matrix.\n"
        f"Found: {found}\n"
        f"Expected: {_EXPECTED_BASELINE}"
    )

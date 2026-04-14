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
# explicit review and matrix/audit updates.
#
# Categories:
# - compatibility sinks: runner-owned helper methods that deliberately keep
#   notebook persistence for backward compatibility after publishing events
# - telemetry bridges: best-effort logging paths not yet fully migrated
# - admin/runtime direct writes: explicitly reviewed non-lifecycle writes
_EXPECTED_BASELINE = {
    # API telemetry bridge
    "scientist/api_routes/analytics_bp.py:145:nb.log_learning_event",
    # Runtime benchmark/metric direct-write bridge
    "scientist/runner/_helpers.py:1536:nb._maybe_commit",
    # Continuous investigation compatibility sinks
    "scientist/runner/continuous_investigation.py:71:getattr.complete_experiment",
    "scientist/runner/continuous_investigation.py:87:getattr.fail_experiment",
    # Continuous loop / session telemetry bridge
    "scientist/runner/continuous_loop.py:35:getattr.log_learning_event",
    # Continuous mode compatibility sinks + telemetry bridge
    "scientist/runner/continuous_modes.py:34:getattr.log_learning_event",
    "scientist/runner/continuous_modes.py:61:getattr.complete_experiment",
    # Continuous validation compatibility sinks
    "scientist/runner/continuous_validation.py:66:getattr.complete_experiment",
    "scientist/runner/continuous_validation.py:82:getattr.fail_experiment",
    # Admin maintenance writes and telemetry bridges
    "scientist/runner/control_actions.py:56:getattr.log_learning_event",
    "scientist/runner/control_actions.py:500:nb._maybe_commit",
    "scientist/runner/control_actions.py:526:nb._maybe_commit",
    # Cycle compatibility sinks + telemetry bridge
    "scientist/runner/cycle.py:40:getattr.log_learning_event",
    "scientist/runner/cycle.py:49:getattr.fail_experiment",
    # Dashboard/runtime telemetry bridge
    "scientist/runner/dashboard.py:54:getattr.log_learning_event",
    # Investigation compatibility sinks
    "scientist/runner/execution_investigation.py:61:getattr.complete_experiment",
    "scientist/runner/execution_investigation.py:78:getattr.fail_experiment",
    # Screening compatibility sinks + telemetry bridges
    "scientist/runner/execution_screening.py:77:getattr.log_learning_event",
    "scientist/runner/execution_screening.py:848:getattr.log_learning_event",
    "scientist/runner/execution_screening.py:875:getattr.complete_experiment",
    "scientist/runner/execution_screening.py:891:getattr.fail_experiment",
    # Search compatibility sinks
    "scientist/runner/execution_search.py:40:getattr.complete_experiment",
    "scientist/runner/execution_search.py:56:getattr.fail_experiment",
    # Validation compatibility sinks
    "scientist/runner/execution_validation.py:62:getattr.complete_experiment",
    "scientist/runner/execution_validation.py:79:getattr.fail_experiment",
    # Automation telemetry bridges
    "scientist/runner/results_auto_escalate_phase7.py:69:getattr.log_learning_event",
    "scientist/runner/results_automation.py:25:getattr.log_learning_event",
    # Synthesis compatibility sink + telemetry bridge
    "scientist/runner/synthesis.py:58:getattr.log_learning_event",
    "scientist/runner/synthesis.py:85:getattr.complete_experiment",
}


def _scan_contract_violations() -> set[str]:
    matches: set[str] = set()
    for root in _SCAN_ROOTS:
        for path in sorted(root.rglob("*.py")):
            rel_path = path.relative_to(_ROOT).as_posix()
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                for name, pattern in _FORBIDDEN_PATTERNS.items():
                    if pattern.search(line):
                        matches.add(f"{rel_path}:{lineno}:{name}")
    return matches


def test_runtime_event_contract_guard_baseline():
    found = _scan_contract_violations()
    assert found == _EXPECTED_BASELINE, (
        "Runtime event contract drift detected.\n"
        "New direct notebook/runtime lifecycle writes in runner/api_routes "
        "must be reviewed and classified in the migration matrix.\n"
        f"Unexpected/new entries: {sorted(found - _EXPECTED_BASELINE)}\n"
        f"Missing/removed entries: {sorted(_EXPECTED_BASELINE - found)}"
    )

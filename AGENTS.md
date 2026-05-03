# Codex Guardrails

These instructions apply to Codex sessions in this repository.

## Before Editing

- Run a code-review-graph query before the first edit when the tool is available. At minimum use `code-review-graph status` plus `rg` searches for the feature, symbol, route, table, or metric you are about to touch.
- Search for existing implementation before adding a new function, route, tool, metric adapter, or test helper. Prefer extending the existing path over duplicating behavior.
- Treat the current git worktree as user-owned. Do not revert, delete, reset, or overwrite unrelated changes.

## Data Safety

- Do not delete or truncate research data, databases, runtime event logs, perf artifacts, or backups.
- Do not run `git reset --hard`, broad `git clean`, or recursive deletes under `research/` unless the user explicitly requests the exact destructive action in the current chat.
- Before any intentional database mutation, verify backup freshness with `python -m research.tools.check_backup_freshness` or make a new backup first.

## Experiment And Test Integrity

- A `program_results` row with `stage1_passed=True` must include the core post-S1 metrics. Use `program_result_kwargs_from_s1`; do not hand-assemble loss-only S1 rows.
- Do not create tests that pass by filling large structures with empty data, placeholder metrics, or unrelated dummy payloads. Minimal fixtures are fine only when each field is intentional and asserted.
- When a test needs a representative result row, include meaningful values for the fields the production code uses.

## Review Bar

- New functionality should have focused tests that would fail for the regression being fixed.
- If adding a guardrail, test both the blocked path and the allowed escape hatch.
- Keep changes scoped. Do not refactor god files or split modules incidentally unless that is the task.

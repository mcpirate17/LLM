# Codex Guardrails

These instructions apply to Codex sessions in this repository.

## Before Editing

- Run a code-review-graph query before the first edit when the tool is available. At minimum use `code-review-graph status` plus `rg` searches for the feature, symbol, route, table, or metric you are about to touch.
- Search for existing implementation before adding a new function, route, tool, metric adapter, or test helper. Prefer extending the existing path over duplicating behavior.
- Treat the current git worktree as user-owned. Do not revert, delete, reset, or overwrite unrelated changes.

## Operating Workflow

- Start with the narrowest useful inspection: `code-review-graph status`, targeted `rg`, then focused file reads. Avoid broad scans of generated data unless the task specifically requires them.
- Prefer repo entrypoints over ad hoc commands: use `make test-changed`, `make test-research`, `make test-aria_core`, `make test-designer`, `make guardrails-dry`, `make perf-summary`, and other Makefile targets when they fit the change.
- Keep long-running research, training, profiling, and backfill commands explicit. State the expected scope before launching them, and avoid starting expensive runs as a side effect of small code or docs edits.
- When touching performance-sensitive paths, measure or explain why measurement is not practical. Preserve existing benchmark outputs and perf artifacts.
- Before finalizing substantial work, check `git status --short` and report only the files and verification that matter to the task.

## Codex Journal

- For substantial repo work, append an Obsidian-compatible entry before the final response using `make codex-journal JOURNAL_NOTE="..."`. Include each meaningful verification command with `JOURNAL_TEST="..."` when there is one.
- In a dirty worktree, scope the journal to touched files with `JOURNAL_PATHS="AGENTS.md conductor/codex_journal.py"` or another precise path list. Keep the default `JOURNAL_MAX_STATUS=80` cap unless the larger status is intentionally useful.
- Journal entries are private local notes under `tasks/codex_journal/`, surfaced in Obsidian through `/home/tim/Documents/CodexVault/LLM Codex Journal`. Do not commit or summarize secrets, database contents, runtime event payloads, or protected research artifacts into the journal.
- If no code or repo state changed, skip the journal unless the user explicitly asks for a note.

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

# Directory Usage Audit - 2026-04-26

This is the focused answer to: "what are all these tiny files/dirs, and how many are even used?"

## Top-Level Verdict

| Path | Status | Why |
|---|---|---|
| `research/` | Active core project | Heavily referenced by `Makefile`, tests, dashboard, runner, tools. |
| `aria_core/` | Active native layer | Built by root `Makefile`; referenced by `research` and `aria_designer/runtime`. |
| `aria_designer/` | Active integrated app | Root README/Makefile describe it as the visual workflow editor; tests and source are tracked. |
| `.github/` | Active | CI/workflows. |
| `.claude/` | Active local agent config | Small; keep. |
| `conductor/` | Mixed/stale | `Makefile` and pre-commit still reference `conductor/guardrail_audit.py`, but most conductor files are currently deleted in git status. Decide: restore/keep only guardrail tooling, or remove references. |
| `HYDRA/` | Legacy/standalone plus optional data source | Ignored as a separate project. `research` can optionally load `../HYDRA/data`; checkpoint pile is not part of active source. |
| `LA3/` | Standalone side project | No meaningful repo cross-reference besides cleanup audit. Small; can archive or keep. |
| `personaplex/` | Standalone side project | Has its own `.git`; no meaningful repo cross-reference. Treat separately. |
| `.code-review-graph/` | Regenerable tool index | Local generated index. Safe to delete if startup/reindex cost is acceptable. |
| `.ruff_cache/`, `.vscode/` | Local/generated | Safe to delete/ignore. |

Count: 3 integrated active project directories (`research`, `aria_core`, `aria_designer`), 2 active config/tooling dirs (`.github`, `.claude`), 1 broken/mixed tooling dir (`conductor`), 3 standalone/legacy side projects (`HYDRA`, `LA3`, `personaplex`), and 2 local generated dirs (`.code-review-graph`, `.ruff_cache`).

## What The Root Files Are

| File | Status | Notes |
|---|---|---|
| `README.md`, `CLAUDE.md`, `Makefile`, `pyproject.toml`, `uv.lock`, `ruff.toml`, `.pre-commit-config.yaml`, `.mcp.json`, `.gitignore` | Keep | Project config/docs. |
| `.current_work.md` | Condense | 1,209-line work journal. Useful, but should be summarized and archived. |
| `GLOBAL_DEV_PROMPT.md`, `TEMPLATE_QUALITY_PLAN.md` | Probably keep | Policy/planning docs. |
| `BACKFILL_DO_NOT_DELETE.txt`, `COMPONENT_EXPLORE_DO_NOT_DELETE.txt`, `DO_NOT_DELETE_TRAINER.txt`, `keys_DO_NOT_DELETE.txt` | Keep until reviewed | Names explicitly say do not delete; likely scratch-but-important instructions. |
| `TOMORROW.md` | Refresh/delete after extraction | Partly stale; it mentioned MagicMock junk that has now been removed. |
| `happy_times.py`, `hive.sh`, `vulture_whitelist.py` | Keep unless unused check says otherwise | Tooling scripts. |
| `CLEANUP_AUDIT_20260426.md`, this file | Keep short-term | Cleanup reports. |

## `research/` Directory Verdict

### Active source/code dirs

These have tracked files and are part of active code/tests:

| Path | Tracked files | Status |
|---|---:|---|
| `research/dashboard/` | 204 | Active UI. |
| `research/eval/` | 74 | Active eval suite. |
| `research/healer/` | 3 | Active/available repair tooling. |
| `research/mathspaces/` | 10 | Active synthesis/math code. |
| `research/orchestrator/` | 1 | Small active package. |
| `research/runtime/` | 448 | Active native/runtime/learning area, but contains generated build outputs. |
| `research/runtime_events/` | 1 | Active runtime event stream file is tracked/modified. |
| `research/schemas/` | 3 | Active schema files. |
| `research/scientist/` | 238 | Active main app/backend/runner. |
| `research/search/` | 12 | Active search code. |
| `research/synthesis/` | 100 | Active synthesis code. |
| `research/tests/` | 269 | Active tests. |
| `research/tools/` | 64 | Active tools; some one-offs should be reviewed against `research/tools/REGISTRY.md`. |
| `research/training/` | 20 | Active training utilities. |

### State/generated dirs

| Path | Status | Notes |
|---|---|---|
| `research/corpus/` | Keep if offline evals needed | Ignored dataset cache; `wikitext103_train.npy` is large but useful. |
| `research/artifacts/` | Generated/ignored | Small currently; historical large artifacts are in git history, not worktree. |
| `research/db_backups/` | Retention decision | DB backup storage. |
| `research/logs/` | Generated | Safe to summarize/delete old logs. |
| `research/perf_artifacts/` | Clean now | Condensed to four summary files. |
| `research/profiling/` | Generated | Profiling DB; keep only if current. |
| `.pytest_cache/`, `__pycache__/` | Generated | Safe delete. |

### `research/` root clutter visible in the screenshot

The big mess is not many directories; it is state files sitting directly in `research/`.

| Pattern/File | What it is | Recommendation |
|---|---|---|
| `lab_notebook.db` | Current active SQLite notebook DB | Keep. |
| `lab_notebook.db.bak_*`, `.snap_*`, `.corrupt_*`, `.malformed_*`, `.pre_eval_sync_*`, `.recovered_dump` | Recovery/snapshot copies | Move to `research/db_backups/` or prune by retention policy. Do not keep scattered in root. |
| `lab_notebook.db-shm`, `lab_notebook.db-wal`, `*.writer-lock` | SQLite/runtime sidecars | Usually generated. Only live sidecars matter while DB is open. |
| `aria_dashboard.log*`, `logs/` | Runtime logs | Summarize/delete old logs. |
| `aria_lab.db`, `baseline_cache.db`, `scaling_reference_cache.db*` | Small cache/side DBs | Keep only if code still reads them; otherwise move under `research/runtime/` or `research/db_backups/`. |
| `.continuous_paused` | Sentinel | Active operational state. Keep unless intentionally resuming continuous. |
| `.current_work.md` | 2,970-line work journal | Condense into a short current-state doc plus archive. |
| `TEMPLATE_AUDIT.md`, `TEMPLATE_OPTIMIZATION_PLAYBOOK.md`, `BASELINE_SCORING_DO_NOT_DELETE.md` | Planning/reference docs | Keep until manually reviewed. |

## `aria_designer/` Directory Verdict

Active/tracked:

| Path | Tracked files | Status |
|---|---:|---|
| `api/` | 43 | Active FastAPI backend. |
| `components/` | 426 | Active component registry/manifests. |
| `e2e/` | 13 | Active E2E tests; generated screenshots/results can be cleaned. |
| `examples/` | 5 | Keep. |
| `runtime/` | 19 | Active bridge/runtime. |
| `schemas/` | 4 | Keep. |
| `tests/` | 30 | Keep. |
| `tools/` | 15 | Keep. |
| `ui/` | 79 | Active UI; `node_modules/` and `dist/` are generated. |
| `workflows/` | 30 | Mostly generated/history; summarize if no longer current. |

Generated/local:

- `.run/` - logs/PIDs; safe to clear when services are stopped.
- `__pycache__/` - safe delete.
- `ui/node_modules/` - safe delete/reinstall.
- `ui/dist/` - safe delete/rebuild.
- `runtime/tests/build/` - safe delete/rebuild.

## Immediate Cleanup Plan For The Visual Clutter

1. Move/prune all `research/lab_notebook.db.*` snapshots/backups into `research/db_backups/` or delete by retention policy.
2. Condense `.current_work.md` and `research/.current_work.md`.
3. Clean generated dirs: caches, `node_modules`, build dirs, `.run`, logs.
4. Decide `conductor`: restore the guardrail tool files or remove Makefile/pre-commit references.
5. Decide whether `HYDRA`, `LA3`, and `personaplex` should stay under `LLM/` or move to an `external/` or sibling-project area.

# Tomorrow DB Split Plan

Goal: reduce `research/lab_notebook.db` write pressure and file growth by splitting live control state, event logs, and bulky artifacts into separate storage paths while keeping restore and verification simple.

## Current State

- `research/lab_notebook.db` has been restored and passes `quick_check`.
- Bulky payloads are already being externalized into compressed artifact files under `research/artifacts/notebook`.
- The DB now contains artifact pointer JSON for selected large fields and training curves.
- A latest-only Google Drive backup bundle exists at `gdrive:LLM/db-backups/20260508T222852`.
- Local historical DB backup files have been pruned.

## Target Layout

- `research/runs.db`
  - Experiments
  - Program results
  - Leaderboard data
  - Follow-up tasks
  - Notebook artifact pointer metadata
- `research/events.db` or append-only NDJSON segments
  - Runtime feed events
  - API/dashboard activity events
  - Scheduler progress events
  - High-volume operational logs
- `research/artifacts/notebook/`
  - Compressed bulky JSON payloads
  - Compressed training curves
  - Compressed graph/perf/report payloads when safe
- `research/lab_notebook.db`
  - Temporary compatibility alias during migration only
  - Retire after all code paths resolve the new locations through defaults/config

## Non-Negotiables

- Do not delete the active DB or artifact directory during the split.
- Before every DB mutation, run `python -m research.tools.check_backup_freshness` or make a fresh latest-only Drive backup bundle.
- Use `PRAGMA quick_check` before and after each migration phase.
- Keep artifact files compressed at rest with integrity metadata.
- Decompress artifacts only at read boundaries that already expect native JSON/list/dict payloads.
- Preserve a restore drill that verifies both DB integrity and artifact-backed reads.

## Phase 1: Inventory And Read Paths

1. List all tables in the current notebook DB with row counts and approximate payload size.
2. Classify each table as `runs`, `events`, `artifact_metadata`, or `legacy`.
3. Search every direct `research/lab_notebook.db` reference and replace hardcoded paths with defaults/config access where needed.
4. Identify raw SQL dashboard/API paths that expect inline JSON and confirm whether they already resolve artifact pointers.
5. Add a short migration manifest file describing table ownership and target DB.

Exit checks:

- Table ownership manifest exists.
- No new direct hardcoded notebook DB paths are introduced.
- Existing artifact-backed program detail and training curve reads still pass.

## Phase 2: Events Split

1. Choose one event storage format:
   - Prefer append-only NDJSON for write-heavy feed/runtime logs if SQL querying is not required.
   - Use `events.db` only for events that need indexed SQL filters.
2. Add a small event writer API with one writer process or queue.
3. Move feed/runtime writes off `lab_notebook.db`.
4. Keep dashboard reads compatible by adding an event reader that can read the new store and, temporarily, any legacy rows.
5. Add a bounded retention/rotation policy for event segments without deleting protected research data by default.

Exit checks:

- Event writes no longer touch the main notebook DB.
- Dashboard live feed works from the new event store.
- Event writer tests cover concurrent write pressure.

## Phase 3: Runs DB Split

1. Create `research/runs.db` from the current notebook schema subset.
2. Copy core experiment tables:
   - `experiments`
   - `program_results`
   - `leaderboard`
   - `followup_tasks`
   - `notebook_artifacts`
3. Keep artifact pointer values exactly as-is so compressed artifact files remain reusable.
4. Add a compatibility layer so `LabNotebook` resolves the runs DB through config/defaults.
5. Keep the old `lab_notebook.db` read-only during a transition window.

Exit checks:

- `runs.db` passes `quick_check`.
- Row counts match for migrated tables.
- Representative program detail, leaderboard, experiment detail, and training curve reads work.
- Writes to experiments/program results land only in `runs.db`.

## Phase 4: Artifact Expansion

1. Review remaining large JSON columns by size and read frequency.
2. Externalize safe bulky fields into `research/artifacts/notebook` using compressed `.json.zst` files.
3. Store pointer JSON with:
   - relative artifact path
   - codec
   - original byte size
   - compressed byte size
   - SHA256 of uncompressed payload
   - created timestamp
4. Decompress lazily when callers request the field through notebook APIs.
5. Do not externalize fields used by raw SQL sorting/filtering until those reads are migrated to API-level resolution or summary columns exist.

Exit checks:

- DB size decreases after `VACUUM`.
- Artifact integrity verification passes.
- Raw SQL dashboard views do not regress.

## Phase 5: Backup, Restore, And Prune Policy

1. Keep one latest-only Drive backup bundle that contains:
   - current active DB files
   - `research/artifacts/notebook`
   - manifest and file hashes
2. Do not keep local DB snapshot piles.
3. Keep local live artifacts because active DB rows point to them.
4. Add a restore drill command that:
   - downloads the latest Drive bundle to temp storage
   - unpacks it
   - runs `quick_check` on every DB
   - verifies sample artifact decompression
   - verifies representative notebook reads
5. Only prune local historical backups after upload verification succeeds.

Exit checks:

- Restore drill passes from Drive.
- Local historical DB backup search is empty.
- Active DBs and artifact directory remain present.

## Phase 6: Concurrency Hardening

1. Ensure one writer process or write queue owns runs DB mutations.
2. Keep dashboard/API read-only except through explicit writer APIs.
3. Use WAL mode, busy timeouts, short transactions, and startup `quick_check`.
4. Add post-batch `quick_check` for major write batches.
5. Add tests that prove direct dashboard writes are blocked or routed through the writer.

Exit checks:

- Concurrent read/write smoke test passes.
- Writer lock tests pass.
- No API route opens the runs DB in writable mode unless it is part of the writer path.

## Tomorrow Command Sequence

```bash
python -m research.tools.db_health --db research/lab_notebook.db
python -m pytest -q research/tests/test_notebook_artifacts.py research/tests/test_restore_lab_notebook.py research/tests/test_aria_db_writer_lock.py research/tests/test_db_health.py
sqlite3 research/lab_notebook.db '.tables'
rg -n "lab_notebook\\.db|LabNotebook\\(|runtime_events|training_curves|program_results|notebook_artifacts" research
```

Then implement Phase 1 inventory and write the migration manifest before moving any table data.

## Rollback

- If a split phase fails before cutover, leave `lab_notebook.db` as the active DB and discard only the newly generated split DB candidate.
- If a split phase fails after cutover, restore the latest Drive bundle into a temporary directory first, verify it, then replace active DB files.
- Never delete `research/artifacts/notebook` during rollback unless a verified restore bundle has been unpacked and tested.

# TODO: Migrate Runtime Persistence to a Real Event Bus

## Status

Open. Required architectural work.

Companion implementation plan:

- [event_bus_migration_plan_2026-04-13.md](/home/tim/Projects/LLM/research/docs/event_bus_migration_plan_2026-04-13.md)

## Why This Exists

The current runtime uses SQLite (`research/lab_notebook.db`) as too many things at once:

- source of truth
- live coordination layer
- analytics sink
- status transport
- cleanup/reporting backing store

That coupling is now a reliability problem. A transient SQLite `disk I/O error` is able to:

- abort experiment startup
- kill or poison continuous mode
- leave phantom in-memory/UI state
- lose final experiment status writes
- break noncritical analytics/logging paths during otherwise valid runs

The runtime needs a real pub/sub contract so experiment execution does not depend on synchronous notebook writes.

## Background And History

Observed on April 12, 2026 while debugging `research/lab_notebook.db` failures.

Key findings from live debugging:

- Basic direct SQLite insert/commit/update/delete with dummy data works.
- `LabNotebook.start_experiment()` and `complete_experiment()` also work in isolated probes.
- Repeated probe script against the live DB passed:
  - `python research/scripts/test_notebook_writes.py --db-path research/lab_notebook.db --iterations 10`
- The failure is not simply "SQLite cannot insert rows".

What actually failed in live runs:

- Fresh notebook connection setup intermittently hit `disk I/O error` on pragmas such as:
  - `PRAGMA journal_mode=WAL`
  - `PRAGMA synchronous=NORMAL`
- Experiment startup could fail in `start_experiment()` when a fresh verification connection immediately re-read the just-inserted row.
- That startup false-negative was patched so primary-connection visibility is accepted when direct verification fails.
- After that patch, startup succeeded, but the run still failed later on ordinary notebook writes inside the live experiment thread.
- Concrete example:
  - `execution_screening.py` -> `_prepare_grammar_config()` -> `nb.log_learning_event(...)`
  - `notebook_advanced_analytics.py` -> `self.conn.execute(...)`
  - `sqlite3.OperationalError: disk I/O error`
- Failure handling then also hit notebook write failures:
  - `fail_experiment()`
  - healer task creation
  - post-run logging/reporting

Operational symptoms already seen:

- experiments reported as started but missing from durable DB state
- completed runs left as `running`
- dashboard `/api/status` claiming a phantom active experiment after launch failure
- cleanup/reporting paths killing continuous mode
- live feed showing experiment IDs that never became durable records

Conclusion:

- The current architecture is too SQLite-coupled.
- Even after reducing connection churn, the runtime still fails because noncritical runtime paths write directly to the notebook in the hot path.

## Immediate Design Conclusion

The notebook must stop being the synchronous runtime coordination mechanism.

We need:

- a real in-process pub/sub event bus for runtime components
- a durable append-only event spool on disk
- one or more projectors/consumers that materialize events into SQLite

SQLite should become a downstream projection target, not the thing every runtime path writes to synchronously.

## Required Contract

Every runtime test, experiment thread, background worker, automation pass, and API path must adhere to the event bus contract:

- producers emit typed events
- consumers subscribe to typed events
- SQLite writes happen in projector/consumer layers, not directly from arbitrary runtime code
- noncritical event sinks must be best-effort
- critical lifecycle events must be durably spooled before acknowledging success

Direct `nb.conn.execute(...)` calls in runtime hot paths should be treated as contract violations unless explicitly part of the notebook projector or minimal lifecycle persistence layer.

## Proposal

### 1. Introduce Runtime Event Types

Define typed events for at least:

- `experiment_start_requested`
- `experiment_started`
- `experiment_start_failed`
- `program_generated`
- `program_screened`
- `screening_summary_updated`
- `learning_event_logged`
- `analytics_event_logged`
- `healer_task_opened`
- `experiment_failed`
- `experiment_completed`
- `report_generation_requested`
- `report_generation_failed`
- `cleanup_requested`
- `cleanup_skipped`

Each event should include:

- `event_id`
- `event_type`
- `timestamp`
- `experiment_id` when applicable
- `producer`
- `payload`
- `schema_version`

## 2. Add A Durable Event Spool

Implement an append-only local spool, likely NDJSON or a dedicated lightweight event log under something like:

- `research/runtime_events/`

Requirements:

- append-only writes
- per-experiment partitioning or easy filtering
- replay-safe `event_id`
- crash-tolerant fsync strategy for critical lifecycle events
- projector can resume from last applied offset/checkpoint

## 3. Add Event Bus Runtime Layer

Add an in-process event bus that:

- accepts events from producers
- fans out to subscribers
- updates in-memory UI/runtime state without needing notebook writes
- is usable from experiment thread, automation workers, healer, dashboard, and analytics modules

The dashboard should consume bus/projection state, not infer liveness from partially-written notebook rows whenever possible.

## 4. Split Critical vs Noncritical Persistence

Critical:

- `experiment_started`
- `experiment_failed`
- `experiment_completed`

Noncritical:

- learning events
- analytics updates
- healer metadata
- report progress
- auxiliary entries
- derived observability records

Rules:

- critical events must be durably spooled before the runtime acknowledges the state transition
- noncritical events may remain queued/spooled and be replayed later
- noncritical persistence failure must not abort a valid experiment

## 5. Introduce SQLite Projectors

Create one or more projectors that consume events and materialize them into:

- `experiments`
- `entries`
- analytics tables
- healer tables
- reporting tables

Projector rules:

- idempotent application by `event_id`
- batch writes
- checkpointed replay
- explicit degraded mode when SQLite is unhealthy
- never block core experiment execution on noncritical projection work

## 6. Move Runtime Code Off Direct Notebook Writes

Audit and migrate hot-path calls including, at minimum:

- `log_learning_event(...)`
- analytics logging
- healer task creation
- post-cycle cleanup bookkeeping
- report generation bookkeeping
- side-channel dashboard status updates

The current pattern:

```python
nb.conn.execute(...)
nb._maybe_commit()
```

should be replaced with:

```python
event_bus.publish(...)
```

and optional projector consumption.

## 7. Make Launch/Failure State Bus-Driven

The dashboard and API should not claim a running experiment purely from in-memory thread state plus ad hoc DB probes.

Instead:

- launch success follows durable `experiment_started`
- launch failure emits `experiment_start_failed`
- UI state derives from latest accepted lifecycle events
- startup/restart reconstructs state from event spool plus notebook projection

## Migration Plan

### Phase 1: Infrastructure

- add event schema
- add in-process pub/sub bus
- add durable append-only spool
- add minimal lifecycle projector

### Phase 2: Lifecycle First

- route experiment start/fail/complete through event bus
- stop using direct notebook verification reads as startup truth
- make `/api/status` and live feed derive from bus/projected lifecycle state

### Phase 3: Noncritical Telemetry

- migrate `learning_log`, analytics, healer metadata, and report progress to emitted events
- make these sinks best-effort

### Phase 4: Contract Enforcement

- audit runtime modules for direct notebook writes
- mark allowed projector-only notebook write paths
- fail tests on new direct hot-path notebook writes

### Phase 5: Replay And Recovery

- startup replay from event spool
- projector checkpointing
- stale/incomplete run reconstruction from lifecycle events

## Acceptance Criteria

The migration is not complete until all of the following are true:

- an experiment can continue if analytics/logging SQLite writes fail
- startup does not depend on immediate fresh-connection readback from SQLite
- launch failure cannot leave phantom `is_running` state
- final experiment lifecycle state can be recovered from event spool even if notebook projection fails
- dashboard liveness derives from lifecycle events, not fragile mixed in-memory/DB heuristics
- noncritical consumers can be disabled or degraded without aborting the run
- tests enforce the event bus contract for runtime code

## Concrete Follow-Up Work

- Add `research/runtime_events/` spool design and implementation.
- Add `RuntimeEvent` schema and publisher API.
- Add notebook lifecycle projector.
- Convert `log_learning_event()` to publish instead of direct notebook write.
- Convert healer task bookkeeping to publish instead of direct notebook write.
- Convert report/cleanup bookkeeping to publish instead of direct notebook write.
- Add a contract audit for direct `nb.conn.execute(...)` usage in runtime hot paths.
- Add tests that simulate projector SQLite failure while experiment execution continues.

## Related Files Investigated During This Failure

- [notebook_core.py](/home/tim/Projects/LLM/research/scientist/notebook/notebook_core.py:1)
- [notebook_experiments.py](/home/tim/Projects/LLM/research/scientist/notebook/notebook_experiments.py:1)
- [notebook_advanced_analytics.py](/home/tim/Projects/LLM/research/scientist/notebook/notebook_advanced_analytics.py:1)
- [execution_screening.py](/home/tim/Projects/LLM/research/scientist/runner/execution_screening.py:1)
- [continuous_loop.py](/home/tim/Projects/LLM/research/scientist/runner/continuous_loop.py:1)
- [experiments_bp.py](/home/tim/Projects/LLM/research/scientist/api_routes/experiments_bp.py:1)
- [_experiment_launch.py](/home/tim/Projects/LLM/research/scientist/api_routes/_experiment_launch.py:1)
- [_helpers.py](/home/tim/Projects/LLM/research/scientist/api_routes/_helpers.py:1)
- [test_notebook_writes.py](/home/tim/Projects/LLM/research/scripts/test_notebook_writes.py:1)

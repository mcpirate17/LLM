# Event Bus Migration Plan

## Status

Draft implementation plan for the TODO in
[event_bus_migration_todo_2026-04-12.md](/home/tim/Projects/LLM/research/docs/event_bus_migration_todo_2026-04-12.md).

## Goal

Replace SQLite as the runtime coordination mechanism with:

- an in-process event bus for runtime state fan-out
- a durable append-only event spool for lifecycle truth
- SQLite projectors that materialize downstream views

The first cut must solve one problem well:

- experiment lifecycle truth must survive SQLite write failures

Everything else is secondary.

## Non-Goals For V1

V1 should not attempt to migrate every notebook write in one pass.

Explicit non-goals:

- replacing all read paths
- migrating all analytics/reporting tables
- redesigning the dashboard payload format
- introducing cross-process networking or an external broker
- making the event system generic enough for every future subsystem

## Current Code Reality

The lifecycle seams are real, but they are still SQLite-centric:

- [notebook_experiments.py](/home/tim/Projects/LLM/research/scientist/notebook/notebook_experiments.py:329) owns `start_experiment()`, `complete_experiment()`, and `fail_experiment()`
- [notebook_advanced_analytics.py](/home/tim/Projects/LLM/research/scientist/notebook/notebook_advanced_analytics.py:614) still writes `learning_log` synchronously
- [execution_screening.py](/home/tim/Projects/LLM/research/scientist/runner/execution_screening.py:820) and other runners call `nb.log_learning_event(...)` in hot paths
- [_helpers.py](/home/tim/Projects/LLM/research/scientist/api_routes/_helpers.py:187) derives status from a mixed in-memory plus notebook heuristic

That means a broad "event bus migration" is too large for one safe step. The rollout has to isolate lifecycle first, then move telemetry later.

## V1 Design Decision

Use a single local append-only spool plus an in-process bus.

Do not start with per-experiment files.

Reason:

- simpler append and fsync path
- simpler projector checkpointing
- simpler startup replay
- easier operator inspection with one chronological stream

If the spool grows too large, rotate by segment later. Do not complicate the first cut.

## Runtime Contract

Every producer emits a `RuntimeEvent`.

Every event has:

- `event_id`: globally unique, time-sortable if practical
- `event_type`
- `schema_version`
- `created_at`
- `producer`
- `run_id`: nullable, usually `experiment_id`
- `sequence`: producer-local monotonic sequence for ordering within a run
- `durability`: `critical` or `best_effort`
- `payload`

Recommended shape:

```json
{
  "event_id": "01J...",
  "event_type": "experiment_started",
  "schema_version": 1,
  "created_at": 1776048000.123,
  "producer": "runner.control_start",
  "run_id": "abc123def456",
  "sequence": 2,
  "durability": "critical",
  "payload": {
    "experiment_type": "screening",
    "config_json": "...",
    "hypothesis": "..."
  }
}
```

## Lifecycle Events In Scope For V1

Only five lifecycle events are required in the first implementation:

- `experiment_start_requested`
- `experiment_started`
- `experiment_start_failed`
- `experiment_completed`
- `experiment_failed`

Everything else stays on the old path until lifecycle truth is stable.

## Lifecycle State Machine

Allowed transitions:

- `none -> experiment_start_requested`
- `experiment_start_requested -> experiment_started`
- `experiment_start_requested -> experiment_start_failed`
- `experiment_started -> experiment_completed`
- `experiment_started -> experiment_failed`

Rules:

- `experiment_started` is the first event that makes a run live
- `experiment_start_failed` closes a requested launch that never became live
- `experiment_completed` and `experiment_failed` are terminal
- duplicate terminal events are ignored if payload-equivalent
- conflicting terminal events are recorded as projector errors and leave the first terminal state intact

This state machine must live in code, not only in docs.

## Publish Semantics

Provide one narrow publisher API:

```python
publish(event: RuntimeEvent) -> PublishResult
```

Behavior by durability:

- `critical`: append to spool, flush, fsync, then acknowledge success
- `best_effort`: append to memory queue or buffered spool writer and return after local acceptance

For V1, keep it simpler:

- all lifecycle events use `critical`
- non-lifecycle events are not required yet

If critical spool append fails:

- `experiment_started` must not be acknowledged
- `experiment_completed` and `experiment_failed` should still update in-memory error surfaces, but the caller must know durable lifecycle persistence failed

## Spool Format

Directory:

- `research/runtime_events/`

Files:

- `segment-000001.ndjson`
- `projector_checkpoint.json`

Append format:

- one JSON object per line
- no in-place mutation
- each line is a full event

Writer rules:

- open file in append mode
- serialize event to one line
- write newline
- flush
- `os.fsync()` for critical events

Rotation rules for V1:

- rotate only on process startup if active segment exceeds a size threshold
- no mid-run rotation logic in the first cut unless trivial

## Projector Design

V1 projector scope:

- project lifecycle events into `experiments`

Do not project analytics, healer, or reports yet.

Projector responsibilities:

- read spool from last checkpoint
- apply only unseen events
- map lifecycle events to `experiments` table writes
- store last applied offset and `event_id`
- enter degraded mode on SQLite failure without stopping the runner

Idempotency approach:

- add a small projector metadata table, for example `applied_runtime_events`
- primary key on `event_id`
- projector inserts marker before or within the same transaction as the projection write

Suggested metadata tables:

- `applied_runtime_events(event_id text primary key, event_type text, applied_at real)`
- `runtime_projector_checkpoints(projector_name text primary key, segment text, line_number int, updated_at real)`

## SQLite Mapping For Lifecycle V1

Event-to-table behavior:

- `experiment_start_requested`
  - optional in V1 projector
  - may be stored only in spool and in-memory registry
- `experiment_started`
  - insert or upsert row in `experiments`
  - status becomes `running`
  - populate config, timestamps, hypothesis metadata
- `experiment_start_failed`
  - if no `experiment_started` exists, write a failed launch record or separate launch-failure entry
  - do not mark the runner as active
- `experiment_completed`
  - update status to `completed`
  - set results and completion timestamps
- `experiment_failed`
  - update status to `failed`
  - set failure summary and completion timestamps

Design note:

`experiment_start_failed` is not the same as `experiment_failed`.
The first is a failed launch. The second is a failed started run.

## In-Memory Runtime State

Introduce a small runtime state registry fed by the bus.

Responsibilities:

- latest known lifecycle state for the active run
- current active experiment id
- last lifecycle event timestamp
- degraded flags such as `spool_unhealthy` and `projector_unhealthy`

This registry should back `/api/status` before SQLite heuristics do.

Fallback order for status resolution:

1. in-process lifecycle registry
2. projected lifecycle state from SQLite
3. legacy fallback heuristics during migration only

The migration is not complete until step 3 can be removed.

## File-Level Rollout

### New Modules

- `research/scientist/runtime_events/schema.py`
- `research/scientist/runtime_events/bus.py`
- `research/scientist/runtime_events/spool.py`
- `research/scientist/runtime_events/projectors/lifecycle_projector.py`
- `research/scientist/runtime_events/state_registry.py`

### First Existing Modules To Touch

- [notebook_experiments.py](/home/tim/Projects/LLM/research/scientist/notebook/notebook_experiments.py:329)
  - stop making direct write verification the source of launch truth
  - eventually demote these methods into projector helpers or compatibility wrappers
- [control_start.py](/home/tim/Projects/LLM/research/scientist/runner/control_start.py:36)
  - publish lifecycle events at launch boundaries
- [_experiment_launch.py](/home/tim/Projects/LLM/research/scientist/api_routes/_experiment_launch.py:1)
  - return success only after durable `experiment_started`
- [_helpers.py](/home/tim/Projects/LLM/research/scientist/api_routes/_helpers.py:187)
  - consult lifecycle registry before notebook fallback
- [core.py](/home/tim/Projects/LLM/research/scientist/runner/core.py:242)
  - stop assuming notebook status is the authoritative running-state source

### Leave Alone In The First Cut

- [notebook_advanced_analytics.py](/home/tim/Projects/LLM/research/scientist/notebook/notebook_advanced_analytics.py:614)
- telemetry-heavy runner modules such as [execution_screening.py](/home/tim/Projects/LLM/research/scientist/runner/execution_screening.py:820)

Those move in phase 2 after lifecycle truth is stable.

## Phase Plan

## Phase 0: Groundwork

- define `RuntimeEvent` schema and lifecycle event names
- define lifecycle state machine and duplicate/conflict policy
- decide exact event id format
- add spool directory creation and retention config

Exit criteria:

- no behavior change yet
- code compiles and tests for schema/state machine pass

## Phase 1: Bus And Spool

- implement in-process event bus
- implement append-only NDJSON spool
- implement lifecycle publisher with critical fsync semantics
- add a minimal in-memory lifecycle registry subscriber

Exit criteria:

- publishing `experiment_started` durably writes one spool record
- status registry updates without touching SQLite

## Phase 2: Lifecycle Projection

- implement lifecycle projector to `experiments`
- add projector checkpoint table/file
- make projector idempotent by `event_id`
- add degraded-mode logging and health surface

Exit criteria:

- replay from empty SQLite reconstructs lifecycle rows from the spool
- SQLite failure does not prevent event spool append or registry update

## Phase 3: Launch Path Migration

- route launch through lifecycle publisher
- return launch success only after durable `experiment_started`
- emit `experiment_start_failed` on launch failure
- update `/api/status` resolution order to registry first

Exit criteria:

- launch failure cannot leave phantom `is_running`
- launch success no longer depends on fresh SQLite readback

## Phase 4: Completion And Failure Migration

- route completion/failure through lifecycle publisher
- projector updates `experiments` terminal states
- preserve results payload handling, but keep lifecycle durability independent from notebook health

Exit criteria:

- final lifecycle state is recoverable from spool alone
- SQLite projector failure no longer loses the authoritative terminal state

## Phase 5: Telemetry Migration

- convert `log_learning_event()` to publish best-effort events
- move healer/report/cleanup bookkeeping behind event consumers
- keep telemetry sink failures non-fatal

Exit criteria:

- hot-path telemetry write failures do not abort valid runs

## Contract Enforcement

After lifecycle is live, enforce the boundary.

Recommended guardrails:

- add a test that forbids new `nb.conn.execute(...)` writes in runner hot paths
- allow direct notebook writes only under notebook query modules and projector modules
- add one grep-based CI test first if a full linter rule is too expensive

Initial enforcement target:

- `research/scientist/runner/`
- `research/scientist/api_routes/`

Exclude for now:

- `research/scientist/notebook/`
- analytics/query readers

## Testing Matrix

Required tests for V1:

- publish `experiment_started` and verify spool append plus fsync path
- replay lifecycle spool into empty SQLite and recover a completed run
- projector sees duplicate event and applies it once
- projector sees conflicting terminal event and preserves first terminal state
- SQLite projector failure leaves runner status alive in the in-memory registry
- launch failure after `start_requested` emits `experiment_start_failed` and leaves `is_running == false`
- startup after crash rebuilds active/terminal lifecycle state from spool

Required fault injection:

- raise `sqlite3.OperationalError` during projection
- raise write/fsync error during critical publish
- simulate process crash after durable `experiment_started` but before projector commit

## Operational Notes

Expose health somewhere visible:

- spool writer health
- projector health
- last projected event id
- lag in events or lines

This avoids repeating the current situation where SQLite health problems look like runner problems.

## Open Questions

- Should `experiment_start_requested` create a durable record if the run never starts, or is spool-only enough?
- Should lifecycle projector own experiment id generation, or should the launcher allocate ids before emitting?
- Do we want one generic projector process/thread, or one projector class invoked inline on a background worker?

My recommendation:

- allocate experiment id before publishing
- keep `start_requested` in the spool even if not projected initially
- run one background projector thread in-process for V1

## Recommended First PR

Keep the first PR smaller than the full migration:

- add schema, bus, spool, and registry
- add lifecycle state machine tests
- publish lifecycle events in parallel with existing notebook writes
- add observability for spool/projector health

Do not remove old notebook lifecycle writes in the first PR.

That gives a safe shadow mode. Once the spool path is proven, switch `/api/status` and launch truth to lifecycle events, then demote the legacy direct-write path.

## Definition Of Done

This migration is only done when all of the following are true:

- launch success is defined by durable `experiment_started`
- launch failure emits `experiment_start_failed` and cannot leave phantom running state
- final run state is recoverable from the spool without relying on SQLite
- `/api/status` prefers lifecycle registry/projected state over notebook heuristics
- projector SQLite failures do not abort valid experiment execution
- new hot-path direct notebook writes are blocked by tests

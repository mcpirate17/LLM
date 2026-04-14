# Runtime Event Bus Contract Audit

## Scope

Audit date: 2026-04-13

Companion migration matrix:

- [runtime_event_bus_migration_matrix_2026-04-13.md](/home/tim/Projects/LLM/research/docs/runtime_event_bus_migration_matrix_2026-04-13.md)

Focus:

- current runtime event bus adoption
- direct notebook writes that still violate or bypass the event-bus contract
- high-priority publishers still missing
- contract boundaries for publishers, subscribers, projector-only code, and read-only/query code

## Current Adoption

The event bus is no longer just scaffolding. It is already part of runtime status and launch flow.

Current runtime-event infrastructure:

- bootstrap singleton and worker ownership:
  - [bootstrap.py](/home/tim/Projects/LLM/research/scientist/runtime_events/bootstrap.py:1)
- bus typed subscriptions and health:
  - [bus.py](/home/tim/Projects/LLM/research/scientist/runtime_events/bus.py:1)
- lifecycle projector:
  - [lifecycle_projector.py](/home/tim/Projects/LLM/research/scientist/runtime_events/projectors/lifecycle_projector.py:1)
- background projector worker:
  - [workers.py](/home/tim/Projects/LLM/research/scientist/runtime_events/workers.py:1)

Current publishers:

- launch requested:
  - [api_routes/_experiment_launch.py](/home/tim/Projects/LLM/research/scientist/api_routes/_experiment_launch.py:58)
- launch failed:
  - [api_routes/_experiment_launch.py](/home/tim/Projects/LLM/research/scientist/api_routes/_experiment_launch.py:82)
- runtime lifecycle forwarding from runner event stream:
  - [runner/control_actions.py](/home/tim/Projects/LLM/research/scientist/runner/control_actions.py:166)

Current consumers:

- registry-first status resolution:
  - [api_routes/_helpers.py](/home/tim/Projects/LLM/research/scientist/api_routes/_helpers.py:194)
- registry snapshot builder:
  - [api_routes/_helpers.py](/home/tim/Projects/LLM/research/scientist/api_routes/_helpers.py:273)
- projected-running fallback:
  - [api_routes/_helpers.py](/home/tim/Projects/LLM/research/scientist/api_routes/_helpers.py:204)

Net:

- the system now has a real publisher path
- the system now has real consumers
- lifecycle truth is still incomplete because terminal and many start paths still write directly to SQLite/notebook helpers

## Highest-Risk Contract Violations

These are the places most likely to keep SQLite in the runtime control plane.

### 1. Terminal Lifecycle Still Goes Through Notebook Methods

Single-run screening thread:

- completion:
  - [execution_screening.py](/home/tim/Projects/LLM/research/scientist/runner/execution_screening.py:883)
- failure:
  - [execution_screening.py](/home/tim/Projects/LLM/research/scientist/runner/execution_screening.py:983)
  - [execution_screening.py](/home/tim/Projects/LLM/research/scientist/runner/execution_screening.py:1004)

Other execution families:

- investigation:
  - [execution_investigation.py](/home/tim/Projects/LLM/research/scientist/runner/execution_investigation.py:489)
  - [execution_investigation.py](/home/tim/Projects/LLM/research/scientist/runner/execution_investigation.py:1170)
- validation:
  - [execution_validation.py](/home/tim/Projects/LLM/research/scientist/runner/execution_validation.py:134)
  - [execution_validation.py](/home/tim/Projects/LLM/research/scientist/runner/execution_validation.py:257)
  - [execution_validation.py](/home/tim/Projects/LLM/research/scientist/runner/execution_validation.py:280)
  - [execution_validation.py](/home/tim/Projects/LLM/research/scientist/runner/execution_validation.py:343)
  - [execution_validation.py](/home/tim/Projects/LLM/research/scientist/runner/execution_validation.py:366)
- search:
  - [execution_search.py](/home/tim/Projects/LLM/research/scientist/runner/execution_search.py:243)
  - [execution_search.py](/home/tim/Projects/LLM/research/scientist/runner/execution_search.py:299)
  - [execution_search.py](/home/tim/Projects/LLM/research/scientist/runner/execution_search.py:320)
  - [execution_search.py](/home/tim/Projects/LLM/research/scientist/runner/execution_search.py:611)
  - [execution_search.py](/home/tim/Projects/LLM/research/scientist/runner/execution_search.py:668)
  - [execution_search.py](/home/tim/Projects/LLM/research/scientist/runner/execution_search.py:689)

Why this matters:

- lifecycle completion/failure is still acknowledged through `nb.complete_experiment(...)` and `nb.fail_experiment(...)`
- that means terminal lifecycle truth can still fail with SQLite rather than first succeeding in the spool

### 2. Launch Paths Still Depend On Notebook Lifecycle Helpers

Primary start methods:

- [control_start.py](/home/tim/Projects/LLM/research/scientist/runner/control_start.py:28)
- [control_start.py](/home/tim/Projects/LLM/research/scientist/runner/control_start.py:197)
- [control_start.py](/home/tim/Projects/LLM/research/scientist/runner/control_start.py:511)
- [control_start.py](/home/tim/Projects/LLM/research/scientist/runner/control_start.py:615)
- [control_start.py](/home/tim/Projects/LLM/research/scientist/runner/control_start.py:713)
- [control_start.py](/home/tim/Projects/LLM/research/scientist/runner/control_start.py:783)
- [control_start.py](/home/tim/Projects/LLM/research/scientist/runner/control_start.py:850)
- [control_start.py](/home/tim/Projects/LLM/research/scientist/runner/control_start.py:917)

Persistence helper still acting as source of truth:

- [notebook_experiments.py](/home/tim/Projects/LLM/research/scientist/notebook/notebook_experiments.py:329)
- [notebook_experiments.py](/home/tim/Projects/LLM/research/scientist/notebook/notebook_experiments.py:457)
- [notebook_experiments.py](/home/tim/Projects/LLM/research/scientist/notebook/notebook_experiments.py:570)

Why this matters:

- `experiment_start_requested` and launch-failure events exist now
- but `experiment_started` is not yet consistently emitted at the runner boundary for all run families
- notebook lifecycle methods are still persistence primitives and control-plane state at the same time

### 3. Shutdown And Recovery Still Write Directly To SQLite

Shutdown interruption path:

- [runner/core.py](/home/tim/Projects/LLM/research/scientist/runner/core.py:229)

Cycle-level failure path:

- [runner/cycle.py](/home/tim/Projects/LLM/research/scientist/runner/cycle.py:1021)

Stale experiment cleanup and cancellation:

- [notebook_experiments.py](/home/tim/Projects/LLM/research/scientist/notebook/notebook_experiments.py:135)
- [notebook_experiments.py](/home/tim/Projects/LLM/research/scientist/notebook/notebook_experiments.py:723)

Why this matters:

- these code paths can resurrect SQLite as the only durable truth during crashes, resumes, stale cleanup, and user cancellation

## Telemetry And Hot-Path Write Violations

These should become best-effort event producers, not synchronous notebook writes.

### Learning Log / Analytics

Central sink:

- [notebook_advanced_analytics.py](/home/tim/Projects/LLM/research/scientist/notebook/notebook_advanced_analytics.py:614)

Hot-path producers:

- [execution_screening.py](/home/tim/Projects/LLM/research/scientist/runner/execution_screening.py:820)
- [execution_screening.py](/home/tim/Projects/LLM/research/scientist/runner/execution_screening.py:1066)
- [execution_screening.py](/home/tim/Projects/LLM/research/scientist/runner/execution_screening.py:1112)
- [execution_screening.py](/home/tim/Projects/LLM/research/scientist/runner/execution_screening.py:1126)
- [execution_screening.py](/home/tim/Projects/LLM/research/scientist/runner/execution_screening.py:1304)
- [execution_screening.py](/home/tim/Projects/LLM/research/scientist/runner/execution_screening.py:1316)
- [execution_screening.py](/home/tim/Projects/LLM/research/scientist/runner/execution_screening.py:1406)
- [execution_screening.py](/home/tim/Projects/LLM/research/scientist/runner/execution_screening.py:1452)
- [execution_screening.py](/home/tim/Projects/LLM/research/scientist/runner/execution_screening.py:1458)
- [cycle.py](/home/tim/Projects/LLM/research/scientist/runner/cycle.py:394)
- [cycle.py](/home/tim/Projects/LLM/research/scientist/runner/cycle.py:497)
- [cycle.py](/home/tim/Projects/LLM/research/scientist/runner/cycle.py:569)
- [cycle.py](/home/tim/Projects/LLM/research/scientist/runner/cycle.py:635)
- [cycle.py](/home/tim/Projects/LLM/research/scientist/runner/cycle.py:721)
- [cycle.py](/home/tim/Projects/LLM/research/scientist/runner/cycle.py:807)
- [cycle.py](/home/tim/Projects/LLM/research/scientist/runner/cycle.py:822)
- [continuous_modes.py](/home/tim/Projects/LLM/research/scientist/runner/continuous_modes.py:188)
- [results_automation.py](/home/tim/Projects/LLM/research/scientist/runner/results_automation.py:207)
- [results_automation.py](/home/tim/Projects/LLM/research/scientist/runner/results_automation.py:221)
- [results_automation.py](/home/tim/Projects/LLM/research/scientist/runner/results_automation.py:233)
- [results_auto_escalate_phase7.py](/home/tim/Projects/LLM/research/scientist/runner/results_auto_escalate_phase7.py:378)
- [dashboard.py](/home/tim/Projects/LLM/research/scientist/runner/dashboard.py:898)
- [synthesis.py](/home/tim/Projects/LLM/research/scientist/runner/synthesis.py:188)
- [synthesis.py](/home/tim/Projects/LLM/research/scientist/runner/synthesis.py:241)
- [synthesis.py](/home/tim/Projects/LLM/research/scientist/runner/synthesis.py:409)
- [analytics_bp.py](/home/tim/Projects/LLM/research/scientist/api_routes/analytics_bp.py:145)

Why this matters:

- these are noncritical writes
- they should not abort experiment launch, execution, cleanup, or reporting

### Direct Writes In Runner Helper And Maintenance Paths

- [runner/_helpers.py](/home/tim/Projects/LLM/research/scientist/runner/_helpers.py:1526)
- [continuous_investigation.py](/home/tim/Projects/LLM/research/scientist/runner/continuous_investigation.py:319)
- [continuous_investigation.py](/home/tim/Projects/LLM/research/scientist/runner/continuous_investigation.py:460)
- [continuous_modes.py](/home/tim/Projects/LLM/research/scientist/runner/continuous_modes.py:197)
- [continuous_modes.py](/home/tim/Projects/LLM/research/scientist/runner/continuous_modes.py:208)
- [control_actions.py](/home/tim/Projects/LLM/research/scientist/runner/control_actions.py:452)
- [control_actions.py](/home/tim/Projects/LLM/research/scientist/runner/control_actions.py:477)
- [control_actions.py](/home/tim/Projects/LLM/research/scientist/runner/control_actions.py:495)

These need classification before migration:

- true runtime control-plane writes
- maintenance/admin-only writes
- projector-only candidates

## API-Side Direct Writes

These are likely contract violations if they happen on live runtime paths rather than purely administrative endpoints.

Probable write endpoints:

- [experiments_bp.py](/home/tim/Projects/LLM/research/scientist/api_routes/experiments_bp.py:93)
- [programs_bp.py](/home/tim/Projects/LLM/research/scientist/api_routes/programs_bp.py:147)
- [programs_bp.py](/home/tim/Projects/LLM/research/scientist/api_routes/programs_bp.py:727)
- [programs_bp.py](/home/tim/Projects/LLM/research/scientist/api_routes/programs_bp.py:739)
- [programs_bp.py](/home/tim/Projects/LLM/research/scientist/api_routes/programs_bp.py:852)
- [leaderboard_bp.py](/home/tim/Projects/LLM/research/scientist/api_routes/leaderboard_bp.py:496)

These need manual classification:

- if they mutate experiment lifecycle or other runtime truth, move them behind publish/projector flow
- if they are admin or curation operations, they may remain direct DB writes outside the runtime hot path

## Likely Read-Only / Query Paths

These are noisy in grep but should not be forced onto the event bus.

Examples:

- strategy/readout endpoints:
  - [strategy_bp.py](/home/tim/Projects/LLM/research/scientist/api_routes/strategy_bp.py:129)
  - [strategy_briefing.py](/home/tim/Projects/LLM/research/scientist/api_routes/_strategy_briefing.py:65)
  - [strategy_recommendations.py](/home/tim/Projects/LLM/research/scientist/api_routes/_strategy_recommendations.py:22)
  - [strategy_diagnostics.py](/home/tim/Projects/LLM/research/scientist/api_routes/_strategy_diagnostics.py:49)
- observability/reporting reads:
  - [observability_bp.py](/home/tim/Projects/LLM/research/scientist/api_routes/observability_bp.py:90)
  - [reporting_bp.py](/home/tim/Projects/LLM/research/scientist/api_routes/reporting_bp.py:90)
  - [reporting_bp.py](/home/tim/Projects/LLM/research/scientist/api_routes/reporting_bp.py:113)
- notebook query modules:
  - [program_query_views.py](/home/tim/Projects/LLM/research/scientist/notebook/program_query_views.py:58)
  - large parts of [notebook_programs.py](/home/tim/Projects/LLM/research/scientist/notebook/notebook_programs.py:50)
  - large parts of [notebook_misc.py](/home/tim/Projects/LLM/research/scientist/notebook/notebook_misc.py:398)

Recommendation:

- do not try to move reads onto the event bus
- the bus is for runtime coordination and async persistence, not for replacing query access

## Publisher/Subscriber Boundary

### Publisher Code

Should emit events, not write SQLite directly, when operating on live runtime state:

- API launch boundary:
  - [api_routes/_experiment_launch.py](/home/tim/Projects/LLM/research/scientist/api_routes/_experiment_launch.py:58)
  - [api_routes/_experiment_launch.py](/home/tim/Projects/LLM/research/scientist/api_routes/_experiment_launch.py:82)
- runner launch methods:
  - [control_start.py](/home/tim/Projects/LLM/research/scientist/runner/control_start.py:28)
  - [control_start.py](/home/tim/Projects/LLM/research/scientist/runner/control_start.py:197)
  - [control_start.py](/home/tim/Projects/LLM/research/scientist/runner/control_start.py:511)
  - [control_start.py](/home/tim/Projects/LLM/research/scientist/runner/control_start.py:615)
  - [control_start.py](/home/tim/Projects/LLM/research/scientist/runner/control_start.py:713)
  - [control_start.py](/home/tim/Projects/LLM/research/scientist/runner/control_start.py:783)
  - [control_start.py](/home/tim/Projects/LLM/research/scientist/runner/control_start.py:850)
  - [control_start.py](/home/tim/Projects/LLM/research/scientist/runner/control_start.py:917)
- runner terminal transitions:
  - [execution_screening.py](/home/tim/Projects/LLM/research/scientist/runner/execution_screening.py:883)
  - [execution_screening.py](/home/tim/Projects/LLM/research/scientist/runner/execution_screening.py:983)
  - [execution_investigation.py](/home/tim/Projects/LLM/research/scientist/runner/execution_investigation.py:489)
  - [execution_validation.py](/home/tim/Projects/LLM/research/scientist/runner/execution_validation.py:134)
  - [execution_search.py](/home/tim/Projects/LLM/research/scientist/runner/execution_search.py:243)
- recovery/cancellation:
  - [runner/core.py](/home/tim/Projects/LLM/research/scientist/runner/core.py:229)
  - [runner/cycle.py](/home/tim/Projects/LLM/research/scientist/runner/cycle.py:1021)

### Subscriber Code

Should consume typed events and update in-memory runtime state, live feeds, and best-effort side channels:

- status registry:
  - [state_registry.py](/home/tim/Projects/LLM/research/scientist/runtime_events/state_registry.py:1)
- runtime status helpers:
  - [api_routes/_helpers.py](/home/tim/Projects/LLM/research/scientist/api_routes/_helpers.py:194)
- existing runner event fan-out:
  - [control_actions.py](/home/tim/Projects/LLM/research/scientist/runner/control_actions.py:166)

Future subscriber candidates:

- telemetry projector(s)
- healer task projector(s)
- report progress projector(s)
- dashboard live-feed adapter if it should listen directly to bus events

### Projector-Only Code

These should become persistence sinks, not publishers:

- [notebook_experiments.py](/home/tim/Projects/LLM/research/scientist/notebook/notebook_experiments.py:329)
- [notebook_experiments.py](/home/tim/Projects/LLM/research/scientist/notebook/notebook_experiments.py:457)
- [notebook_experiments.py](/home/tim/Projects/LLM/research/scientist/notebook/notebook_experiments.py:570)
- [notebook_advanced_analytics.py](/home/tim/Projects/LLM/research/scientist/notebook/notebook_advanced_analytics.py:614)
- healer notebook writes:
  - [notebook_healer.py](/home/tim/Projects/LLM/research/scientist/notebook/notebook_healer.py:31)

## Recommended Priority Order

### P0

- publish `experiment_started` in every `control_start.py` launch method
- publish `experiment_completed` and `experiment_failed` at execution thread terminal boundaries
- stop relying on notebook lifecycle helpers as the first durable acknowledgment

### P1

- move shutdown/interruption/cycle-failure/cancel/stale-recovery flows onto lifecycle events
- make notebook lifecycle methods compatibility sinks only

### P2

- convert `log_learning_event(...)` callers into best-effort telemetry producers
- add telemetry/healer/report projectors

### P3

- classify admin-only API writes and leave those out of the runtime contract if they are not hot-path runtime coordination

## Enforcement Recommendation

Add one grep-based CI/unit test first.

Suggested initial rule:

- forbid new direct write patterns under `research/scientist/runner/` and `research/scientist/api_routes/`:
  - `nb.conn.execute(`
  - `nb._maybe_commit(`
  - `nb.start_experiment(`
  - `nb.complete_experiment(`
  - `nb.fail_experiment(`
  - `nb.log_learning_event(`

Allowlist temporarily:

- explicit admin/maintenance endpoints after classification
- runtime event projectors
- notebook modules themselves until projector migration is complete

## Bottom Line

The event bus now exists in real runtime paths, but the contract is still only partially enforced.

The largest remaining architectural risk is not missing infrastructure. It is the amount of existing runner code that still treats notebook writes as immediate runtime truth, especially:

- terminal lifecycle transitions
- noncritical learning/analytics writes in hot paths
- shutdown/recovery/cancel flows

That is the surface area the next migration work should target.

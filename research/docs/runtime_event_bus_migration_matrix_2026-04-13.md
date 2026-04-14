# Runtime Event Bus Migration Matrix

## Purpose

This is the execution companion to
[runtime_event_bus_contract_audit_2026-04-13.md](/home/tim/Projects/LLM/research/docs/runtime_event_bus_contract_audit_2026-04-13.md).

Each call site below is classified as one of:

- `publish event`
- `projector-only`
- `admin-only direct write`
- `read-only query`

The goal is to turn the audit into an implementation queue.

## Classification Rules

- `publish event`
  - live runtime or API orchestration path
  - current code is mutating runtime truth or noncritical runtime telemetry
  - should emit a typed event and let subscribers/projectors persist it

- `projector-only`
  - persistence sink or compatibility notebook helper
  - should not be the primary source of lifecycle truth

- `admin-only direct write`
  - maintenance, repair, curation, or operator action outside the experiment hot path
  - can remain direct DB mutation if clearly separated from runtime coordination

- `read-only query`
  - query/reporting/read model
  - should not be moved onto the event bus

## P0: Lifecycle Start And Terminal Truth

| Area | Call Site | Current Pattern | Classification | Target |
|---|---|---|---|---|
| API launch | [_experiment_launch.py:58](</home/tim/Projects/LLM/research/scientist/api_routes/_experiment_launch.py:58>) | `publish_launch_requested()` | `publish event` | Keep as `experiment_start_requested`; add correlation to eventual real experiment id |
| API launch | [_experiment_launch.py:82](</home/tim/Projects/LLM/research/scientist/api_routes/_experiment_launch.py:82>) | `publish_launch_failed()` | `publish event` | Keep as `experiment_start_failed`; include launch correlation metadata |
| Runner synthesis start | [control_start.py:144](</home/tim/Projects/LLM/research/scientist/runner/control_start.py:144>) | `_emit_event("experiment_started", ...)` | `publish event` | Keep, but make this explicit lifecycle publish contract rather than SSE-name matching |
| Runner continuous start | [control_start.py:197](</home/tim/Projects/LLM/research/scientist/runner/control_start.py:197>) | mode start with no lifecycle event | `publish event` | Add canonical `experiment_started` or define explicit non-experiment session contract |
| Runner investigation start | [control_start.py:597](</home/tim/Projects/LLM/research/scientist/runner/control_start.py:597>) | `_emit_event("investigation_started", ...)` | `publish event` | Also publish canonical `experiment_started` |
| Runner validation start | [control_start.py:696](</home/tim/Projects/LLM/research/scientist/runner/control_start.py:696>) | `_emit_event("validation_started", ...)` | `publish event` | Also publish canonical `experiment_started` |
| Runner scale-up start | [control_start.py:761](</home/tim/Projects/LLM/research/scientist/runner/control_start.py:761>) | `_emit_event("scale_up_started", ...)` | `publish event` | Also publish canonical `experiment_started` |
| Runner evolution start | [control_start.py:833](</home/tim/Projects/LLM/research/scientist/runner/control_start.py:833>) | `_emit_event("evolution_started", ...)` | `publish event` | Also publish canonical `experiment_started` |
| Runner novelty start | [control_start.py:900](</home/tim/Projects/LLM/research/scientist/runner/control_start.py:900>) | `_emit_event("novelty_started", ...)` | `publish event` | Also publish canonical `experiment_started` |
| Runner resume | [control_start.py:958](</home/tim/Projects/LLM/research/scientist/runner/control_start.py:958>) | direct `UPDATE experiments SET status='running'` | `publish event` | Replace with explicit resume lifecycle event or canonical `experiment_started` with resume metadata |
| Screening terminal success | [execution_screening.py:883](</home/tim/Projects/LLM/research/scientist/runner/execution_screening.py:883>) and [953](</home/tim/Projects/LLM/research/scientist/runner/execution_screening.py:953>) | `nb.complete_experiment(...)` plus `_emit_event("experiment_completed", ...)` | `publish event` | Make event the durable first-class acknowledgment; demote notebook write to compatibility/projector path |
| Screening terminal failure | [execution_screening.py:983](</home/tim/Projects/LLM/research/scientist/runner/execution_screening.py:983>), [1004](</home/tim/Projects/LLM/research/scientist/runner/execution_screening.py:1004>), [990](</home/tim/Projects/LLM/research/scientist/runner/execution_screening.py:990>) | `nb.fail_experiment(...)` plus `_emit_event("experiment_failed", ...)` | `publish event` | Same as above; event first |
| Investigation terminal | [execution_investigation.py:489](</home/tim/Projects/LLM/research/scientist/runner/execution_investigation.py:489>), [1170](</home/tim/Projects/LLM/research/scientist/runner/execution_investigation.py:1170>) | notebook lifecycle write | `publish event` | Add canonical `experiment_completed` / `experiment_failed` |
| Validation terminal | [execution_validation.py:134](</home/tim/Projects/LLM/research/scientist/runner/execution_validation.py:134>), [257](</home/tim/Projects/LLM/research/scientist/runner/execution_validation.py:257>), [280](</home/tim/Projects/LLM/research/scientist/runner/execution_validation.py:280>), [343](</home/tim/Projects/LLM/research/scientist/runner/execution_validation.py:343>), [366](</home/tim/Projects/LLM/research/scientist/runner/execution_validation.py:366>) | notebook lifecycle write | `publish event` | Add canonical terminal lifecycle events |
| Search terminal | [execution_search.py:243](</home/tim/Projects/LLM/research/scientist/runner/execution_search.py:243>), [299](</home/tim/Projects/LLM/research/scientist/runner/execution_search.py:299>), [320](</home/tim/Projects/LLM/research/scientist/runner/execution_search.py:320>), [611](</home/tim/Projects/LLM/research/scientist/runner/execution_search.py:611>), [668](</home/tim/Projects/LLM/research/scientist/runner/execution_search.py:668>), [689](</home/tim/Projects/LLM/research/scientist/runner/execution_search.py:689>) | notebook lifecycle write | `publish event` | Add canonical terminal lifecycle events |
| Continuous terminal | [continuous_modes.py:315](</home/tim/Projects/LLM/research/scientist/runner/continuous_modes.py:315>), [483](</home/tim/Projects/LLM/research/scientist/runner/continuous_modes.py:483>), [714](</home/tim/Projects/LLM/research/scientist/runner/continuous_modes.py:714>) | notebook lifecycle write | `publish event` | Decide whether each loop iteration is an experiment lifecycle or a session lifecycle |
| Continuous investigation terminal | [continuous_investigation.py:1263](</home/tim/Projects/LLM/research/scientist/runner/continuous_investigation.py:1263>), [1290](</home/tim/Projects/LLM/research/scientist/runner/continuous_investigation.py:1290>) | notebook lifecycle write | `publish event` | Add canonical terminal lifecycle events |
| Continuous validation terminal | [continuous_validation.py:728](</home/tim/Projects/LLM/research/scientist/runner/continuous_validation.py:728>), [748](</home/tim/Projects/LLM/research/scientist/runner/continuous_validation.py:748>) | notebook lifecycle write | `publish event` | Add canonical terminal lifecycle events |

## P1: Recovery, Cancel, And Compensating State

| Area | Call Site | Current Pattern | Classification | Target |
|---|---|---|---|---|
| Shutdown recovery | [core.py:241](</home/tim/Projects/LLM/research/scientist/runner/core.py:241>) and [246](</home/tim/Projects/LLM/research/scientist/runner/core.py:246>) | direct read/update of `experiments.status='interrupted'` | `publish event` | Add interruption lifecycle event or explicit compensating recovery event |
| Cycle abort | [cycle.py:1021](</home/tim/Projects/LLM/research/scientist/runner/cycle.py:1021>) | `nb.fail_experiment(...)` | `publish event` | Publish terminal failure first |
| Cancel API | [experiments_bp.py:359](</home/tim/Projects/LLM/research/scientist/api_routes/experiments_bp.py:359>) and [396](</home/tim/Projects/LLM/research/scientist/api_routes/experiments_bp.py:396>) | cancel/rerun lifecycle mutation | `publish event` | Add cancel/resume lifecycle contract |
| Notebook stale cleanup | [notebook_experiments.py:135](</home/tim/Projects/LLM/research/scientist/notebook/notebook_experiments.py:135>) | rewrites stale running rows | `admin-only direct write` | Treat as compensating recovery until replaced with replay/reconciliation |
| Notebook cancel helper | [notebook_experiments.py:723](</home/tim/Projects/LLM/research/scientist/notebook/notebook_experiments.py:723>) | direct status rewrite | `projector-only` | Demote to compatibility sink or remove from runtime boundary |

## P2: Telemetry And Best-Effort Runtime Side Effects

| Area | Call Site | Current Pattern | Classification | Target |
|---|---|---|---|---|
| Learning log sink | [notebook_advanced_analytics.py:614](</home/tim/Projects/LLM/research/scientist/notebook/notebook_advanced_analytics.py:614>) | `log_learning_event(...)` writes directly to SQLite | `projector-only` | Convert to telemetry projector sink |
| Screening telemetry | [execution_screening.py:820](</home/tim/Projects/LLM/research/scientist/runner/execution_screening.py:820>) and repeated calls at [1066](</home/tim/Projects/LLM/research/scientist/runner/execution_screening.py:1066>), [1112](</home/tim/Projects/LLM/research/scientist/runner/execution_screening.py:1112>), [1126](</home/tim/Projects/LLM/research/scientist/runner/execution_screening.py:1126>), [1304](</home/tim/Projects/LLM/research/scientist/runner/execution_screening.py:1304>), [1316](</home/tim/Projects/LLM/research/scientist/runner/execution_screening.py:1316>), [1406](</home/tim/Projects/LLM/research/scientist/runner/execution_screening.py:1406>), [1452](</home/tim/Projects/LLM/research/scientist/runner/execution_screening.py:1452>), [1458](</home/tim/Projects/LLM/research/scientist/runner/execution_screening.py:1458>) | `nb.log_learning_event(...)` in hot path | `publish event` | Add best-effort `learning_event_logged` / `analytics_event_logged` |
| Orchestration telemetry | [cycle.py:394](</home/tim/Projects/LLM/research/scientist/runner/cycle.py:394>), [497](</home/tim/Projects/LLM/research/scientist/runner/cycle.py:497>), [569](</home/tim/Projects/LLM/research/scientist/runner/cycle.py:569>), [635](</home/tim/Projects/LLM/research/scientist/runner/cycle.py:635>), [721](</home/tim/Projects/LLM/research/scientist/runner/cycle.py:721>), [807](</home/tim/Projects/LLM/research/scientist/runner/cycle.py:807>), [822](</home/tim/Projects/LLM/research/scientist/runner/cycle.py:822>) | repeated learning-log writes | `publish event` | Best-effort orchestration telemetry events |
| Continuous-mode telemetry | [continuous_modes.py:188](</home/tim/Projects/LLM/research/scientist/runner/continuous_modes.py:188>) | `nb.log_learning_event(...)` | `publish event` | Best-effort telemetry event |
| Automation telemetry | [results_automation.py:207](</home/tim/Projects/LLM/research/scientist/runner/results_automation.py:207>), [221](</home/tim/Projects/LLM/research/scientist/runner/results_automation.py:221>), [233](</home/tim/Projects/LLM/research/scientist/runner/results_automation.py:233>) | `nb.log_learning_event(...)` | `publish event` | Best-effort automation telemetry |
| Escalation telemetry | [results_auto_escalate_phase7.py:378](</home/tim/Projects/LLM/research/scientist/runner/results_auto_escalate_phase7.py:378>) | `nb.log_learning_event(...)` | `publish event` | Best-effort escalation telemetry |
| Dashboard-triggered telemetry | [dashboard.py:898](</home/tim/Projects/LLM/research/scientist/runner/dashboard.py:898>) | `nb.log_learning_event(...)` | `publish event` | Best-effort UI/runtime telemetry |
| Synthesis telemetry | [synthesis.py:188](</home/tim/Projects/LLM/research/scientist/runner/synthesis.py:188>), [241](</home/tim/Projects/LLM/research/scientist/runner/synthesis.py:241>), [409](</home/tim/Projects/LLM/research/scientist/runner/synthesis.py:409>) | `nb.log_learning_event(...)` | `publish event` | Best-effort synthesis telemetry |
| API analytics action | [analytics_bp.py:145](</home/tim/Projects/LLM/research/scientist/api_routes/analytics_bp.py:145>) | API-triggered learning log write | `publish event` | Best-effort analytics action event |

## P3: Maintenance And Admin-Only Direct Writes

| Area | Call Site | Current Pattern | Classification | Target |
|---|---|---|---|---|
| Runner helper maintenance | [runner/_helpers.py:1526](</home/tim/Projects/LLM/research/scientist/runner/_helpers.py:1526>) | direct notebook mutation | `admin-only direct write` | Keep if not on experiment hot path; otherwise split out |
| Continuous investigation maintenance | [continuous_investigation.py:319](</home/tim/Projects/LLM/research/scientist/runner/continuous_investigation.py:319>) and [460](</home/tim/Projects/LLM/research/scientist/runner/continuous_investigation.py:460>) | direct notebook mutation | `admin-only direct write` | Keep only if purely maintenance; otherwise convert |
| Continuous-mode linking writes | [continuous_modes.py:197](</home/tim/Projects/LLM/research/scientist/runner/continuous_modes.py:197>) and [208](</home/tim/Projects/LLM/research/scientist/runner/continuous_modes.py:208>) | experiment/hypothesis linking updates | `publish event` | If these affect runtime truth, convert to evented metadata updates |
| Control actions admin ops | [control_actions.py:452](</home/tim/Projects/LLM/research/scientist/runner/control_actions.py:452>), [477](</home/tim/Projects/LLM/research/scientist/runner/control_actions.py:477>), [495](</home/tim/Projects/LLM/research/scientist/runner/control_actions.py:495>) | maintenance writes, WAL ops, signature backfill | `admin-only direct write` | Keep outside runtime bus contract |
| Reporting backfill | [reporting_bp.py:486](</home/tim/Projects/LLM/research/scientist/api_routes/reporting_bp.py:486>) and [499](</home/tim/Projects/LLM/research/scientist/api_routes/reporting_bp.py:499>) | recompute/backfill | `admin-only direct write` | Keep as maintenance/backfill |
| Experiment metrics backfill | [experiments_bp.py:471](</home/tim/Projects/LLM/research/scientist/api_routes/experiments_bp.py:471>) | repair/backfill write | `admin-only direct write` | Keep as maintenance/backfill |
| Leaderboard/admin curation | [leaderboard_bp.py:496](</home/tim/Projects/LLM/research/scientist/api_routes/leaderboard_bp.py:496>) | direct curation write | `admin-only direct write` | Keep out of runtime contract unless it affects active run truth |
| Program curation endpoints | [programs_bp.py:147](</home/tim/Projects/LLM/research/scientist/api_routes/programs_bp.py:147>), [727](</home/tim/Projects/LLM/research/scientist/api_routes/programs_bp.py:727>), [739](</home/tim/Projects/LLM/research/scientist/api_routes/programs_bp.py:739>), [852](</home/tim/Projects/LLM/research/scientist/api_routes/programs_bp.py:852>) | direct program/leaderboard mutation | `admin-only direct write` | Keep as curation path unless used during active runtime orchestration |

## Projector-Only And Compatibility Sinks

| Area | Call Site | Current Pattern | Classification | Target |
|---|---|---|---|---|
| Notebook lifecycle create | [notebook_experiments.py:329](</home/tim/Projects/LLM/research/scientist/notebook/notebook_experiments.py:329>) | `start_experiment(...)` writes and verifies durability directly | `projector-only` | Demote to compatibility sink behind lifecycle projector |
| Notebook lifecycle complete | [notebook_experiments.py:457](</home/tim/Projects/LLM/research/scientist/notebook/notebook_experiments.py:457>) | `complete_experiment(...)` writes terminal state | `projector-only` | Demote to compatibility sink behind lifecycle projector |
| Notebook lifecycle fail | [notebook_experiments.py:570](</home/tim/Projects/LLM/research/scientist/notebook/notebook_experiments.py:570>) | `fail_experiment(...)` writes terminal state | `projector-only` | Demote to compatibility sink behind lifecycle projector |
| Notebook healer writes | [notebook_healer.py:31](</home/tim/Projects/LLM/research/scientist/notebook/notebook_healer.py:31>) | direct healer task/event persistence | `projector-only` | Future healer projector sink |
| Notebook analytics sink | [notebook_advanced_analytics.py:614](</home/tim/Projects/LLM/research/scientist/notebook/notebook_advanced_analytics.py:614>) | direct learning log persistence | `projector-only` | Future telemetry projector sink |

## Read-Only Queries To Leave Alone

| Area | Call Site | Current Pattern | Classification | Target |
|---|---|---|---|---|
| Status consumer | [_helpers.py:194](</home/tim/Projects/LLM/research/scientist/api_routes/_helpers.py:194>) | registry/projector/notebook read resolution | `read-only query` | Keep as consumer |
| Registry snapshot | [_helpers.py:273](</home/tim/Projects/LLM/research/scientist/api_routes/_helpers.py:273>) | lifecycle registry read | `read-only query` | Keep as consumer |
| Observability core | [_observability_core.py:120](</home/tim/Projects/LLM/research/scientist/api_routes/_observability_core.py:120>) and related reads | SQL reporting reads | `read-only query` | Leave alone |
| Strategy and recommendations | [strategy_bp.py:129](</home/tim/Projects/LLM/research/scientist/api_routes/strategy_bp.py:129>), [_strategy_recommendations.py:22](</home/tim/Projects/LLM/research/scientist/api_routes/_strategy_recommendations.py:22>) | query/reporting reads | `read-only query` | Leave alone |
| Auto-escalate data | [auto_escalate_data.py:16](</home/tim/Projects/LLM/research/scientist/runner/auto_escalate_data.py:16>) through [132](</home/tim/Projects/LLM/research/scientist/runner/auto_escalate_data.py:132>) | helper reads | `read-only query` | Leave alone |
| Notebook query modules | [program_query_views.py:58](</home/tim/Projects/LLM/research/scientist/notebook/program_query_views.py:58>) and read-heavy notebook query helpers | SQL read model | `read-only query` | Leave alone |

## Execution Order

1. Convert every start path in `control_start.py` to publish canonical `experiment_started`.
2. Convert every terminal execution path to publish canonical `experiment_completed` / `experiment_failed` before compatibility notebook writes.
3. Add a recovery/cancel lifecycle contract for resume, interruption, cycle abort, and cancel.
4. Convert `log_learning_event(...)` callers into best-effort telemetry producers.
5. Explicitly mark admin-only direct-write endpoints as out-of-contract.
6. Add a guard test that fails new runtime direct-write patterns in `runner/` and `api_routes/`.

## Immediate Guardrail Candidates

Search patterns to block in runtime orchestration code:

- `nb.start_experiment(`
- `nb.complete_experiment(`
- `nb.fail_experiment(`
- `nb.log_learning_event(`
- `nb.conn.execute(`
- `nb._maybe_commit(`

Initial scope:

- `research/scientist/runner/`
- `research/scientist/api_routes/`

Initial allowlist:

- `research/scientist/runtime_events/`
- notebook modules under `research/scientist/notebook/`
- explicitly classified admin-only maintenance endpoints

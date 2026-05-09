# Lab Notebook DB Split Manifest

Generated: 2026-05-09

Source DB: `research/lab_notebook.db`

Integrity baseline: `python -m research.tools.db_health --db research/lab_notebook.db`
reported `quick_check: ok`.

This manifest is Phase 1 inventory only. It does not authorize deleting or
truncating any current DB, artifact directory, backup table, runtime log, or
backup bundle.

## Target Stores

- `runs`: `research/runs.db`
- `events`: `research/events.db` or append-only NDJSON segments, selected in
  Phase 2 by query needs.
- `artifact_metadata`: copy with the runs DB because active run rows point to
  compressed artifacts under `research/artifacts/notebook`.
- `legacy`: retained in place during split; copy only if a restore/audit path
  still requires it.

## Read Path Notes

- `LabNotebook.get_program_detail()` resolves artifact pointer JSON before
  parsing known program-result JSON fields.
- `LabNotebook.get_training_curve()` reads current compressed curve artifacts
  through `notebook_artifacts` when inline `training_curves` rows are absent.
- Dashboard live-feed replay now reads runtime-spool `live_feed` events and
  legacy `entries(entry_type='live_feed')` rows during the transition.
- Raw SQL API/dashboard paths that read `graph_json` still expect inline JSON.
  `graph_json` is not in the current program-result externalization set, so
  these paths remain compatible for Phase 1.
- Raw SQL sorting/filtering should not target newly externalized JSON fields
  until those reads move through notebook APIs or get summary columns.

## Phase 2 Event Split Decision

- High-volume dashboard live-feed replay events use the existing append-only
  runtime NDJSON spool under `research/runtime_events`.
- Selected runner live-feed events are written as runtime events with
  `event_type='live_feed'`; the original dashboard event type remains in
  payload metadata.
- `/api/live-feed` merges new spool-backed events with legacy DB-backed
  `live_feed` entries so existing historical rows still render.
- Lifecycle projector bookkeeping (`applied_runtime_events` and
  `runtime_projector_checkpoints`) now uses sidecar `events.db` next to the
  active notebook DB. Experiment status projection still updates the notebook
  `experiments` table because those rows are run-control state, not event log
  retention.

## Phase 3 Runs DB Cutover

- `research/runs.db` was built from a read-only snapshot of
  `research/lab_notebook.db` with `python -m research.tools.create_runs_db`.
- The builder keeps active run/control tables and drops legacy backup tables,
  `repair_log`, `applied_runtime_events`, and `runtime_projector_checkpoints`
  from the runs copy.
- Core row counts matched after copy:
  - `experiments`: 3812
  - `program_results`: 21628
  - `leaderboard`: 8100
  - `followup_tasks`: 350
  - `notebook_artifacts`: 35151
  - `program_graph_features`: 22216
  - `program_graph_ops`: 175733
  - `program_graph_pairs`: 232349
- Active defaults for `LabNotebook`, dashboard API startup, runner startup,
  scheduler caches, graph-feature enrichment, slot-constraint loading, and
  intelligence predictor training helpers now resolve to `RUNS_DB`.
- `research/lab_notebook.db` remains present as the legacy compatibility source
  and rebuild source for `research/runs.db`.

## Phase 4 Artifact Expansion

- `python -m research.tools.externalize_notebook_artifacts --db research/runs.db
  --min-bytes 2048 --apply --vacuum` externalized the remaining safe large
  payloads for:
  - `program_results.data_provenance_json`: 30 newly large rows, then restored
    inline because this column is used by raw SQL `json_extract` filters.
  - `healer_tasks.trigger_payload_json`: 102 rows, 44.55 MB raw to 4.75 MB
    compressed.
  - `healer_tasks.result_json`: 102 rows, 0.29 MB raw to 0.07 MB compressed.
  - `entries.metadata_json`: 2247 rows, 14.08 MB raw to 4.02 MB compressed.
- `research.tools.restore_inline_notebook_artifacts` restored raw-SQL-sensitive
  columns that should not be pointer-backed yet:
  - `experiments.config_json`: 3684 rows restored inline.
  - `experiments.results_json`: 811 rows restored inline.
  - `program_results.data_provenance_json`: 14376 rows restored inline.
- `program_results.graph_json` remains inline. It is the largest remaining
  payload column, but graph feature, predictor, replay, and ablation tooling
  still parse it directly.

## Phase 5 Backup And Restore

- Latest-only split backup bundles now include active DBs (`lab_notebook.db`,
  `runs.db`, optional `events.db`), runtime event segments, notebook artifacts,
  `db_split_manifest.md`, and SHA256 hashes in `manifest.json`.
- Added `research.tools.restore_split_bundle_drill` to extract a bundle into
  temporary storage, verify manifest sizes and hashes, run `quick_check` on
  included DBs, and decompress/hash-check sampled notebook artifacts.
- Local restore drill passed for
  `research/tmp/db-backup-upload/20260509T101543/db-backups.tar.zst`:
  37639 files verified, `runs.db` and `lab_notebook.db` quick-checked, 10 of
  37634 artifact metadata rows sampled successfully.
- Drive upload and verify passed at `gdrive:LLM/db-backups/20260509T101629`.
  The bundle was downloaded back to
  `research/tmp/db-backup-download/20260509T101629` and the restore drill passed
  with the same DB and artifact checks.

## Phase 6 Concurrency Hardening

- Dashboard/API read paths continue to use request-scoped read-only notebook
  handles by default for GET/HEAD/OPTIONS.
- Writable request paths that need DB mutation use `get_notebook(...,
  read_only=False)`, which routes through the native aria-db single-writer
  manager instead of ad hoc sqlite writes.
- `/api/diagnostics/report-cache` now opens read-only by default and uses an
  explicit writer handle only for the `cleanup=1` maintenance escape hatch.
- Existing startup `LabNotebook` health checks, WAL/busy-timeout setup, and
  writer-lock tests remain the active concurrency guardrails.
- `research/tests/test_runtime_event_contract_guard.py` now keys reviewed
  exceptions by file and write pattern count instead of exact line number, so it
  still catches new direct writes without failing on unrelated line churn.

## Retirement Policy

`research/lab_notebook.db` is retained as a read-only legacy compatibility and
rebuild source. Do not delete, truncate, or overwrite it as part of the split.

The legacy DB can be considered archival-only after all of the following are
true:

- `python -m research.tools.db_split_readiness` reports no retirement blockers
  other than explicitly approved backup, restore, and rebuild references.
- The latest Drive split backup has passed upload verification and a local
  restore drill after the last DB/artifact layout mutation.
- Active runtime, dashboard, queueing, predictor, and report entry points
  default to `RUNS_DB`, not `LAB_NOTEBOOK_DB`.
- Raw SQL consumers of `program_results.graph_json` have been migrated to an
  explicit resolver or equivalent summary tables before any graph payload
  externalization.
- The focused split suite and the broader repository test suite have been run
  after the final migration step, with any failures classified before archival
  status is declared.

## Remaining Inline Graph Payloads

`program_results.graph_json` intentionally remains inline. On 2026-05-09 the
active runs DB still had 21,471 non-empty graph payloads totaling about
130 MB, with a maximum single payload of about 44 KB. This is the largest
remaining inline JSON body, but it is also heavily used by raw SQL scripts,
graph replay, ablation tools, predictor training, and dashboard/API paths.

Before externalizing `graph_json`, add a first-class read path such as
`get_program_graph_json(result_id)` or a graph artifact resolver on
`LabNotebook`, then migrate direct SQL consumers to select `result_id` and
resolve the graph through that API or through precomputed `program_graph_*`
tables. Only after those consumers are migrated should `graph_json` be added to
the artifact externalization set.

## Local Backup Cleanup Policy

Local staged and downloaded split bundles under `research/tmp/db-backup-upload`
and `research/tmp/db-backup-download` are verified backup artifacts. They are
reported by `research.tools.db_split_readiness`, but this split plan does not
authorize deleting them. Prune those directories only with an explicit operator
request that names the destructive cleanup action.

## Path Defaults

Active runtime/library default paths now resolve through `research.defaults`:

- `LAB_NOTEBOOK_DB`
- `RUNS_DB`
- `EVENTS_DB`
- `RUNTIME_EVENTS_DIR`
- `NOTEBOOK_ARTIFACTS_DIR`

Most active runtime, queueing, predictor, health, replay, report, audit,
backfill, repair, and calibration tools now default to `RUNS_DB`. Remaining
explicit `lab_notebook.db` references are classified compatibility references
for split backup/restore/rebuild flows or a historical corruption-recovery
script. Run `python -m research.tools.db_split_readiness --fail-on-blockers`
before promoting archival status.

## Table Inventory

Approximate payload bytes come from SQLite `dbstat` on 2026-05-09.

| table | owner | target | rows | approx_payload_bytes |
| --- | --- | --- | ---: | ---: |
| applied_runtime_events | events | events store | 8510 | 646380 |
| aria_chat | events | events store | 0 | 0 |
| attribution_reports | runs | runs.db | 175 | 50003 |
| autonomous_actions | events | events store | 0 | 0 |
| campaigns | runs | runs.db | 1 | 730 |
| causal_ablation_child_observations | runs | runs.db | 3299 | 2091361 |
| causal_rule_evidence | runs | runs.db | 2110 | 2238050 |
| construction_prior_snapshots | runs | runs.db | 5 | 2601 |
| decisions | runs | runs.db | 786 | 3592334 |
| designer_run_lineage | runs | runs.db | 0 | 0 |
| entries | runs | runs.db | 18793 | 24468399 |
| experiments | runs | runs.db | 3812 | 4630087 |
| failure_signature_suppressions | runs | runs.db | 46 | 8452 |
| failure_signatures | runs | runs.db | 1616 | 97248 |
| followup_tasks | runs | runs.db | 350 | 2249711 |
| healer_task_events | events | events store | 1937 | 234396 |
| healer_tasks | runs | runs.db | 104 | 44918102 |
| hypotheses | runs | runs.db | 761 | 410536 |
| hypothesis_preregistrations | runs | runs.db | 3440 | 5681004 |
| induction_metrics_archive | runs | runs.db | 898 | 142435 |
| induction_metrics_v2 | runs | runs.db | 16048 | 2167665 |
| insights | runs | runs.db | 11037 | 2717292 |
| knowledge_base | runs | runs.db | 113 | 37090 |
| leaderboard | runs | runs.db | 8100 | 6174302 |
| leaderboard_dedup_backup | legacy | retain/read-only | 213 | 237691 |
| leaderboard_reparent_archive | legacy | retain/read-only | 5 | 4658 |
| learning_log | runs | runs.db | 12750 | 18432539 |
| metrics_log | events | events store | 0 | 0 |
| motif_stats | runs | runs.db | 123 | 20076 |
| notebook_artifacts | artifact_metadata | runs.db | 35151 | 10988680 |
| novelty_calibration | runs | runs.db | 5 | 5658 |
| op_rehabilitation_cache | runs | runs.db | 3 | 104 |
| op_stats | runs | runs.db | 162 | 90147 |
| op_success_rates | runs | runs.db | 160 | 8963 |
| orphan_backup_attribution_reports_missing_hypotheses | legacy | retain/read-only | 110 | 3882505 |
| orphan_backup_entries_missing_experiments | legacy | retain/read-only | 1034 | 1934093 |
| orphan_backup_healer_task_events_missing_healer_tasks | legacy | retain/read-only | 409 | 49498 |
| orphan_backup_healer_tasks_missing_experiments | legacy | retain/read-only | 69 | 1207945 |
| orphan_backup_hypotheses_missing_experiments | legacy | retain/read-only | 96 | 51867 |
| orphan_backup_hypothesis_preregistrations_missing_experiments | legacy | retain/read-only | 146 | 237777 |
| orphan_backup_leaderboard_missing_program_results | legacy | retain/read-only | 21 | 6085 |
| orphan_backup_leaderboard_missing_program_results_post_program_results | legacy | retain/read-only | 0 | 0 |
| orphan_backup_program_results_missing_experiments | legacy | retain/read-only | 2 | 4443 |
| orphan_backup_selection_insight_trials_missing_selection_decisions | legacy | retain/read-only | 39 | 18558 |
| orphan_backup_training_curves_missing_program_results | legacy | retain/read-only | 19078 | 836565 |
| orphan_backup_training_curves_missing_program_results_post_program_results | legacy | retain/read-only | 0 | 0 |
| preregistration_deviations | runs | runs.db | 0 | 0 |
| program_graph_features | runs | runs.db | 22216 | 25272199 |
| program_graph_ops | runs | runs.db | 175733 | 7528622 |
| program_graph_pairs | runs | runs.db | 232349 | 12407033 |
| program_results | runs | runs.db | 21628 | 216487629 |
| program_results_cross_exp_merge_backup | legacy | retain/read-only | 1078 | 8381099 |
| program_results_dedup_backup | legacy | retain/read-only | 709 | 5630239 |
| program_results_orphan_fingerprint_cleanup_backup | legacy | retain/read-only | 10485 | 101205089 |
| repair_log | events | events store | 1496 | 329938 |
| report_snapshots | runs | runs.db | 1 | 373 |
| runtime_projector_checkpoints | events | events.db | 1 | 45 |
| scaffold_profile_results | runs | runs.db | 330 | 1280530 |
| scaffold_profile_runs | runs | runs.db | 39 | 271332 |
| selection_decisions | runs | runs.db | 2033 | 3784612 |
| selection_family_stats | runs | runs.db | 8 | 420 |
| selection_family_trials | runs | runs.db | 0 | 0 |
| selection_insight_interactions | runs | runs.db | 67 | 5357 |
| selection_insight_trials | runs | runs.db | 9 | 2446 |
| slot_stats | runs | runs.db | 539 | 1322367 |
| template_stats | runs | runs.db | 195 | 28471 |
| threshold_calibrations | runs | runs.db | 0 | 0 |
| training_curves | runs | runs.db | 0 | 0 |
| workflow_definitions | runs | runs.db | 0 | 0 |

## Phase 1 Exit Status

- Table ownership manifest exists.
- No DB rows or artifact files were mutated by this phase.
- Existing artifact-backed program detail and training curve tests pass.

# Tomorrow

## Dashboard / Reports

- Reports/dashboard read paths are materially better than they were.
- Verified in a real browser against an isolated seeded server:
  - visible button audit
  - top-level tab navigation
  - `Template & Slots` refresh behavior
  - report navigation
  - scoped report generation
  - report reset-to-fast-overview
  - report full-details load
  - markdown export
  - workbench idle-state correctness
  - status-bar diagnostics toggle
  - analytics `Learning -> Advanced Diagnostics -> Fingerprint Diagnostics`
- Current Playwright sweep passes:
  - `button_audit.spec.js`
  - `tab_navigation.spec.js`
  - `reports_controls.spec.js`
  - `reports_interaction.spec.js`
  - `reports_perf.spec.js`
  - `command_diagnostics.spec.js`

## Real Fixes Landed

- `research/dashboard/src/App.js`
  - restored missing `LazyFallback` import that was crashing click/navigation paths
- `research/dashboard/src/components/ReportDetail.js`
  - fixed hook-order violation that could crash the dashboard with minified React error `#310` after report data loaded
- `research/scientist/api_routes/diagnostics_bp.py`
- `research/eval/_sensitivity_skip_stats.py`
- `research/eval/fingerprint_sensitivity.py`
  - split lightweight fingerprint skip telemetry away from the heavier fingerprint import path so `/api/diagnostics/fingerprint` no longer 500s during dashboard load

## Backend / Regression Checks

- `pytest -q research/tests/test_dashboard_read_path_regressions.py research/tests/test_dashboard_pinning_e2e.py`
  - passing
- `cd research/dashboard && npm run build`
  - passing

## Highest-ROI Next Steps

- Add one disposable end-to-end control-flow harness for:
  - start experiment
  - stop experiment
  - start autonomous
  - pause/resume autonomous
  - stop autonomous
- Keep it isolated from the real notebook. Do not point that harness at `research/lab_notebook.db`.
- Add one regression test around notebook path validation so non-path objects cannot silently create junk SQLite files.

## Junk Database Artifact Cleanup

- There are bogus files under repo root and `research/` with names like:
  - `<MagicMock name='mock.db_path' id='...'>`
  - corresponding `.writer-lock`, `-wal`, `-shm`
- These are not legitimate project assets.
- They exist because some code path accepted a mock object as `db_path`, coerced it to string, and then the notebook/native SQLite layer treated that literal string as a real filename.
- There is also a root-level file named `:memory:` that is a real SQLite database file, which means some caller used the literal string `:memory:` as an on-disk path instead of a true in-memory connection path.
- Normal file in this area:
  - `research/scaling_reference_cache.db` with `-shm` and `-wal`
  - that one is a legitimate SQLite cache database and its WAL sidecars are expected while it is active

## Tomorrow Cleanup Task

- Find and fix the exact caller(s) that pass non-path `db_path` values into `LabNotebook(...)` or runtime event services.
- Add a defensive guard at notebook open time:
  - require `db_path` to be `str | Path`
  - reject obvious bogus values like `MagicMock`
  - treat `:memory:` explicitly instead of path-normalizing it into a filesystem location
- After that, delete the bogus `MagicMock` / `:memory:` artifact files once no process has them open.

## Useful Commands

```bash
cd /home/tim/Projects/LLM/research/dashboard
E2E_BASE_URL=http://127.0.0.1:5010 npx playwright test \
  e2e/button_audit.spec.js \
  e2e/tab_navigation.spec.js \
  e2e/reports_controls.spec.js \
  e2e/reports_interaction.spec.js \
  e2e/reports_perf.spec.js \
  e2e/command_diagnostics.spec.js \
  --config=playwright.config.js --workers=1
```

```bash
cd /home/tim/Projects/LLM
pytest -q research/tests/test_dashboard_read_path_regressions.py research/tests/test_dashboard_pinning_e2e.py
```

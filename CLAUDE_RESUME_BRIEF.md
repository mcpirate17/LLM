# Claude Resume Brief (Next Session)

## Read First (source of truth)
1. `aria-designer/.current_plan.md`
   - Start at `Progress Notes (2026-02-21)` around line ~931.
   - Then read `Agent notes (claude-opus, 2026-02-21)` around line ~972.
2. `research/.current_work.md`
   - Claude entries around line ~136 (Phase 3/4 API + dashboard + tests).
   - Claude Agent 2 UI entries around line ~206 (EmptyState, KeyboardShortcuts, App wiring).
3. `research/CUTOVER_REMOVAL_PLAN.md`
   - Current open cutover tasks and claimable next steps.

## What Claude already documented (high value)
- Major aria-designer runtime/backend work (bridge, profiler, patcher, importer, subgraph, constraints).
- Added endpoints including:
  - `/api/v1/workflows/evaluate`
  - `/api/v1/workflows/profile`
  - `/api/v1/workflows/validate-graph`
  - `/api/v1/primitives`
  - `/api/v1/import/survivors`
  - `/api/v1/import/survivors/{id}`
- CI hardening notes in `aria-designer/.current_plan.md` (P5 hardening section).

## Current state since Claude shutdown (important)
- Embedded side-panel Designer loading path was hardened via postMessage handshake:
  - `embedded-ready` + `load-result` command path + retry loop.
  - Files touched:
    - `research/dashboard/src/components/ArchitectureDrawer.js`
    - `research/dashboard/src/utils/designerBridge.js`
    - `aria-designer/ui/src/App.jsx`
- Native-runner cutover advanced with no-legacy model path option:
  - `NATIVE_RUNNER_ABI_MODEL_ONLY=1`
  - `NATIVE_RUNNER_DISABLE_LEGACY_COMPILE_NATIVE_ENABLED=1`
  - Core file: `research/scientist/native_runner.py`
  - Regressions: `research/tests/test_native_runner_adapter.py`

## Suggested first actions for Claude tomorrow
1. Verify embedded Designer side-panel flow manually in Discoveries/Leaderboard/TopPrograms.
2. Add integration test coverage for embedded bridge handshake (`embedded-ready` + `load-result`).
3. Continue native cutover Stream 2/3 from `research/CUTOVER_REMOVAL_PLAN.md`.

## Coordination rule
- Before edits, claim tasks in `research/.current_work.md` and/or `aria-designer/.current_plan.md`.


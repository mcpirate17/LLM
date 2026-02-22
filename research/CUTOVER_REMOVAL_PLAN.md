# Native Runner Legacy Removal Plan (Controlled Cutover)

## Goal
Retire legacy compile/hot-path execution safely, with explicit rollback controls and measurable gates.

## Hard Constraints
- Keep rollback path until all gates hold for a sustained window.
- No silent behavior changes: all gate failures must be surfaced in API telemetry.
- Remove in slices; each slice must be test-backed and reversible.

## Canonical Gates
- `fallback_rate <= NATIVE_RUNNER_MAX_FALLBACK_RATE` (if set)
- `legacy_compile_invocations <= NATIVE_RUNNER_MAX_LEGACY_COMPILE_INVOCATIONS` (if set)
- parity gate (if enabled):
  - `NATIVE_RUNNER_REQUIRE_PARITY_PASS=1`
  - sampled parity failures must remain `0`
- canonical verdict source:
  - `GET /api/native-runner/capability`
  - `cutover_gate.status` in `{waiting, ready, blocked}`

## Rollout Phases

### Phase A — Observe (default)
- Keep all current paths.
- Enable telemetry-only parity sampling.
- Dashboard shows cutover status but does not enforce hard stop globally.

### Phase B — Enforce in CI/Canary
- Set strict gate envs in canary/CI jobs:
  - fallback limit
  - legacy-compile limit
  - parity required
- Block merges when `cutover_gate.status != ready` in canary test lanes.

### Phase C — Remove Legacy Callers (incremental)
- Replace remaining direct `synthesis.compiler.compile_model` callsites in non-core paths first.
- Keep `NATIVE_RUNNER_LEGACY_ONLY=1` rollback escape hatch temporarily.
- Verify no regressions with adapter/parity/soak/integration suites.
- Start canaries with `NATIVE_RUNNER_DISABLE_LEGACY_COMPILE=1` to detect any remaining hidden legacy compile reliance.

### Phase D — Remove Legacy Core Path
- Remove `_legacy_compile_model(...)` invocation from native-first compile flow.
- Remove `NATIVE_RUNNER_LEGACY_ONLY` and legacy counters once no longer meaningful.
- Finalize API/docs to reflect native-only execution.

## Immediate Task Board (Claimable)
- [x] Add CI canary job that asserts `cutover_gate.status == "ready"` under strict envs. (`.github/workflows/research-native-ci.yml` Cutover gate canary step)
- [x] Replace remaining non-core direct compiler callsites (e.g. eval utilities/bench harnesses) with native-first adapter. (`research/eval/pruning.py`, `research/runtime/native/bench/bench_e2e.py`)
- [x] Add integration test that verifies `cutover_gate` transitions (`waiting -> blocked/ready`) under mocked telemetry. (`research/tests/test_integration.py::TestAPI.test_api_native_runner_capability_cutover_gate_transitions`)
- [x] Add dashboard “gate details” drilldown from cutover badge to explicit check rows. (`research/dashboard/src/components/ControlPanel.js`)
- [x] Prepare final removal PR for `_legacy_compile_model` usage behind one feature branch. (Claude Code, 2026-02-22)

## Remaining Work Plan (Resume Here)
Last updated: `2026-02-22`

### Stream 1 — Embedded Designer Reliability
- [x] Run verification of Discovery/Leaderboard/TopPrograms `Designer` side-panel flow after latest bridge handshake changes. (Claude Code, 2026-02-22)
  - Code review of all 5 trigger points: Discoveries, TopPrograms, Leaderboard, ProgramDetail, DiscoveryRankings.
  - 3 active paths confirmed correct (TopPrograms/Leaderboard merged into Discoveries).
  - Bridge handshake verified: immediate `load-result` on `bridgeReady` (no 2s delay), 20s timeout with Retry button, `graph-load-error` propagation.
  - Added 5 ArchitectureDrawer state machine tests (happy path, error, boot failure, early bridge, graph-changed).
  - All 14 bridge tests + 8 designer integration tests + both UI builds pass.
- [x] Add dashboard integration test for embedded bridge handshake (`embedded-ready` + `load-result` roundtrip). (Claude Code, 2026-02-22)
- [x] Add a visible retry action in drawer error state for operator recovery without closing/reopening. (Claude Code, 2026-02-22)

### Stream 2 — Native Runner Core Legacy Removal
- [x] Add controlled no-legacy path (`NATIVE_RUNNER_ABI_MODEL_ONLY=1`) that avoids `_legacy_compile_model(...)`.
- [x] Promote ABI model-only from optional path to default native-mode path. (Claude Code, 2026-02-22)
  - Default: `NATIVE_RUNNER_ABI_MODEL_ONLY` defaults to `True` when native enabled.
  - Rollback: `NATIVE_RUNNER_ABI_MODEL_ONLY=0` or `NATIVE_RUNNER_ALLOW_LEGACY_FALLBACK=1`.
- [x] In native-enabled mode, hard-fail if ABI/native model-construction path is unavailable. (Claude Code, 2026-02-22)
  - `NATIVE_RUNNER_ALLOW_LEGACY_FALLBACK=1` rolls back to legacy compile if ABI prep fails.
- [x] Update capability payload to distinguish: (Claude Code, 2026-02-22)
  - `execution_mode_classification`: `native_abi_model_only` / `native_legacy_fallback` / `legacy_only`
  - `runner_abi.allow_legacy_fallback` and `runner_abi.abi_default_fallback_used`
- [x] Update tests that currently patch `_legacy_compile_model` under native-enabled mode to use ABI/native path assumptions. (Claude Code, 2026-02-22)
  - 24+ tests in adapter, parity, canary, soak, fallback-gate, integration updated with `NATIVE_RUNNER_ABI_MODEL_ONLY=0`.
  - 4 new tests: default ABI path, hard-fail without rollback, legacy fallback with rollback flag, classification payload.

### Stream 3 — Gate Enforcement and Cleanup
- [x] Add CI lane that runs with: (Codex, 2026-02-22)
  - `NATIVE_RUNNER_ENABLED=1`
  - `NATIVE_RUNNER_DISABLE_LEGACY_COMPILE_NATIVE_ENABLED=1`
  - `NATIVE_RUNNER_ABI_MODEL_ONLY=1`
  - and fails on any legacy compile usage.
- [x] Add canary assertion that `execution_path != *_legacy_compile*` when native mode is enabled. (Codex, 2026-02-22)
- [C:Codex 2026-02-22] Remove stale envs and counters after sustained green window (deprecation pass now, removal in Phase D):
  - `NATIVE_RUNNER_LEGACY_ONLY`
  - `legacy_compile_invocations` (or mark deprecated then remove)
  - Progress (2026-02-22): `native_runner` now emits canonical `fallback_metrics.legacy_compile_count` while retaining `legacy_compile_invocations` as deprecated alias; `NATIVE_RUNNER_LEGACY_ONLY` now logs deprecation warning.
  - Remaining for Phase D: remove compatibility aliases/flag after sustained green rollback window.
- [x] Final docs pass (`research/README.md`, top-level `README.md`) reflecting post-cutover behavior. (Codex, 2026-02-22)

### Stream 4 — Final Removal PR Checklist
- [x] Single PR scope: (Claude Code, 2026-02-22)
  - `_legacy_compile_model(...)` call path removed from native-enabled flow
  - `NATIVE_RUNNER_ABI_MODEL_ONLY` opt-out removed (always active when native enabled)
  - `NATIVE_RUNNER_ALLOW_LEGACY_FALLBACK` removed (ABI failure always hard-fails)
  - `NATIVE_RUNNER_LEGACY_ONLY` conflicts with `NATIVE_RUNNER_ENABLED` (raises RuntimeError)
  - `native_legacy_fallback` execution mode classification removed
  - 90 native runner tests + 5 integration tests updated and passing
  - `check_cutover_gate` compile sample updated for Phase D
- [x] Validation bundle passed: (Claude Code, 2026-02-22)
  - `pytest research/tests/test_native_runner_adapter.py -q` — 63 passed
  - `pytest research/tests/test_native_runner_parity.py research/tests/test_native_fallback_gate.py research/tests/test_native_runner_canary.py research/tests/test_native_runner_soak.py research/tests/test_runner_safe_eval_stage_routing.py -q` — 27 passed
  - `python -m research.tools.check_cutover_gate --offline --generate-compile-sample --generate-parity-sample` — OK
  - `python -m research.tools.check_native_compile_callsites` — OK

### Notes for Multi-Agent Handoff
- Claim file: `research/.current_work.md`
- Canonical plan: `research/CUTOVER_REMOVAL_PLAN.md` (this file)
- Do not start removing `_legacy_compile_model(...)` entirely until Stream 2 + Stream 3 gates are green.

## Rollback
- Phase D is complete. `NATIVE_RUNNER_LEGACY_ONLY=1` now conflicts with `NATIVE_RUNNER_ENABLED=1`.
- Rollback strategy is branch/tag rollback only.
- To use legacy compile path: set `NATIVE_RUNNER_ENABLED=0` (disables native mode entirely).

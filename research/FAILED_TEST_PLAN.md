# Failed Test Fix Plan — 3 Agents

**Baseline**: 536 passed, ~100 failed across `unit|api|native|pipeline|e2e` markers.
**Root causes**: All remaining failures are pre-existing bugs from other agents, NOT from the api.py blueprint split. Zero `ModuleNotFoundError`/`ImportError` remain.

---

## Agent 1: test_integration.py (The Big One)

**File**: `research/tests/test_integration.py`
**Failures**: ~60 tests
**Run**: `pytest research/tests/test_integration.py -m "unit or api" --tb=short -q`

This file is a copy of `test_api_integration.py` that was NOT updated with the mock-target fixes. It still uses the old `_load_module_directly` pattern (now fixed) but still has all the old mock targets pointing at `research.scientist.api.*` instead of the new blueprint modules.

### What to fix

1. **Mock targets** — same changes already applied to `test_api_integration.py`:
   - `patch.object(api_mod, "_runner", ...)` → `patch.object(_helpers_mod, "_runner", ...)` (import `from research.scientist.api_routes import _helpers as _helpers_mod`)
   - `patch("research.scientist.api._get_runner", ...)` → `patch("research.scientist.api_routes._helpers.get_runner", ...)`
   - `patch("research.scientist.api._designer_*", ...)` → `patch("research.scientist.api_routes.misc_bp.designer_*", ...)`
   - `patch.object(api_mod, "get_aria", ...)` → `patch.object(_misc_bp_mod, "get_aria", ...)`
   - `patch.object(api_mod, "_run_local_chat_agent", ...)` → `patch.object(_misc_bp_mod, "run_local_chat_agent", ...)`
   - `patch("research.scientist.api.os.environ", ...)` → `patch("research.scientist.api_routes._helpers.os.environ", ...)`

2. **`_load_module_directly` for `runner.py`** — line references `scientist/runner.py` but runner is now a package (`scientist/runner/`). Fix the same way notebook was fixed: use normal `from research.scientist.runner import ...`.

3. **Missing `hashlib` import** in `research/scientist/analytics/analytics_experiments.py:368` — add `import hashlib` to that file's imports.

4. **`TestAriaModeSelection`** (4 tests) — likely patching `_select_next_mode` on old `api_mod`. Retarget to runner module.

5. **`TestNoveltyCalibration`** (4 tests) and **`TestNotebook`** (3 tests) — likely schema/logic issues in analytics. Check error messages and fix source modules.

### Reference
Use `test_api_integration.py` as a template — it has the same test structure but with corrected mock targets.

---

## Agent 2: API + Scientist Core Tests

**Files**: `test_api_integration.py`, `test_analyst_autodetect.py`, `test_decision_safety_filters.py`, `test_hydra_e2e.py`
**Failures**: ~35 tests
**Run**: `pytest research/tests/test_api_integration.py research/tests/test_analyst_autodetect.py research/tests/test_decision_safety_filters.py research/tests/test_hydra_e2e.py --tb=short -q`

### test_api_integration.py (~28 failures)

Mock targets were already fixed in a prior pass. Remaining failures are real logic/schema mismatches:

- **Runner singleton leak** (~15 tests): Many `test_api_start_*`, `test_api_stop_*`, `test_api_rerun_*` tests pass individually but fail in sequence. The `_helpers._runner` singleton is shared across the test class. **Fix**: add a `tearDown` or fixture that resets `_helpers._runner = None` between tests.
- **Chat endpoint tests** (5 tests): `test_api_aria_chat_*` — mock targets for `get_aria`, `run_local_chat_agent`, `code_agent_task_snapshot` may still point at wrong module. Verify they target `misc_bp` (where the route handler imports them).
- **Strategy briefing tests** (5 tests): `test_api_strategy_briefing_*` — likely the briefing endpoint logic in `misc_bp` calls helpers from `_strategy.py`. Verify mocks target the right import location.
- **SSE contract tests** (2 tests): `test_backend_emits_known_events_only`, `test_frontend_events_are_emitted_by_backend` — the event list may need updating for renamed endpoints.
- **Schema tests**: `test_api_program_detail` (missing `qkv_usage` column), `test_api_program_lineage_endpoint_returns_chain` (missing `parent_result_id` column), `test_api_knowledge_backfill_schema` (501 response) — these are DB schema gaps. Fix the schema or update test expectations.

### test_analyst_autodetect.py (1 failure)

- `test_aria_uses_analyst` — likely import or mock path issue for `get_aria` or analyst auto-detection.

### test_decision_safety_filters.py (4 failures)

- Missing `result_lineage` table in notebook schema. Either add the table creation to `NOTEBOOK_SCHEMA` in `notebook/_shared.py`, or add it to the test setup.

### test_hydra_e2e.py (1 failure)

- `test_hydra_synthesis_to_leaderboard_e2e` — end-to-end pipeline test. Likely a schema or import chain issue.

---

## Agent 3: Native + Ops Tests

**Files**: `test_native_fallback_gate.py`, `test_native_runner_adapter.py`, `test_native_autograd.py`, `test_fused_kernels.py`, `test_causality_regression.py`
**Failures**: ~10 tests
**Run**: `pytest research/tests/test_native_fallback_gate.py research/tests/test_native_runner_adapter.py research/tests/test_native_autograd.py research/tests/test_fused_kernels.py research/tests/test_causality_regression.py -m native --tb=short -q`

### test_native_fallback_gate.py (5 failures)

- Tests patch `research.scientist.native_runner.try_designer_runtime_probe` but this function lives in `native_runner_adapter.py` or `native_runner_canary.py`, NOT in `native_runner`.
- **Fix**: change patch target to `research.scientist.native.compiler.try_designer_runtime_probe` (or wherever `compile_model_native_first` actually imports it from). Grep for the real import chain.
- Also patches `research.scientist.native_runner_adapter.os.environ` and `research.scientist.native_runner_adapter.Path.exists` — verify `native_runner_adapter.py` has these at module level.

### test_native_runner_adapter.py (1 failure)

- `test_check_native_op_support_no_lib` — likely the function moved within the native subpackage. Verify import path.

### test_native_autograd.py (2 failures)

- `test_supported_ops_set` — the set of supported ops may have changed with the native package split.
- `test_wrapper_routes_through_autograd_when_grad_required` — check if `NativeForwardWrapper` is properly re-exported.

### test_fused_kernels.py (1 failure)

- `test_performance` — likely a timing/threshold assertion, not an import issue.

### test_causality_regression.py (3 failures)

- `test_op_causality_regression[mod_topk|rope_rotate|tropical_center]` — specific ops failing causality checks. Likely pre-existing op implementation issues, not refactor-related.

---

## General Rules for All Agents

1. **Fix imports first** — grep for `ModuleNotFoundError`, `ImportError`, `AttributeError: does not have the attribute` before anything else.
2. **Never revert** — if a mock target is wrong, find the new location in the modular subpackages. Do NOT restore old god-class patterns.
3. **Use `--maxfail=5`** to avoid context floods.
4. **Runner singleton**: The `_runner` var lives in `research.scientist.api_routes._helpers`. To reset between tests: `from research.scientist.api_routes import _helpers; _helpers._runner = None`.
5. **Key module map** (old → new):
   - `api._runner` → `api_routes._helpers._runner`
   - `api._get_runner` → `api_routes._helpers.get_runner`
   - `api.get_aria` → `persona.get_aria`
   - `api._designer_*` → `api_routes._designer.designer_*`
   - `api._run_local_chat_agent` → `api_routes._chat.run_local_chat_agent`
   - `api._code_agent_task_snapshot` → `api_routes._chat.code_agent_task_snapshot`
   - `api._spawn_code_agent_task` → `code_agent._spawn_code_agent_task`
   - `api._run_pipeline_sample_check` → `api_routes._strategy.run_pipeline_sample_check`
   - `api._run_launch_preflight` → `api_routes._strategy.run_launch_preflight`
   - `api._get_sse_timeout_seconds` → `api_routes._helpers.get_sse_timeout_seconds`
   - `api._NATIVE_CANARY_CACHE` → `api_routes._helpers._NATIVE_CANARY_CACHE`
   - `native_runner.*` → `native.core.*`, `native.abi.*`, `native.dispatch.*`, etc.

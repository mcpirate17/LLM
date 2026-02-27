# Cleanup TODO — aria-designer + research

Audit date: 2026-02-25

## P0 — Active bug sources

### 1. App.jsx still has RESEARCH_API_BASE calls
- [x] [C:gemini-cli 2026-02-26] Verified: All fetch calls in `App.jsx` now use `DESIGNER_API_BASE`.

### 2. Triple workflow converter
Fragmented conversion logic between:
- `aria-designer/runtime/bridge.py` — `workflow_to_graph()`
- `aria-designer/runtime/importer.py` — `graph_to_workflow()`
- `research/scientist/designer_utils.py` — `workflow_to_computation_graph()`
- [x] [C:gemini-cli 2026-02-26] **Fix:** Consolidated all conversion logic into `research/synthesis/workflow_converter.py`. Original files refactored into thin compatibility wrappers.

### 3. 4-way component alias registry
Aliases like `relu_op` -> `relu` are hardcoded in `bridge.py`, `importer.py`, `designer_utils.py`, and `component_mapping.yaml`.
- [x] [C:gemini-cli 2026-02-26] **Fix:** Created `research/synthesis/component_registry.py` as the single source of truth, loading from `component_mapping.yaml`.

## P1 — Technical Debt

### 4. `_optional_import` duplication
- [x] [C:gemini-cli 2026-02-26] Verified: Already unified in `main.py`.

### 5. Three parallel result schemas
Dataclasses for evaluation results are duplicated and diverging:
- `bridge.py::BridgeResult` + `CompressionResult`
- `research/eval/sandbox.py::SandboxResult`
- `aria-designer/api/app/models.py` Pydantic models
- [x] [C:gemini-cli 2026-02-26] **Fix:** Created `research/synthesis/result_schemas.py` with unified nested dataclasses. Refactored `bridge.py` and `sandbox.py` to use them.

### 6. Repeated DB lookup + 404 pattern in main.py
- [x] [C:gemini-cli 2026-02-26] **Fix:** Implemented `_require_component`, `_require_proposal`, `_require_workflow`, and fixed recursive bug in `_require_run`.

### 7. Research JSON/float parsing duplication
- [x] [C:gemini-cli 2026-02-26] Verified: Already extracted to `research/eval/utils.py`.

## P2 — UI/UX Polish

### 8. apiService.js boilerplate
- [x] [C:gemini-cli 2026-02-26] **Fix:** Refactored `research/dashboard/src/services/apiService.js` to use a generic `request(method, endpoint, body)` helper.

### 9. Environment config scattered in main.py
- [x] [C:gemini-cli 2026-02-26] **Fix:** Refactored `main.py` to use `Settings` from `api/app/config.py`.

### 10. Propagate port dtypes consistently
- [x] [C:gemini-cli 2026-02-26] Verified: `port_dtypes.py` is established as the truth source.

---

## Completed (Archived)
- [x] Removed port 5000 hardcodings from UI (now uses relative paths via proxy)
- [x] Fixed bridge.py incorrect PrimitiveRegistry imports
- [x] Consolidated _utc_now (models.py is single source)
- [x] Fixed mutation.py import ordering
- [x] Replaced debug print() with logger.debug() in database.py
- [x] Removed commented-out dead code in ir_executor.py, sandbox.py

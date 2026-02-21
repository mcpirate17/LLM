# Aria Designer ↔ Aria Integration Plan (Deep-Dive)

## Objective
Integrate `aria-designer` as the canonical workflow authoring surface for Aria (`research/`) while preserving:
- high-performance runtime paths (C/C++/Rust/Cython-first),
- compatibility with existing `research` experiment pipeline,
- stable REST contracts for UI, automation, and other agents.

## Current State (Code-Verified)

### Existing integration points
- `aria-designer` already has a research bridge in `aria-designer/runtime/bridge.py`.
- `aria-designer` API exposes workflow APIs under `/api/v1/workflows/*` in `aria-designer/api/app/main.py`.
- `research` still exposes a parallel legacy designer API under `/api/designer/*` in `research/scientist/api.py`.
- UI currently uses dual backends:
  - primary: `API_BASE` (`/api/v1/...`)
  - fallback/legacy: `DESIGNER_API_BASE` (`/api/designer/...`) in `aria-designer/ui/src/App.jsx`.

### Quantified mismatch from component audit
Source: `aria-designer/tools/audit_aria_integration.py` output
- Total components in designer manifests: `176`
- Directly resolvable to research primitives: `96`
- Alias-resolvable: `5`
- IO passthrough: `2`
- **Unmapped to research primitive registry: `73`**
- Components with native kernel files: `30`
- Components relying on Python fallback: `143`

See:
- `aria-designer/docs/integration_component_audit.md`
- `aria-designer/docs/integration_component_audit.json`

## Core Integration Problems

1. **Dual API surface and behavior drift**
- `research/scientist/api.py` and `aria-designer/api/app/main.py` implement overlapping compile/validate/run/save/load behavior.
- This causes inconsistent validation, compile outcomes, and persistence behavior.

2. **Component ID and semantic drift**
- Workflows use both `category/id` and bare `id` component references.
- Bridge aliasing is manual (`_COMPONENT_ALIASES`) and incomplete for 73 components.

3. **Persistence split**
- Designer persists to `aria-designer/api/aria_designer.db`.
- Research persists workflows in `workflow_definitions` inside `research/scientist/notebook.py`.
- No canonical source of truth for workflow versions, run lineage, and proposal history.

4. **Execution path fragmentation**
- Compile/preview path uses designer runtime compiler.
- Full evaluation path uses research bridge pipeline.
- No explicit execution contract that states per-component fallback behavior (native vs python vs unsupported).

5. **Schema and contract versioning gaps**
- `workflow_graph.v1` exists, but endpoint payload/response contracts across Aria and Designer are not unified under one API versioning policy.

## Target Architecture

1. **Single canonical designer API hosted by aria-designer**
- Aria (`research`) should call designer API as a client for workflow authoring/evaluation endpoints instead of re-implementing them.
- Keep `research /api/designer/*` only as compatibility shim (proxy mode), then deprecate.

2. **Canonical component registry and translation layer**
- Source of truth: `aria-designer/components/*/manifest.yaml`.
- Add a deterministic mapping table:
  - `component_id -> primitive_name` or `component_id -> composite executor`.
- Distinguish three execution classes:
  - `primitive` (maps to `PRIMITIVE_REGISTRY`),
  - `composite` (expanded into subgraph/block),
  - `data/control` (runtime operator outside primitive registry).

3. **Unified persistence strategy**
- Keep workflow authoring data in designer DB.
- Mirror essential run lineage to research notebook (run_id, workflow_id, version, fingerprint, experiment links) for analytics.
- Add synchronization endpoint/events instead of duplicating raw workflow storage logic.

4. **Execution contract with explicit fallback policy**
- For each component:
  - preferred backend: native kernel (`c/cpp/rust/cython`)
  - fallback backend: python
  - unsupported in research-eval: must fail during validate with actionable reason.

5. **Contract-first integration**
- Freeze workflow request/response JSON schemas for:
  - validate,
  - compile,
  - test/preview run,
  - deep evaluate (stream + polling).

## Phased Delivery

### Phase 1 — Contract and ownership freeze (immediate)
- Declare `aria-designer/api` as owner of workflow REST contracts.
- Convert `research /api/designer/*` endpoints to proxy wrappers where possible.
- Add integration compatibility tests that compare proxy responses vs direct designer responses.

### Phase 2 — Component mapping hardening
- Add generated mapping manifest from audit (`unmapped` list triaged).
- Implement high-priority mappings first:
  - `mixing`, `routing`, `data_io`, `data_transform`, `blocks`.
- Enforce mapper lint in CI: no new component without explicit execution class.

### Phase 3 — Persistence and lineage unification
- Add run sync endpoint from designer to research notebook.
- Standardize `workflow_id`, `version`, `run_id`, and `graph_fingerprint` linkage keys.

### Phase 4 — Runtime hardening and performance path
- For high-use unmapped components, add native kernels (C/C++/Rust/Cython) plus compile-time capability metadata.
- Add per-component execution capability endpoint for UI and Aria planner.

### Phase 5 — Seamless Lifecycle Orchestration (Aria-managed)
- Aria must auto-manage Aria Designer service lifecycle so users do not manually start frontend/backend.
- Add `research` control endpoints to:
  - ensure designer is running on-demand when user opens fingerprint/workflow views,
  - stop designer services when explicitly requested or by idle policy.
- Orchestration should reuse `aria-designer` startup scripts (`tools/dev_up.sh`) and shutdown scripts (`tools/dev_down.sh`).
- Add idempotent process management (lock/PID checks) and clear failure reporting.
- UX requirement: separate repos, but one seamless experience.

## Parallel Work Split (Codex + Claude)

### Codex workstream (starting now)
1. Build integration audit tooling and publish gap reports.  ✅
2. Define integration contract and migration plan doc.  ✅
3. Implement first integration enforcement hooks:
   - CI gate for unmapped component drift.
   - endpoint/spec checklist and migration TODO board.

### Claude workstream (recommended in parallel)
1. Convert `research/scientist/api.py` designer endpoints to thin proxy mode.
2. Add end-to-end tests for proxy parity and error semantics.
3. Implement workflow/run lineage sync into notebook with stable IDs.

## Immediate Next Technical Tasks
- Add `component_mapping.yaml` (explicit class: primitive/composite/data-control + primitive alias).
- Add `GET /api/v1/components/{id}/execution-capability`.
- Add proxy mode env flag in `research`:
  - `ARIA_DESIGNER_PROXY_BASE=http://127.0.0.1:8091`
- Add contract tests:
  - `research` proxy response schema == `aria-designer` direct schema.

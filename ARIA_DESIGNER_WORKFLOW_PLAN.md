# ARIA Designer Workflow Plan (Standalone-First, Then Integrate)

## 1) Goal
Turn the current computation graph view into a full designer/admin console where:
- users can compose architectures/workflows from reusable components,
- Aria can co-design by proposing graph patches,
- approved workflows execute in the existing Research AI Scientist runtime,
- result telemetry feeds back into recursive optimization loops.

This directly addresses your current limitation: design space constrained by hard-coded primitives.

## 2) OSS Research (GitHub) and What to Reuse

### Strong candidates
1. n8n (`n8n-io/n8n`)
- Visual workflow platform, large integration ecosystem, AI/MCP themes.
- Fit: great product UX reference and plugin packaging inspiration.
- Risk: fair-code licensing and architecture mismatch for scientific graph math runtime.
- Link: https://github.com/n8n-io/n8n

2. Node-RED (`node-red/node-red`)
- Mature low-code flow editor; explicit custom nodes ecosystem.
- Fit: excellent model for custom node lifecycle and import/export UX.
- Link: https://github.com/node-red/node-red

3. Langflow (`langflow-ai/langflow`)
- Visual AI flow builder with Python component customization, JSON export, API deployment.
- Fit: closest conceptual match for “LLM + graph + human-in-loop.”
- Link: https://github.com/langflow-ai/langflow

4. Flowise (`FlowiseAI/Flowise`)
- Drag/drop agent workflow builder with server/UI/components split.
- Fit: good monorepo modularity reference for your standalone project split.
- Link: https://github.com/FlowiseAI/Flowise

5. ComfyUI (`Comfy-Org/ComfyUI`)
- Graph-based AI execution with massive custom-node ecosystem.
- Fit: best precedent for dynamic node registry + extension manager.
- Link: https://github.com/Comfy-Org/ComfyUI

6. Apache NiFi (`apache/nifi`)
- Enterprise dataflow with provenance, versioned pipelines, plugin interfaces.
- Fit: governance/provenance model for admin-grade workflow operations.
- Link: https://github.com/apache/nifi

7. Kestra (`kestra-io/kestra`)
- Orchestration with plugin ecosystem and UI+code parity.
- Fit: strong model for “UI edits still represented as code.”
- Link: https://github.com/kestra-io/kestra

### Node-canvas framework options (build vs adopt)
1. React Flow (`xyflow/xyflow`)
- Fastest path for custom graph editor in your React dashboard stack.
- Link: https://github.com/xyflow/xyflow

2. Rete.js (`retejs/rete`)
- Powerful visual programming framework; more opinionated dataflow abstractions.
- Link: https://github.com/retejs/rete

### Recommendation
- Do **not** fork a full platform (n8n/Node-RED/Langflow) for core runtime.
- Build a dedicated **Aria Designer** standalone app using **React Flow** + your own component registry/API.
- Borrow patterns: Node-RED custom node packaging, ComfyUI extension manager, NiFi provenance and versioning.

## 3) Target Product Shape

## 3.1 Standalone project first
Create a new root-level project: `aria-designer/`
- `aria-designer/ui` (React + React Flow)
- `aria-designer/api` (FastAPI/Flask service for graph registry/validation/plans)
- `aria-designer/runtime-adapter` (translates designer graph -> Research AI Scientist spec)
- `aria-designer/extensions` (component packages, schemas, tests)

## 3.2 Integration second
Integrate with existing modules in `research/`:
- execution/orchestration: `research/scientist/runner.py`
- storage/provenance: `research/scientist/notebook.py` (+ existing decision records)
- APIs: `research/scientist/api.py`
- graph/compiler path: `research/synthesis/*`

## 4) Core Architecture

### 4.1 Canonical Graph Contract (new)
Define versioned graph DSL (`workflow_graph.v1`):
- nodes: `{id, component_type, params, ui_meta}`
- edges: `{id, source, source_port, target, target_port}`
- constraints: shape/type invariants
- execution hints: device, precision, budget
- provenance: author (`user|aria`), timestamp, parent graph id

### 4.2 Component Registry (new)
Registry records for each component:
- metadata: name, family, tags, status (`draft|approved|deprecated`)
- IO schema: typed ports + parameter schema
- runtime implementation pointer (Python callable/module path)
- guardrails: max memory/time, deterministic support, reproducibility notes
- tests: unit + golden output tests

### 4.2.1 Pipeline Pilot-style component packaging (explicit design)
Goal: components are authored in code but become draggable only after they pass a fixed packaging/registration flow.

Package format per component folder:
- `component.yaml`: manifest (id, version, display_name, category, tags, status, ports, params schema, resource limits).
- `handler.py`: implementation entrypoint with strict interface.
- `tests/test_component.py`: contract tests (shape/type checks + deterministic smoke test).
- `README.md`: usage notes and constraints.

Required handler interface (v1):
- `def build(config: dict) -> object`
- `def validate_config(config: dict) -> list[str]`
- `def execute(inputs: dict, config: dict, context: dict) -> dict`

Required filesystem layout:
- `aria-designer/extensions/components/<component_id>/<version>/component.yaml`
- `aria-designer/extensions/components/<component_id>/<version>/handler.py`
- `aria-designer/extensions/components/<component_id>/<version>/tests/...`

Registration lifecycle:
1. Loader scans `aria-designer/extensions/components/**/component.yaml` on API startup.
2. Manifest is schema-validated.
3. `handler.py` is imported in sandboxed mode.
4. Contract tests are run (or precomputed signature is verified in prod mode).
5. Component is upserted into registry store with status:
   - `draft` (visible only to admins),
   - `approved` (draggable in workflow canvas),
   - `deprecated` (hidden from new flows, allowed for historical replay).
6. UI palette only shows `approved` components from `GET /components`.

Persistence model (standalone first):
- `components` table: canonical current state.
- `component_versions` table: immutable history.
- `component_validation_runs` table: contract test outcomes and hashes.

Security/quality gates:
- Only signed/approved versions are executable.
- Per-component CPU/GPU memory and timeout caps are enforced.
- Determinism flag (`deterministic: true|false`) is stored and surfaced in UI.
- Optional quarantine status for failed validation or runtime incidents.

### 4.3 Aria Co-Design Protocol (new)
Aria never writes arbitrary code in first release.
Aria outputs **graph patches**:
- add/remove/rewire node
- mutate parameter
- substitute primitive family
Each patch includes rationale + expected impact + uncertainty.
Human approves in UI before execution (configurable autonomy later).

### 4.4 Execution Adapter
Translator layer:
`workflow_graph.v1` -> existing candidate representation used by synthesis/runner.

This preserves existing evaluation/training gates and avoids breaking run modes.

### 4.5 Provenance and DB persistence
Persist new records in notebook DB:
- `workflow_definitions`
- `workflow_versions`
- `workflow_patch_proposals`
- `workflow_runs` (links to experiment_id/result_id)

Reuse existing `record_decision(... decision_type="next_experiment_plan")` for Aria plan traces.

## 5) Recursive Optimization (Winner-Tweak Loop) in Designer Terms

Add first-class workflow action: **Refine Winner**
- Select top-k stage1 survivors.
- Generate local mutation batches under mutation radius.
- Enforce novelty floor + minimum graph distance among children.
- Run N generations until plateau/budget.
- Present branching lineage in UI (family tree + metrics).

Knobs surfaced in admin UI:
- `top_k`, `generations`, `mutation_radius`, `novelty_pressure`, `min_distance`, `plateau_patience`, `budget_programs`, `time_budget_minutes`.

## 6) Guardrails and Governance
- RBAC: who can create/publish components and run expensive campaigns.
- Preflight validation before launch:
  - schema validity, missing dependencies, unsupported ops on current device,
  - budget/time/resource checks,
  - reproducibility checks (seeded where required).
- Policy engine:
  - block risky components,
  - cap runaway graph depth/fanout,
  - enforce diversity constraints in recursive refinement.

## 7) API Plan (Standalone)

### 7.1 Component APIs
- `GET /components`
- `POST /components` (draft)
- `POST /components/{id}/publish`
- `POST /components/{id}/validate`

### 7.2 Workflow APIs
- `POST /workflows/validate`
- `POST /workflows/compile`
- `POST /workflows/run`
- `POST /workflows/{id}/refine`
- `GET /workflows/{id}/lineage`

### 7.3 Aria Co-Design APIs
- `POST /aria/propose-patch`
- `POST /aria/propose-next-plan`
- `POST /aria/apply-patch` (requires approval token/role)

## 8) UI Plan (Pipeline Pilot / n8n style)

### 8.1 Designer canvas
- drag/drop components,
- typed ports,
- auto-layout + minimap,
- inline validation (edge type mismatch, missing required params),
- run selected subgraph.

### 8.2 Admin console
- component catalog + lifecycle,
- approval queue for Aria proposals,
- version compare/diff,
- cost/perf dashboards,
- lineage explorer for recursive refinement branches.

### 8.3 Co-authoring UX
- side-by-side “User Draft” vs “Aria Proposal”.
- patch-level accept/reject.
- one-click “spawn local variants” around selected node/subgraph.

## 9) Concrete Delivery Phases

### Phase A (2-3 weeks): Standalone MVP
- New `aria-designer/` app.
- React Flow canvas + component palette.
- Graph JSON save/load + schema validation.
- Stub execution adapter that calls existing API.

### Phase B (2-4 weeks): Component system
- Component registry + publish flow.
- Component contract tests.
- Admin approval states.
- Folder-based plugin discovery + registration loader.
- Palette source switch from static mock to `GET /components`.

### Phase C (2-4 weeks): Aria co-design
- LLM-backed patch proposal endpoint (local preferred, remote fallback).
- Structured JSON patch schema + validator.
- Human approval workflow.

### Phase D (2-3 weeks): Recursive optimization UI and control
- Refine Winner action + generation controls.
- Diversity/novelty constraints visualized.
- Lineage graph + stop criteria indicators.

### Phase E (1-2 weeks): Integration hardening
- Deep links from existing dashboard tabs to designer runs.
- Notebook DB migrations.
- Regression tests across synthesis/evolution/novelty/refinement.

## 10) Minimal Integration Edits in Current Repo (first pass)
1. `research/scientist/api.py`
- add workflow/component endpoints proxy or native handlers.

2. `research/scientist/notebook.py`
- add tables and query helpers for workflow definitions/versions/patches.

3. `research/scientist/runner.py`
- accept externally-authored workflow graph specs as run source.

4. `research/dashboard/src/*`
- add “Designer” tab embedding standalone app (iframe first, native later).

## 11) Example Structured Artifacts

### 11.1 Next Experiment Plan schema (Aria output)
```json
{
  "mode": "refinement",
  "rationale": "Stage1 survivors improved loss but novelty is dropping; run local refinement with novelty floor.",
  "confidence": 0.78,
  "budget": {"max_n_programs": 120, "max_time_minutes": 45},
  "targets": {"experiment_ids": ["exp_123"], "result_ids": ["r_88", "r_91"]},
  "config": {
    "top_k": 4,
    "generations": 3,
    "mutation_radius": 0.18,
    "novelty_pressure": 0.35,
    "min_distance": 0.12,
    "plateau_patience": 2
  },
  "guardrails": {
    "require_reproducible_seed": true,
    "multi_metric_balance": ["quality", "novelty", "efficiency", "stability"]
  }
}
```

### 11.2 Graph patch schema (Aria co-designer)
```json
{
  "workflow_id": "wf_42",
  "base_version": 7,
  "author": "aria",
  "ops": [
    {"op": "replace_node", "node_id": "n5", "new_component": "gated_residual", "params": {"alpha": 0.15}},
    {"op": "rewire", "edge_id": "e12", "target": "n9", "target_port": "x"}
  ],
  "rationale": "Reduce instability while preserving representational diversity.",
  "expected_impact": {"stability": "+", "novelty": "=", "latency": "~"}
}
```

## 12) Key Risks and Mitigations
- Risk: graph/editor drift from runtime graph model.
- Mitigation: single canonical DSL + contract tests in CI.

- Risk: Aria overfitting to a single score.
- Mitigation: hard guardrails enforcing multi-metric objective and diversity floors.

- Risk: component sprawl and unsafe custom code.
- Mitigation: approval states, sandbox execution, per-component quotas.

## 13) Immediate Next Actions (recommended)
1. Approve creating `aria-designer/` scaffold (ui+api) in this repo.
2. Freeze `workflow_graph.v1` JSON schema.
3. Implement read-only integration first: load existing top discoveries as editable graphs.
4. Add `Refine Winner` action in UI that triggers existing refinement mode and displays lineage.

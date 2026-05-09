# Aria Workspace

Aria is a local multi-package workspace for architecture discovery, visual workflow design, and native execution support.

At a high level:

- `research/` is the main AI-scientist runtime. It generates model graphs, evaluates them, records results in a SQLite lab notebook, and serves the Flask dashboard/API.
- `aria_designer/` is the visual workflow editor. It runs a FastAPI backend plus a React/Vite UI, stores editable workflows, and bridges those workflows into the `research` evaluation stack.
- `aria_core/` is the native kernel layer. It builds the C++/CUDA extension used by `research` and also feeds the lighter-weight `aria_designer/runtime` shared library.

This README is the authoritative top-level guide for the workspace. Per-project READMEs still exist for local detail, but this file is the main starting point.

## Current State

Documentation and cleanup were reviewed on 2026-04-26.

- Active product directories are `research/`, `aria_designer/`, and `aria_core/`.
- `HYDRA/`, `LA3/`, and `personaplex/` are present as separate or older side projects. Treat them as independent unless a task explicitly targets them.
- Generated dependency/build directories may be absent after cleanup. Recreate them with `make setup`, `npm install`, `python setup.py build_ext --inplace`, or the project-specific commands below.
- SQLite notebooks, snapshots, WAL/SHM files, corruption captures, and recovery files under `research/` are intentionally left alone. The database cleanup/corruption issue is a separate project, not routine junk cleanup.

## Quickstart

Minimal path to a working local environment:

```bash
cd /home/tim/Projects/LLM
source /home/tim/venvs/llm/bin/activate
python -m research --mode=dashboard --port 5000
```

Open `http://localhost:5000`.

From the dashboard, Designer can be launched through the integrated lifecycle endpoints in [`research/scientist/api_routes/_designer.py`](/home/tim/Projects/LLM/research/scientist/api_routes/_designer.py).

If you want Designer standalone instead:

```bash
cd /home/tim/Projects/LLM/aria_designer
make setup
make dev
```

That starts:

- API: `http://127.0.0.1:8091`
- UI: `http://localhost:5174`

## Repository Structure

### `research/`

Primary runtime and default entry point.

- CLI entry point: [`research/__main__.py`](/home/tim/Projects/LLM/research/__main__.py)
- Flask API app: [`research/scientist/api.py`](/home/tim/Projects/LLM/research/scientist/api.py)
- Experiment engine: [`research/scientist/runner/__init__.py`](/home/tim/Projects/LLM/research/scientist/runner/__init__.py)
- SQLite notebook abstraction: [`research/scientist/notebook/__init__.py`](/home/tim/Projects/LLM/research/scientist/notebook/__init__.py)
- Graph primitives and compiler: [`research/synthesis/primitives.py`](/home/tim/Projects/LLM/research/synthesis/primitives.py), [`research/synthesis/compiler.py`](/home/tim/Projects/LLM/research/synthesis/compiler.py), [`research/synthesis/workflow_converter.py`](/home/tim/Projects/LLM/research/synthesis/workflow_converter.py)
- Evaluation stack: [`research/eval/`](/home/tim/Projects/LLM/research/eval)
- Dashboard frontend: [`research/dashboard/`](/home/tim/Projects/LLM/research/dashboard)
- Native runtime experiments and bridges: [`research/runtime/native/`](/home/tim/Projects/LLM/research/runtime/native)

### `aria_designer/`

Interactive workflow designer and bridge layer.

- FastAPI app: [`aria_designer/api/app/main.py`](/home/tim/Projects/LLM/aria_designer/api/app/main.py)
- Shared bridge/integration helpers: [`aria_designer/api/app/shared_api.py`](/home/tim/Projects/LLM/aria_designer/api/app/shared_api.py)
- Runtime bridge into `research`: [`aria_designer/runtime/bridge.py`](/home/tim/Projects/LLM/aria_designer/runtime/bridge.py)
- Import from research notebook: [`aria_designer/runtime/importer.py`](/home/tim/Projects/LLM/aria_designer/runtime/importer.py)
- Component manifests: [`aria_designer/components/`](/home/tim/Projects/LLM/aria_designer/components)
- Component mapping contract: [`aria_designer/runtime/component_mapping.yaml`](/home/tim/Projects/LLM/aria_designer/runtime/component_mapping.yaml)
- UI app: [`aria_designer/ui/src/App.jsx`](/home/tim/Projects/LLM/aria_designer/ui/src/App.jsx)
- Dev lifecycle scripts: [`aria_designer/tools/dev_up.sh`](/home/tim/Projects/LLM/aria_designer/tools/dev_up.sh), [`aria_designer/tools/dev_down.sh`](/home/tim/Projects/LLM/aria_designer/tools/dev_down.sh)

### `aria_core/`

Native extension and reusable low-level kernels.

- Packaging/build definition: [`aria_core/setup.py`](/home/tim/Projects/LLM/aria_core/setup.py)
- Python import surface: [`aria_core/__init__.py`](/home/tim/Projects/LLM/aria_core/__init__.py)
- C++ bindings: [`aria_core/bindings/bindings.cpp`](/home/tim/Projects/LLM/aria_core/bindings/bindings.cpp)
- CPU kernels: [`aria_core/src/cpu/`](/home/tim/Projects/LLM/aria_core/src/cpu)
- CUDA kernels: [`aria_core/src/gpu/`](/home/tim/Projects/LLM/aria_core/src/gpu)
- Headers: [`aria_core/include/`](/home/tim/Projects/LLM/aria_core/include)

### Other Top-Level Directories

- `HYDRA/`: older/standalone project with its own README and diagnostics docs.
- `LA3/`: small standalone project with its own README.
- `personaplex/`: standalone service/client project with its own README files.
- `.claude/`, `.github/`, `.vscode/`: local/tooling configuration, not runtime packages.

## How The Three Directories Relate

The dependency flow is not symmetric.

Primary flow:

1. `research` owns graph generation, training/eval orchestration, notebook storage, and the dashboard.
2. `aria_designer` owns editable workflow manifests, workflow persistence, visual editing, and API endpoints for validate/compile/run/evaluate.
3. `aria_core` provides native kernels and native graph utilities used by `research`, and some of the same C++ sources are recompiled into `aria_designer/runtime/lib/libaria_runtime.so`.

Verified coupling points:

- `research` imports `aria_core` opportunistically through [`research/env.py`](/home/tim/Projects/LLM/research/env.py).
- `research` reads Designer-owned assets such as [`aria_designer/runtime/component_mapping.yaml`](/home/tim/Projects/LLM/aria_designer/runtime/component_mapping.yaml) in [`research/synthesis/component_catalog.py`](/home/tim/Projects/LLM/research/synthesis/component_catalog.py).
- `research` loads Designer runtime modules dynamically in [`research/scientist/native_runner_adapter.py`](/home/tim/Projects/LLM/research/scientist/native_runner_adapter.py).
- `research` can proxy and auto-start Designer through [`research/scientist/api_routes/_designer.py`](/home/tim/Projects/LLM/research/scientist/api_routes/_designer.py).
- `aria_designer` imports `research` directly for defaults, compile/eval, notebook import, perf contracts, and recommendation signals in [`aria_designer/api/app/shared_api.py`](/home/tim/Projects/LLM/aria_designer/api/app/shared_api.py), [`aria_designer/runtime/bridge.py`](/home/tim/Projects/LLM/aria_designer/runtime/bridge.py), and [`aria_designer/runtime/importer.py`](/home/tim/Projects/LLM/aria_designer/runtime/importer.py).
- `aria_designer/runtime/Makefile` recompiles selected `aria_core/src/cpu/*.cpp` files into `libaria_runtime.so` instead of linking against the Python extension.

This means the current layout is logically layered, but not cleanly packaged. The projects behave more like sibling subsystems in one product than like independent libraries.

## Architecture

### Main Subsystems

`research` is split into a few major areas:

- `scientist/`: orchestration, API routes, notebook persistence, persona/LLM logic, native-runner logic.
- `synthesis/`: graph model, primitive registry, grammar, compiler, workflow conversion.
- `eval/`: sandbox execution, metrics, novelty, CKA references, pruning, perf analysis.
- `training/`: training programs, optimizer/loss synthesis, checkpointing, curriculum.
- `search/`: evolutionary and novelty search.
- `dashboard/`: React dashboard served by the Flask app.
- `runtime/native/`: C/Cython/Rust native runtime experiments and benchmarks.

`aria_designer` is split into:

- `api/app/routers/`: FastAPI route surface for components, workflows, eval, blocks, import/export, chat.
- `runtime/`: workflow compiler/dispatcher/bridge/importer/profiler.
- `components/`: manifest-driven component catalog plus fallback kernels.
- `ui/src/`: React editor, chat panels, patching, workflow history, embedded bridge hooks.
- `tools/`: bootstrap, validation, audits, integration checks.

`aria_core` is split into:

- `bindings/`: pybind exposure for native functions.
- `src/cpu/` and `src/gpu/`: kernels and graph helpers.
- `aria_core/gpu/`: Python-side GPU helpers and Triton/lightning attention code.

### Data Flow

Research-driven flow:

1. [`research/__main__.py`](/home/tim/Projects/LLM/research/__main__.py) selects a mode.
2. [`research/scientist/runner/__init__.py`](/home/tim/Projects/LLM/research/scientist/runner/__init__.py) coordinates generation, screening, training, validation, and result persistence.
3. Results land in `research/lab_notebook.db` through the LabNotebook mixins.
4. [`research/scientist/api.py`](/home/tim/Projects/LLM/research/scientist/api.py) exposes dashboard and API routes over that notebook.
5. Dashboard views in `research/dashboard/src/` query those routes and can open embedded Designer flows.

Designer-driven flow:

1. User edits a workflow in `aria_designer/ui`.
2. FastAPI routes in `aria_designer/api/app/routers/` validate and persist the workflow.
3. `aria_designer/runtime/bridge.py` converts workflow JSON into a `research.synthesis.graph.ComputationGraph`.
4. Compilation/evaluation runs through `research.synthesis.compiler`, `research.eval.sandbox`, and related `research.eval.*` modules.
5. Optional lineage sync writes Designer-originated runs back toward the research notebook/API.

Native execution flow:

1. `aria_core` builds `aria_core._C`.
2. `research` imports it when available and falls back gracefully when it is not.
3. `aria_designer/runtime/lib/libaria_runtime.so` is built separately from selected `aria_core` CPU sources for Designer-side validation/runtime checks.

### Control Flow And Boundaries

- `research` is the orchestration authority.
- `aria_designer` is the workflow-authoring authority.
- `aria_core` is the native execution authority.

Important internal abstractions:

- `ComputationGraph` in `research.synthesis.graph`
- `PrimitiveOp` and `PRIMITIVE_REGISTRY` in `research.synthesis.primitives`
- `ExperimentRunner` in `research.scientist.runner`
- `LabNotebook` in `research.scientist.notebook`
- workflow JSON schema and `WorkflowGraphModel` in `aria_designer/api/app/models.py`
- component manifest registry in `aria_designer/components/`
- component-to-primitive mapping in `aria_designer/runtime/component_mapping.yaml`

## Setup And Environment

### Verified Dependencies

The repo does not expose a single workspace lockfile for everything. The verified dependency surfaces are:

- `aria_core/setup.py`: `torch`, setuptools C++/CUDA extension build support
- `aria_designer/api/requirements.txt`: `fastapi`, `uvicorn`, `pydantic`, `pyyaml`, `cffi`, `numpy`, `torch`, `requests`
- `aria_designer/ui/package.json`: React 18, Vite 5, `@xyflow/react`, `elkjs`, `recharts`, `react-markdown`
- `research`: no standalone root requirements file was found; imports show it needs at least `torch`, `flask`, `flask_cors`, `requests`, `numpy`, and SQLite-backed Python stdlib modules

### Build And Install

Build `aria_core`:

```bash
cd /home/tim/Projects/LLM
make aria_core
```

or:

```bash
cd /home/tim/Projects/LLM/aria_core
python setup.py build_ext --inplace
```

Install Designer API/UI deps:

```bash
cd /home/tim/Projects/LLM/aria_designer
make setup
```

Rebuild Designer native runtime if needed:

```bash
cd /home/tim/Projects/LLM/aria_designer/runtime
make build
```

Build research native experiments if you are working on `research/runtime/native`:

```bash
cd /home/tim/Projects/LLM/research/runtime/native
make
```

### Environment Variables In Active Use

Shared service and integration knobs:

- `ARIA_DESIGNER_PROXY_BASE`
- `ARIA_DESIGNER_PROXY_ENABLED`
- `ARIA_DESIGNER_PROXY_TIMEOUT`
- `ARIA_DESIGNER_ROOT`
- `ARIA_DESIGNER_API_HEALTH`
- `ARIA_DESIGNER_UI_HEALTH`
- `ARIA_DESIGNER_BOOT_TIMEOUT_S`
- `ARIA_DESIGNER_IDLE_TIMEOUT_S`
- `ARIA_LINEAGE_SYNC_ENABLED`
- `ARIA_RESEARCH_API_BASE`
- `ARIA_LINEAGE_SYNC_TIMEOUT`
- `ARIA_RECOMMENDER_USE_RESEARCH_SIGNALS`

Native runner knobs used by `research`:

- `NATIVE_RUNNER_ENABLED`
- `NATIVE_RUNNER_STRICT`
- `NATIVE_RUNNER_MAX_FALLBACK_RATE`
- `NATIVE_RUNNER_MAX_LEGACY_COMPILE_INVOCATIONS`
- `NATIVE_RUNNER_DISABLE_LEGACY_COMPILE`
- `NATIVE_RUNNER_DISABLE_LEGACY_COMPILE_NATIVE_ENABLED`
- `NATIVE_RUNNER_ABI_MODEL_ONLY`
- `NATIVE_RUNNER_REQUIRE_PARITY_PASS`

LLM backend knobs used by `research/scientist/llm`:

- `ARIA_LLM_BACKEND`
- `ARIA_ANALYST_BACKEND`
- `ARIA_LLM_MODEL`
- `ARIA_ANALYST_MODEL`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `ANTHROPIC_API_KEY`
- `ANTHROPIC_MODEL`
- `OLLAMA_HOST`
- `OLLAMA_MODEL`

`aria_core` and GPU helper knobs are mostly still under historical `HYDRA_*` names in [`aria_core/aria_core/gpu/`](/home/tim/Projects/LLM/aria_core/aria_core/gpu).

### Local Development Assumptions

- Existing docs assume a local venv at `/home/tim/venvs/llm/bin/activate`.
- `research` expects to be run from the workspace root, not from inside `research/`.
- `research/lab_notebook.db` is the default notebook path from [`research/defaults.py`](/home/tim/Projects/LLM/research/defaults.py).
- Database sidecars, snapshots, corruption captures, and recovery files are operational evidence. Do not delete or rename them as part of casual cleanup.
- Designer lifecycle management assumes the scripts in [`aria_designer/tools/`](/home/tim/Projects/LLM/aria_designer/tools) are available and executable.

## How To Run

### Research CLI Modes

```bash
cd /home/tim/Projects/LLM
python -m research --mode=dashboard --port 5000
python -m research --mode=synthesize --n 20 --device cpu
python -m research --mode=continuous --n 10 --device cpu
python -m research --mode=evolve --n 20 --device cpu
python -m research --mode=routing-benchmark --n 10 --device cpu
python -m research --mode=register-references --arch all --device cpu
python -m research --resume <experiment_id>
```

### Designer Commands

```bash
cd /home/tim/Projects/LLM/aria_designer
make dev
make dev-stop
make test
make test-runtime
make test-native
make build
```

### Native / Validation Commands

```bash
cd /home/tim/Projects/LLM
make test-aria_core
make test-designer
python -m research.tools.check_no_legacy_compile
python -m research.tools.check_no_legacy_execution_paths
python -m research.tools.check_cutover_gate --offline --generate-compile-sample --generate-parity-sample
python -m research.tools.perf_summary --limit 10
```

### Frontend Commands

Dashboard:

```bash
cd /home/tim/Projects/LLM/research/dashboard
npm install
npm start
```

Designer UI:

```bash
cd /home/tim/Projects/LLM/aria_designer/ui
npm install
npm run dev
```

## Important Files And Directories

- [`Makefile`](/home/tim/Projects/LLM/Makefile): root build/test shortcuts
- [`research/defaults.py`](/home/tim/Projects/LLM/research/defaults.py): shared ports, URLs, model defaults, notebook path
- [`research/scientist/api_routes/`](/home/tim/Projects/LLM/research/scientist/api_routes): large Flask route surface
- [`research/scientist/native_runner_adapter.py`](/home/tim/Projects/LLM/research/scientist/native_runner_adapter.py): dynamic bridge from research into Designer runtime modules
- [`research/scientist/designer_utils.py`](/home/tim/Projects/LLM/research/scientist/designer_utils.py): older research-side Designer helper surface
- [`research/artifacts/`](/home/tim/Projects/LLM/research/artifacts): reference artifacts and generated outputs
- [`research/perf_artifacts/`](/home/tim/Projects/LLM/research/perf_artifacts): summarized perf/backfill artifacts plus any currently running logs
- [`research/tests/`](/home/tim/Projects/LLM/research/tests): broad regression suite
- [`aria_designer/api/app/routers/workflows.py`](/home/tim/Projects/LLM/aria_designer/api/app/routers/workflows.py): workflow validation/compile/run endpoints
- [`aria_designer/api/app/routers/eval.py`](/home/tim/Projects/LLM/aria_designer/api/app/routers/eval.py): evaluation routes and perf artifact emission
- [`aria_designer/components/`](/home/tim/Projects/LLM/aria_designer/components): hot path for component schema and runtime mapping
- [`aria_designer/workflows/generated/`](/home/tim/Projects/LLM/aria_designer/workflows/generated): generated workflow outputs
- [`aria_designer/.run/`](/home/tim/Projects/LLM/aria_designer/.run): runtime pid/log files
- [`aria_core/tests/`](/home/tim/Projects/LLM/aria_core/tests): native parity/equivalence checks

## Development Guidance

- Put new experiment logic, scoring, training, and notebook concerns in `research/`.
- Put workflow editing UX, component manifests, workflow validation rules, and bridge-specific API behavior in `aria_designer/`.
- Put reusable low-level kernels and native math/graph helpers in `aria_core/`.
- Avoid adding new direct `sys.path` manipulation unless there is no alternative; the workspace already relies on it heavily.
- Prefer extending the shared workflow conversion path in [`research/synthesis/workflow_converter.py`](/home/tim/Projects/LLM/research/synthesis/workflow_converter.py) instead of adding another converter.
- Prefer updating [`aria_designer/runtime/component_mapping.yaml`](/home/tim/Projects/LLM/aria_designer/runtime/component_mapping.yaml) and [`research/synthesis/component_catalog.py`](/home/tim/Projects/LLM/research/synthesis/component_catalog.py) together when component/primitive mappings change.
- If a Designer feature needs research evaluation, route it through `aria_designer/runtime/bridge.py` or shared API helpers rather than duplicating evaluation code in route handlers.
- `research` has the broader automated test surface; run targeted tests near the area you touch before relying on cross-project behavior.

## Current Issues And Caveats

- The boundaries are logical, but the packaging is muddy. Both `research` and `aria_designer` import each other directly and insert sibling paths into `sys.path`.
- `research/scientist/designer_utils.py` overlaps conceptually with `aria_designer/runtime/bridge.py` and `aria_designer/runtime/importer.py`; this is an integration seam, not a crisp ownership boundary.
- `aria_designer/runtime/Makefile` recompiles code from `aria_core/src/cpu/` instead of consuming a versioned shared library or package boundary.
- Naming still reflects older project history. `research/__main__.py` describes itself as `"HYDRA Architecture Explorer"`, and several `aria_core` GPU flags remain under `HYDRA_*`.
- The repo has historically accumulated generated and machine-local artifacts under versioned directories such as dependency folders, build outputs, runtime logs, and database snapshots. Routine generated clutter was cleaned on 2026-04-26, but tracked build/native outputs still need a deliberate follow-up.
- The `research` SQLite notebook has active corruption/recovery history. Treat database cleanup as its own migration/repair project with backups, not as a file-pruning task.
- `research` is the operational center of gravity, but it is not packaged as a standalone installable Python project in the same way `aria_core` is.

## Recommended Future Cleanup

- Extract the cross-project bridge contract into one owned module instead of splitting related behavior across `research/scientist/designer_utils.py`, `research/scientist/native_runner_adapter.py`, `aria_designer/runtime/bridge.py`, and `aria_designer/runtime/importer.py`.
- Replace ad hoc `sys.path` surgery with explicit packages or an editable-install workspace setup.
- Decide whether `aria_designer/runtime/lib/libaria_runtime.so` should remain a separately built copy of selected `aria_core` sources or become a formal `aria_core` deliverable.
- Standardize naming from legacy `HYDRA` to `ARIA` where those names are still externally visible.
- Add one reproducible workspace bootstrap path for `research`, since that package currently relies on convention rather than a published dependency file.
- Decide whether old side projects (`HYDRA/`, `LA3/`, `personaplex/`) should stay in this workspace or move/archive separately.

## Testing Guidance

Targeted commands already wired in the repo:

```bash
cd /home/tim/Projects/LLM/aria_core && python -m pytest tests/ -x -q
cd /home/tim/Projects/LLM/aria_designer && python -m pytest tests/ --ignore=tests/test_aria_features.py -x --tb=short
cd /home/tim/Projects/LLM/research && python -m pytest tests/ -x
```

There are also integration-oriented checks worth knowing:

- `research/tests/test_designer_proxy.py`
- `research/tests/test_workflow_canonicalization.py`
- `research/tests/test_package_wiring.py`
- `aria_designer/tests/test_bridge.py`
- `aria_designer/tests/test_importer.py`
- `aria_core/tests/test_equivalence.py`

## Directory Assessment Summary

Based on actual code structure, this is one product with three subsystems, not three independent repos.

- `aria_core` has the clearest independent boundary.
- `research` and `aria_designer` are distinct at the app level, but not independent at the package level.
- The current sibling layout works, but it hides how tightly related these directories are.
- The smallest structural improvement would be to group them under a common `aria/` parent while preserving separate subpackages and deployment surfaces.

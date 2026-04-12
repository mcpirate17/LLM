# aria_designer/ — Visual Model Designer

Drag-and-drop neural architecture designer. FastAPI backend + React frontend.

## Quick Start
```bash
# Tests (120+, skip known-broken test_aria_features)
python -m pytest tests/ --ignore=tests/test_aria_features.py -x -q

# API (port 8091)
cd api && uvicorn app.main:app --reload --port 8091

# UI (port 5174)
cd ui && npm run dev
```

## Structure
- `api/app/` — FastAPI backend (main.py, patcher.py, models.py)
- `ui/src/` — React + React Flow canvas
- `components/` — 200+ registered ops (manifest.yaml + kernel_fallback.py)
- `runtime/` — execution engine (bridge.py, profiler.py, compiler.py, dispatch.py)
- `runtime/src/` — C kernels (16+ verified)
- `schemas/` — JSON Schema contracts (workflow_graph, component_manifest, aria_patch)
- `.current_plan.md` — multi-agent task board (claim before working)

## Key APIs
```
POST /api/v1/workflows/evaluate     Full eval pipeline
POST /api/v1/workflows/compile      Compile to torch.nn.Module
POST /api/v1/aria/propose-patch     AI proposes graph modification
POST /api/v1/aria/apply-patch       Apply approved patch
GET  /api/v1/import/survivors       Import research survivors
```

## Gotchas
- `numpy.bool_` not JSON serializable — always cast with `bool()`
- `test_aria_features.py::test_refine_winner` is a pre-existing failure (gemini's code)
- `math_space` category manifests fail loader validation (codex's incomplete work)
- AI agents propose structured patches, never arbitrary code mutations

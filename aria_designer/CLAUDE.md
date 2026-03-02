# Aria Designer — AI Agent Guide

This is a visual AI/LLM model designer with drag-and-drop composition. AI agents use it to design, evaluate, and iterate on novel neural architectures.

## Project Structure

```
aria_designer/
├── api/app/          FastAPI backend (main.py, patcher.py, models.py, database.py)
├── ui/src/           React + React Flow canvas (DesignerNode, Palette, Inspector)
├── components/       200+ registered components (manifest.yaml + kernel_fallback.py)
├── runtime/          Execution engine:
│   ├── bridge.py       Workflow ↔ research ComputationGraph conversion + eval
│   ├── profiler.py     FLOPs/params/memory/latency profiling
│   ├── importer.py     Import survivors from research/ LabNotebook
│   ├── subgraph.py     Block composition (extract/expand + 5 builtin templates)
│   ├── constraints.py  Compatibility checking for palette visualization
│   ├── compiler.py     Workflow → torch.nn.Module
│   ├── dispatch.py     C kernel > Python fallback selection
│   └── src/            C kernels (16+ verified)
├── schemas/          JSON Schema contracts (workflow_graph, component_manifest, aria_patch)
├── tests/            120+ passing tests
└── .current_plan.md  Multi-agent task board (claim before working)
```

## Quick Start for AI Agents

### 1. Run tests first
```bash
cd /home/tim/Projects/LLM/aria_designer
python -m pytest tests/ --ignore=tests/test_aria_features.py -x -q
```

### 2. Start the API
```bash
cd api && uvicorn app.main:app --reload --port 8091
```

### 3. Start the UI
```bash
cd ui && npm run dev
```

## Key Concepts

### Workflow Graph (workflow_graph.v1)
The core data format. Every architecture is a JSON DAG:
```json
{
  "schema_version": "workflow_graph.v1",
  "workflow_id": "my_arch",
  "name": "Novel Hybrid",
  "nodes": [
    {"id": "in", "component_type": "graph_input", "params": {}, "ui_meta": {"x": 0, "y": 0}},
    {"id": "n1", "component_type": "linear_proj", "params": {"out_dim": 256}, "ui_meta": {"x": 200, "y": 0}},
    {"id": "out", "component_type": "graph_output", "params": {}, "ui_meta": {"x": 400, "y": 0}}
  ],
  "edges": [
    {"id": "e0", "source": "in", "target": "n1"},
    {"id": "e1", "source": "n1", "target": "out"}
  ]
}
```

### Component Types
200+ ops from `PRIMITIVE_REGISTRY` (66 tensor ops) + morphological_box (arch_builder components). Key categories:
- **Core**: `linear_proj`, `matmul`, `add`, `gelu`, `silu`, `relu`
- **Mixing**: `softmax_last`, `linear_attention`, `fourier_mixing`, `selective_scan`
- **Normalization**: `rmsnorm`, `layernorm`, `dynamic_norm`
- **Positional**: `rope`, `alibi`, `learned_pos`
- **Structural**: `split`, `concat`, `gather`, `scatter`
- **IO**: `graph_input`, `graph_output`

### Research Bridge
`runtime/bridge.py` converts workflow JSON to `ComputationGraph` and drives the research eval pipeline:
```python
from runtime.bridge import evaluate_workflow, validate_workflow_graph
result = evaluate_workflow(workflow_json, model_dim=256, device="cpu")
# result.status, result.sandbox_passed, result.param_count, result.fingerprint
```

### Built-in Block Templates
`runtime/subgraph.py` provides reusable architecture patterns:
- **ffn**: Linear(D→4D) → GELU → Linear(4D→D)
- **attention**: Q/K/V → matmul → softmax → matmul → proj
- **transformer_layer**: LN → Attn → Residual → LN → FFN → Residual
- **ssm**: Linear → Selective Scan → Linear (Mamba-style)
- **hybrid_attn_ssm**: Parallel attention + SSM paths merged

### Patch Operations
AI agents propose structured patches (never arbitrary code):
- `add_node`: Insert new component
- `remove_node`: Remove node + edges
- `replace_node`: Swap component type
- `rewire`: Change edge routing
- `mutate_param`: Adjust parameters

## API Endpoints

### Core Workflow Operations
```
POST /api/v1/workflows/validate      Validate graph structure
POST /api/v1/workflows/compile       Compile to torch.nn.Module
POST /api/v1/workflows/evaluate      Full research pipeline eval (sandbox + fingerprint)
POST /api/v1/workflows/profile       FLOPs/memory/latency profiling
POST /api/v1/workflows/preview       Forward pass with intermediate shapes
POST /api/v1/workflows/estimate      Quick param/FLOPs estimate
PUT  /api/v1/workflows/{id}          Save workflow
GET  /api/v1/workflows/{id}          Load workflow
```

### Aria Co-Design
```
POST /api/v1/aria/propose-patch      Propose graph modification
POST /api/v1/aria/apply-patch        Apply approved patch
POST /api/v1/aria/reject-patch       Reject proposal
POST /api/v1/aria/suggest-components Suggest improvements
POST /api/v1/aria/refine-winner      Generate evolutionary variations
GET  /api/v1/aria/proposals          List proposals
```

### Blocks & Constraints
```
GET  /api/v1/blocks/builtin          List block templates (ffn, attention, etc.)
GET  /api/v1/blocks/builtin/{key}    Get specific block template
POST /api/v1/blocks/extract          Extract nodes as reusable block
POST /api/v1/blocks/expand           Expand block back to nodes
POST /api/v1/constraints/check       Check if component is compatible
POST /api/v1/constraints/palette     Compatibility for all palette items
```

### Import/Export
```
GET  /api/v1/import/survivors        List importable research survivors
POST /api/v1/import/survivors/{id}   Import and save a survivor
POST /api/v1/export/onnx             Export as ONNX model
GET  /api/v1/primitives              List all available primitive ops
```

## Multi-Agent Coordination

This project uses a shared task board in `.current_plan.md`. Before working:
1. Read `.current_plan.md` section 11 (Task Board)
2. Find unclaimed `[ ]` tasks
3. Change to `[C:your_name date]` and add notes
4. When done, change to `[✓]`

Active agents: claude-opus, gemini, codex. Always re-read the plan before editing to avoid conflicts.

### Rules of Engagement for Shared Plans

1. **Claim before coding.** Update the plan claim AND `.current_plan.md` before writing code. Do NOT overwrite another agent's existing claim.
2. **No silent reclaims.** If a task is claimed, don't change the owner. If it seems stale (>48h), add a note asking for status instead.
3. **Don't mark done prematurely.** A task is only `[✓]` when code compiles, tests pass, and the work is listed in `.current_plan.md` completed section.
4. **Conflict resolution.** First agent with a timestamped `.current_plan.md` entry wins. Second agent's changes get reviewed for merge by the user.
5. **Re-read the plan file fresh** before claiming — another agent may have updated it since your last read.

## Testing

```bash
# Full suite (120+ tests)
python -m pytest tests/ --ignore=tests/test_aria_features.py -x -q

# Specific modules
python -m pytest tests/test_bridge.py -v      # 20 tests
python -m pytest tests/test_profiler.py -v    # 8 tests
python -m pytest tests/test_patcher.py -v     # 19 tests
python -m pytest tests/test_importer.py -v    # 11 tests
python -m pytest tests/test_subgraph.py -v    # 19 tests
python -m pytest tests/test_constraints.py -v # 10 tests
python -m pytest tests/test_perf_regression.py -v # 19 tests

# Component contract tests
python -m pytest components/ -x -q
```

## Known Issues

- `numpy.bool_` is not JSON serializable — always cast with `bool()` before returning
- `np.True_ is True` returns `False` — use `== True` or `bool()` cast
- `test_aria_features.py::test_refine_winner` has a pre-existing failure
- `math_space` category manifests fail validation (category not in loader whitelist)

# research/ — Neural Architecture Search Pipeline

Autonomous architecture discovery via grammar-based program synthesis.

## Environment
```bash
source /home/tim/venvs/llm/bin/activate
pytest tests/ -x --tb=short           # after any change
python -m py_compile <file>            # quick syntax check
```

## Structure
- `scientist/` — orchestration (runner, notebook, analytics, API on port 5000)
- `synthesis/` — graph generation (grammar, primitives, compiler, templates)
- `eval/` — evaluation pipeline (sandbox, training_core, wikitext, hellaswag)
- `training/` — micro-training for eval stages
- `dashboard/` — React monitoring UI (port 5000, build in `dashboard/build/`)
- `tools/` — backfill scripts, audits, one-shot analysis

## Key Internals
- `LabNotebook` (SQLite) is the central store. Async writes via `_submit_write` — call `nb.flush_writes()` before reads.
- `ComputationGraph` caches properties in `_cache`. Mutate only via `add_op`/`set_output`.
- Grammar weights: `default * (s1_rate/mean)^2 * (1+novelty)`, clamped [0.5, 8.0].
- `compile_model(use_ir=True)` default uses IRExecutor which lacks `_routing_ctx` — use `getattr(layer, '_routing_ctx', None)`.

## Shared Utilities (use these, don't reinvent)
- `scientist/json_utils.py` — `json_safe()`, `fast_dumps()`, `SafeJSONEncoder`
- `scientist/shared_utils.py` — `safe_float()`, `clamp()`, `safe_json_loads()`
- `scientist/runner/_helpers.py` — `clear_gpu_memory()`, `apply_adaptive_grad_clip()`
- `scientist/trust_policy.py` — `sql_trusted_clause()`, `TRUSTED_TRUST_LABELS`
- `tools/_script_audit.py` — `start_script_experiment()`, `complete_script_experiment()`
- `tools/backfill.py` — `store_probe_results()`, `clear_gpu()`, `query_candidates()`

## Test Commands
```bash
pytest -m "unit and not slow" -n auto --dist loadgroup   # fast unit
pytest -m api -n 2 --dist loadgroup                      # API contracts
pytest -m native -n 1                                     # native/CUDA (single worker)
pytest -m pipeline -n 2 --dist loadgroup                  # pipeline integration
```

## Build
```bash
# aria_core C++ kernels
cd aria_core && python setup.py build_ext --inplace

# Rust scheduler
cd runtime/native/rust/aria-scheduler && maturin develop --release

# Cython bridge
cd runtime/native/cython && python setup.py build_ext --inplace

# C kernels (aria_designer runtime)
cd ../aria_designer/runtime && make clean && make build
```

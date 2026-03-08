# Performance & Code Quality Findings

**Date**: 2026-03-08 (updated)
**Scope**: `research/` codebase
**Status**: Items marked [DONE] are fixed and verified. Others are actionable backlog.

---

## A. DRY Violations

### A.1 Runner Package Helper Duplication [DONE]
- **Impact**: ~1,584 lines of duplicate code removed
- **What**: 4 functions (`_native_proactive_gating`, `_native_runner_progress_report`, `_rebuild_graph_with_overrides`, `propose_ablation_suite`) were copied into all 11 runner submodules
- **Fix**: Created `scientist/runner/_helpers.py` as single source. Each submodule imports only what it uses.

### A.2 Runner Constants Duplication [DONE]
- **What**: `_LIVE_LOSS_CURVE_MAX_POINTS` and `_TRAINING_STEP_SSE_EVERY` defined in 11 files
- **Fix**: Moved to `_types.py`, imported where needed

### A.3 Device Resolution Pattern (15+ locations) [DONE by codex-gpt5]
- **Files**: `runner/execution.py`, `runner/continuous.py`, `runner/synthesis.py`, `runner/dashboard.py`, `runner/screening.py`
- **Pattern**: `dev_str = config.device if torch.cuda.is_available() else "cpu"; dev = torch.device(dev_str)`
- **Fix**: Add `resolve_device(config_device: str) -> torch.device` to `shared_utils.py`

### A.4 Parameter Estimation Duplication [DONE by codex-gpt5]
- **Files**: `synthesis/grammar.py` (lines 370-377 and 656-666 — two definitions of `_estimate_params`)
- **Also**: `synthesis/templates.py` (lines 125-134), `synthesis/graph.py` (lines 187-189) — same formula eval pattern
- **Fix**: Single `estimate_op_params(op, d_in)` function in `primitives.py`

### A.5 Duplicate Validation Logic [DONE by codex-gpt5]
- **Files**: `synthesis/grammar.py:606` `_validate_graph()` vs `synthesis/validator.py:84` `validate_graph()`
- **Both**: Check n_ops, depth, params_ratio, efficiency — similar checks in different ways
- **Fix**: Have `_validate_graph()` delegate to `validate_graph()` and check `.valid`

### A.6 `_to_safe_float` in api.py Duplicates `shared_utils.safe_float` [DONE]
- **File**: `scientist/api.py` (~line 560-588)
- **Fix**: Replace with `from .shared_utils import safe_float as _to_safe_float`

---

## B. Dead Code & Bugs

### B.1 notebook.py `result_id` NameError [DONE]
- **File**: `scientist/notebook.py:2130-2133`
- **What**: `_sync_fingerprint_leaderboard(result_id)` called in `add_entry()` but `result_id` not in scope — crashes at runtime
- **Fix**: Removed the broken sync call (it belongs in `record_program_result`, not `add_entry`)

### B.2 Runner Submodule Import Bloat [DEFERRED]
- **What**: Each of 10 runner submodules has ~80 identical imports copied from the original monolith. Most files use only 5-20.
- **Why deferred**: Mixin pattern means any method can be called on `ExperimentRunner` which combines all mixins. Tests use `unittest.mock.patch("research.scientist.runner.<symbol>")` which requires symbols at the package level. Aggressive pruning breaks patch targets.
- **Safe to do**: Move shared constants to `_types.py`, move shared helpers to `_helpers.py` (already done)
- **Unsafe**: Removing any import from a submodule without verifying no other mixin method uses it

### B.3 `_chat_should_use_code_tools()` in api.py
- **File**: `scientist/api.py` (~line 884)
- **What**: Always returns `True` — trivial, called once. Inline and delete.

---

## C. God Functions / God Files

| File | Function | Lines | Priority |
|------|----------|-------|----------|
| `scientist/api.py` | `create_app()` | 6,369 | CRITICAL — split into Flask Blueprints |
| `runner/continuous.py` | `_run_inline_validation()` | 1,025 | HIGH — extract validation sub-phases |
| `runner/execution.py` | `_run_validation_thread()` | 1,014 | HIGH — extract candidate eval + result aggregation |
| `runner/execution.py` | `_execute_experiment()` | 771 | HIGH — extract screening loop |
| `scientist/persona.py` | `_rule_based_mode_recommendation()` | 563 | MEDIUM — extract decision tree branches |
| `scientist/api.py` | `api_strategy_briefing()` | 648 | MEDIUM |
| `runner/execution.py` | `_micro_train()` | 582 | MEDIUM |
| `runner/results.py` | `_auto_escalate()` | 487 | MEDIUM |
| `synthesis/compiler.py` | `_init_params()` | 310 | MEDIUM — use dispatch table |
| `scientist/native_runner.py` | `compile_model_native_first()` | 591 | LOW |

---

## D. Performance: Python-over-Native Violations

### D.1 `get_primitive()` Caching [ALREADY DONE]
- `synthesis/primitives.py:836` already has `@lru_cache(maxsize=1024)`

### D.2 `__slots__` on Hot Dataclasses [ALREADY DONE]
- `ShapeInfo`, `OpNode` (graph.py), `BehavioralFingerprint` (fingerprint.py), `RunConfig`, `LiveProgress` (_types.py) already have `@dataclass(slots=True)`
- **WARNING**: Do NOT add `slots=True` to `GrammarConfig` — dynamically sets `_split_counter` at `grammar.py:442`

### D.3 O(N² log N) Topological Sort [DONE]
- **File**: `synthesis/graph.py:407-446`
- **What**: `topological_order()` runs `sort()` on ready queue per iteration. Sort key builds strings (op_name, config items) fresh each time.
- **Fix options**:
  1. Pre-compute sort keys before loop, use `heapq` for O(N log N) total
  2. Cache `topological_order()` result (graph is immutable after `set_output()`)
  3. Consider C extension via aria_core for topological sort

### D.4 Unvectorized Influence Matrix in Fingerprinting [DONE]
- **File**: `eval/fingerprint.py:338-353`
- **What**: Python loop perturbing one token at a time, N model forward passes
- **Fix**: Batch all perturbations as rows in a single tensor, one forward pass via `torch.vmap()` or stacked batch dim

### D.5 String Building in `fingerprint()` Hot Path [DONE]
- **File**: `synthesis/graph.py:560-573`
- **What**: Loop builds fingerprint string with f-strings + `sorted()` + `join()` per node. Called per candidate during novelty scoring.
- **Fix**: Pre-allocate list, minimize string ops, or use hash-based fingerprint (SHA256 of serialized graph)

### D.6 Repeated `list(graph.nodes.items())` Copies [ALREADY DONE]
- **File**: `synthesis/grammar.py:518, 524, 531`
- **What**: 3 separate iterations + `list()` copies over same node dict
- **Fix**: Single pass collecting all needed data; collect delete-IDs first, delete in second pass

### D.7 Inefficient Novelty Metric Computation [DONE]
- **File**: `eval/metrics.py:170-175`
- **What**: Manual zero-masking before log creates extra arrays
- **Fix**: `np.log(np.clip(probs, 1e-10, 1.0))` — single vectorized operation

### D.8 `BehavioralFingerprint.to_dict()` Incompatible with `slots=True` [DONE]
- **File**: `eval/fingerprint.py:112`
- **What**: `self.__dict__.copy()` doesn't work with `__slots__`
- **Fix**: `{f.name: getattr(self, f.name) for f in dataclasses.fields(self)}`
- **Status**: NEEDS VERIFICATION — may already be fixed

---

## E. Code Quality Issues

### E.1 Bare `except:` Clauses [DONE]
| File | Line | Severity |
|------|------|----------|
| `tools/arch_linter.py` | 66 | HIGH — hides JSON/network errors |
| `tools/purge_dead_branches.py` | 60 | HIGH — hides data corruption |
| `tools/check_all_zero_robustness.py` | 24 | MEDIUM |
| `tools/recalc_op_stats.py` | 47 | MEDIUM |
| `eval/metrics.py` | 193 | MEDIUM |
| `scientist/analytics.py` | 1587 | MEDIUM |

### E.2 Magic Numbers in Grammar Weights [DONE]
- **File**: `synthesis/grammar.py:926-977`
- **What**: Hardcoded weights (2.0, 1.5, 0.5, 3.0, 4.0, 5.0, 8.0) for action selection with no named constants

### E.3 Magic Numbers in Novelty Scoring [DONE]
- **File**: `eval/metrics.py:143, 166, 170, 206, 216, 225`
- **What**: Hardcoded blend weights and category divisors

---

## F. Native-First Migration Status (carried forward)

Native fast-paths in `synthesis/compiler.py`:
- `poincare_add` -> `aria_core.poincare_add_f32`
- `tropical_add` -> `aria_core.tropical_add_f32`
- `padic_gate` -> `aria_core.padic_gate_f32`
- `lif_neuron` -> `aria_core.lif_neuron_f32`
- `spike_rate_code` -> `aria_core.spike_rate_code_f32`
- `stdp_attention` -> `aria_core.stdp_attention_f32`
- `sparse_threshold` -> `aria_core.sparse_threshold_f32`
- `difficulty_scorer` -> `aria_core.difficulty_scorer_f32`
- `lane_router` -> `aria_core.lane_router_threshold_f32`

Coverage: 66 native-backed handlers, 87 Python-only.

---

## G. Architecture Notes

### G.1 Runner Mixin Pattern — Import Constraints
The `ExperimentRunner` class is composed of 10 mixins via multiple inheritance. Each mixin is in its own file (`scientist/runner/*.py`). Implications:
- **Cannot prune imports per-file** without full cross-mixin analysis
- **Tests patch** `research.scientist.runner.<symbol>` — `__init__.py` must re-export `torch`, `threading`, etc.
- **Safe optimizations**: shared constants → `_types.py`, shared helpers → `_helpers.py`
- **Unsafe**: removing any import without verifying no other mixin's method uses it at runtime

### G.2 `slots=True` Safety Rules
**Cannot use `slots=True`:**
- `GrammarConfig` — dynamically sets `_split_counter` at `grammar.py:442`
- Any dataclass used with `__dict__` access
- Any dataclass that is subclassed with additional fields

**Already using `slots=True` safely:** `ShapeInfo`, `OpNode`, `ComputationGraphIR`, `RunConfig`, `LiveProgress`, `BehavioralFingerprint`

---

## H. Dashboard Performance

### H.1 React Bundle Code-Splitting [DONE]
- **What**: All 30 component imports were eager — every page load fetched the entire app including ArchitectureDrawer (designer iframe), all tab panels, modals
- **Fix**: Converted 18 tab/drawer/modal components to `React.lazy()` with `<Suspense>` fallbacks. Removed 4 unused imports (`AriaStatus`, `TopPrograms`, `MetricsChart`, `StrategyAdvisor`). Main bundle: 100KB + 20 lazy chunks loaded on demand.
- **Impact**: Initial page load only fetches overview components; designer, tabs, drawers load when accessed

---

## Priority Order for Remaining Work

1. **D.3** Topological sort optimization (perf, medium effort)
2. **D.4** Vectorize influence matrix (perf, medium effort)
3. **A.3** Device resolution helper (DRY, low effort)
4. **A.4** Parameter estimation consolidation (DRY, medium effort)
5. **A.6** api.py `_to_safe_float` dedup (DRY, low effort)
6. **E.1** Fix bare except clauses (quality, low effort)
7. **C.1** Split api.py `create_app()` (architecture, high effort)
8. **C.5** Split `_rule_based_mode_recommendation()` (architecture, medium effort)

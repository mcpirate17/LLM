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

### B.2 Runner Submodule Import Bloat [DONE]
- **What**: Each of 10 runner submodules had ~47 identical imports copied from the original monolith.
- **Original concern (WRONG)**: "Mixin pattern means methods reference names from other mixin's imports." False — Python resolves globals per-module, not per-class.
- **Bug found & fixed**: Tests used `patch("research.scientist.runner.threading.Thread")` etc., which patched `__init__.py`'s namespace — NOT the submodule where the call happens. Fixed 14 broken patch targets across 4 test files.
- **Import pruning**: AST-based removal of 599 unused imports across 10 submodules. Each file now imports only what it uses (11-34 imports, down from 47-48).

### B.3 `_chat_should_use_code_tools()` in api.py [DONE]
- **File**: `scientist/api.py` (~line 884)
- **What**: Always returns `True` — trivial, called once. Inline and delete.

---

## C. God Functions / God Files [PLANNED — see `GOD_FILE_SPLIT_PLAN.md`]

Split plan created and prioritized. Remaining god functions are tracked in `GOD_FILE_SPLIT_PLAN.md` (Phases 1-7).

### C.1 `synthesis/compiler.py:_init_params()` Dispatch Refactor [DONE by codex-gpt5]
- **What**: 300+ line monolithic `if/elif` initializer chain.
- **Fix**: Replace with dispatch-table driven entry + grouped helper initializers.

### C.2 `runner/execution.py:_execute_experiment()` Phase Split [DONE by codex-gpt5]
- **What**: 700+ line orchestration method.
- **Fix**: Extract morphology path + orchestrator/dedup helpers into child phase mixin module, parent delegates.

### C.3 `runner/execution.py:_run_validation_thread()` Phase Split [DONE by codex-gpt5]
- **What**: 1000+ line validation orchestration method.
- **Fix**: Extract seed-sweep/model-reconstruction helpers into child validation phase mixin, parent delegates.

### C.4 `runner/execution.py:_micro_train()` Phase Split [DONE by codex-gpt5]
- **What**: 500+ line training worker method.
- **Fix**: Extract deterministic batch, discovery eval, and optional heldout/discovery loss helpers into child micro-train mixin, parent delegates.

### C.5 `runner/continuous.py:_run_inline_validation()` Phase Split [DONE by codex-gpt5]
- **What**: 1000+ line inline validation orchestration method.
- **Fix**: Extract candidate-selection/bootstrap/runtime-prep helpers into child validation phase mixin, parent delegates.

### C.6 `runner/results.py:_auto_escalate()` Phase Split [DONE by codex-gpt5]
- **What**: 400+ line escalation orchestration method handling both screening and investigation branches.
- **Fix**: Extract branch-specific orchestration into child phase mixin helpers, parent dispatcher delegates.

| File | Function | Lines | Status |
|------|----------|-------|--------|
| `scientist/api.py` | `create_app()` | 6,369 | PLANNED — Phase 1: Flask Blueprints (partial work in `api_routes/`) |
| `runner/continuous.py` | `_run_inline_validation()` | 1,025 | [DONE] — Phase 7 |
| `runner/execution.py` | `_run_validation_thread()` | 1,014 | [DONE] — Phase 3 |
| `runner/execution.py` | `_execute_experiment()` | 771 | [DONE] — Phase 3 |
| `scientist/persona.py` | `_rule_based_mode_recommendation()` | 563 | [DONE] — split into decision-tree branches (d6be57b) |
| `scientist/api.py` | `api_strategy_briefing()` | 648 | PLANNED — Phase 1 |
| `runner/execution.py` | `_micro_train()` | 582 | [DONE] — Phase 3 |
| `runner/results.py` | `_auto_escalate()` | 487 | [DONE] — Phase 7 |
| `synthesis/compiler.py` | `_init_params()` | 310 | [DONE] — Phase 6 (dispatch table) |
| `scientist/native_runner.py` | `compile_model_native_first()` | 591 | LOW — well-structured, defer |
| `scientist/notebook/` | (151 methods) | 6,591 | [DONE] — 10 mixin files + `_shared.py`, `__slots__`, `sanitize_for_db` |
| `scientist/analytics/` | (68 methods) | 4,487 | [DONE] — 6 mixin files, `__slots__`, `@staticmethod`, np.percentile |
| `scientist/runner.py` | (dead monolith) | 15,795 | [DONE] — deleted `_runner_dead.py` |

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
- **Status**: Verified fixed

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

### F.1 `conv1d_seq` Native Dispatch in Compiler [DONE by codex-gpt5]
- **File**: `synthesis/compiler.py`
- **What**: `@register_op("conv1d_seq")` currently always uses Python/PyTorch path.
- **Fix**: Use `aria_core.conv1d_seq_f32` on CPU float32 non-grad path, keep PyTorch fallback for autograd/CUDA.

### F.2 `token_pool_restore` Native Dispatch in Compiler [DONE by codex-gpt5]
- **File**: `synthesis/compiler.py`
- **What**: `@register_op("token_pool_restore")` currently always uses Python path.
- **Fix**: Use `aria_core.token_pool_restore_f32` on CPU float32 non-grad path, keep PyTorch fallback for autograd/CUDA.

### F.3 `tropical_center` Native Dispatch in Compiler [DONE by codex-gpt5]
- **File**: `synthesis/compiler.py`
- **What**: `@register_op("tropical_center")` currently always uses Python `cummin` path.
- **Fix**: Use `aria_core.tropical_center_f32` on CPU float32 non-grad path, keep `cummin` fallback.

### F.4 `rope_rotate` Native Dispatch in Compiler [DONE by codex-gpt5]
- **File**: `synthesis/compiler.py`
- **What**: `@register_op("rope_rotate")` currently always uses Python trig/reshape path.
- **Fix**: Use `aria_core.rope_rotate_f32` on CPU float32 non-grad path, keep current Python fallback.

### F.5 `rwkv_time_mixing` Native Dispatch in Compiler [DONE by codex-gpt5]
- **File**: `synthesis/compiler.py`
- **What**: `@register_op("rwkv_time_mixing")` currently runs Python-heavy recurrence/approximation path.
- **Fix**: Use `aria_core.rwkv_time_mixing_f32` on CPU float32 non-grad path when module params are available; keep existing fallback.

### F.6 `causal_mask` Native Dispatch in Compiler [DONE by codex-gpt5]
- **File**: `synthesis/compiler.py`
- **What**: `@register_op("causal_mask")` currently always uses Python `cumsum` path.
- **Fix**: Use `aria_core.causal_mask_f32` on CPU float32 non-grad path with fallback.

### F.7 `sort_seq` Native Dispatch in Compiler [DONE by codex-gpt5]
- **File**: `synthesis/compiler.py`
- **What**: `@register_op("sort_seq")` currently always uses Python `argsort` + `gather`.
- **Fix**: Use `aria_core.sort_seq_f32` on CPU float32 non-grad path with validated output-shape fallback.

### F.8 `argsort_seq` Native Dispatch in Compiler [DONE by codex-gpt5]
- **File**: `synthesis/compiler.py`
- **What**: `@register_op("argsort_seq")` currently always uses Python `argsort`.
- **Fix**: Use `aria_core.argsort_seq_f32` on CPU float32 non-grad path with output-normalization fallback.

### F.9 `topk_gate` Native Dispatch in Compiler [DONE by codex-gpt5]
- **File**: `synthesis/compiler.py`
- **What**: `@register_op("topk_gate")` currently always uses Python linear + softmax + split path.
- **Fix**: Use `aria_core.topk_gate_f32` on CPU float32 non-grad path when `gate_proj` has expected shape; keep Python fallback.

### F.10 `cosine_similarity` Native Dispatch in Compiler [DONE by codex-gpt5]
- **File**: `synthesis/compiler.py`
- **What**: `@register_op("cosine_similarity")` currently always uses `F.cosine_similarity`.
- **Fix**: Use `aria_core.cosine_similarity_f32` on CPU float32 non-grad path with output-shape validation; keep fallback.

### F.11 `embedding_lookup` Native Dispatch in Compiler [DONE by codex-gpt5]
- **File**: `synthesis/compiler.py`
- **What**: `@register_op("embedding_lookup")` currently pass-through only.
- **Fix**: Add guarded native `aria_core.embedding_lookup_f32` when input is integer token IDs and `embed_table` exists; keep pass-through fallback.

### F.12 `basis_expansion` Native Dispatch in Compiler [DONE by codex-gpt5]
- **File**: `synthesis/compiler.py`
- **What**: `@register_op("basis_expansion")` currently always uses Python trig expansion.
- **Fix**: Add guarded `aria_core.basis_expansion_f32` fast path with strict shape validation and fallback.

---

## G. Architecture Notes

### G.1 Runner Mixin Pattern — Import Rules [UPDATED]
The `ExperimentRunner` class is composed of 10 mixins via multiple inheritance. Each mixin is in its own file (`scientist/runner/*.py`).
- **Imports ARE per-module** — Python resolves globals per-module, not per-class. Each submodule only needs its own imports. (599 unused imports pruned in B.2.)
- **Test patches must target submodules** — `patch("research.scientist.runner.control.threading.Thread")`, NOT `patch("research.scientist.runner.threading.Thread")`. The latter patches `__init__.py`, not the submodule where the call happens.
- `__init__.py` re-exports `RunConfig`, `LiveProgress`, `propose_ablation_suite`, `torch`, `threading` for backward compatibility.

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

All A, B, D, E, H items are DONE. Most C items are DONE. Remaining:

1. **C — `scientist/api.py`** `create_app()` + `api_strategy_briefing()` (Phase 1: Flask Blueprints — in progress by claude-opus)
2. **C — `scientist/native_runner.py`** `compile_model_native_first()` 591 lines (LOW — well-structured, defer)
3. **F** Native-first migration — 87 Python-only handlers remaining (long-tail)

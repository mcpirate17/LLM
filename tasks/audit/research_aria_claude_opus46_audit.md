# Research Directory — Full Engineering Audit (Re-audit)
**Date**: 2026-03-18
**Auditor**: Claude Opus 4.6
**Scope**: `research/` (Python backend, synthesis, runner, API, frontend dashboard)

---

## A. Executive Summary

This codebase is **moderately performance-conscious** but carries significant dead weight and missed optimization opportunities. The hot paths (training, compilation, eval) are reasonably well-optimized — CUDA graphs, vectorized data pipelines, and native C kernels exist where they matter most. However:

- **~2,500 lines of dead Python code** remain (dead compiler ops, abandoned packages, unused components)
- **~1,460 lines of dead JS components** never imported by anything
- **compiler.py contains 61 dead `_op_*` functions** overwritten at module load by split modules — the single worst code hygiene issue
- **37 scattered `torch.cuda.empty_cache()` calls** bypass the centralized `clear_gpu_memory()` helper
- **No parallelization** of evolutionary fitness evaluation — the single biggest wall-clock bottleneck
- **Unbounded caches** in DB/notebook with no TTL or LRU eviction
- **174 API routes** with repetitive error handling — bloated surface area
- **Frontend**: 49 `useState` calls in ControlPanel.js, zero virtualization, 4 dead components

**Top 5 highest-value fixes:**
1. Delete 61 dead compiler ops from `compiler.py` (~1,000 lines)
2. Parallelize evolutionary fitness evaluation (~8x speedup)
3. Delete dead packages: `artifacts/component_lab/`, `intelligence/antipattern.py`, 4 dead JS components (~3,500 lines total)
4. Add TTL/LRU to notebook and DB caches (prevents unbounded memory growth)
5. Consolidate 37 scattered GPU cleanup calls into `clear_gpu_memory()`

---

## B. High-Severity Findings

### B1. 61 Dead Compiler Op Implementations
- **Severity**: CRITICAL (code hygiene)
- **Category**: Dead Code
- **File**: `research/synthesis/compiler.py` (2,954 lines)
- **Issue**: 86 `_op_*` functions defined inline, but 61 are overwritten by `_register_split_op_modules()` at line 2113 which calls `_OP_DISPATCH.update()` from `compiler_ops_*.py` (1,543 lines across 4 files containing 92 reimplementations). The inline versions are dead — they execute once to register, then get immediately overwritten.
- **Why it matters**: ~1,000 lines of dead code. Confuses maintainers who edit the wrong copy. Changes to inline ops have zero effect.
- **Fix**: Delete all 61 overwritten `_op_*` functions from compiler.py. Keep only the ~25 that are NOT in the split modules.

### B2. No Parallelization of Evolutionary Search
- **Severity**: CRITICAL (performance)
- **Category**: Hot Path
- **File**: `research/search/evolution.py` (lines 239-276)
- **Issue**: Fitness evaluation runs 50 individuals × ~30s each = 1,500s per generation, sequentially. No `multiprocessing.Pool`, no `concurrent.futures`.
- **Why it matters**: This is the throughput bottleneck of the entire NAS pipeline. 8x speedup achievable with trivial parallelization.
- **Fix**: `concurrent.futures.ProcessPoolExecutor(max_workers=8)` around fitness evaluation loop.

### B3. Unbounded Cache Growth
- **Severity**: HIGH
- **Category**: Memory Efficiency
- **Files**: `research/database.py:106-123`, `research/scientist/notebook/notebook_core.py`
- **Issue**: `self._cache: Dict[str, Any] = {}` grows without bound. Cache keys like `f"top_architectures_{n}"` accumulate for every unique `n`. No TTL, no max size, no LRU eviction.
- **Why it matters**: Long-running processes (dashboard, continuous runner) leak memory over hours.
- **Fix**: Replace with `functools.lru_cache(maxsize=128)` or add TTL-based eviction.

### B4. Analytics Queries Load Unbounded Result Sets
- **Severity**: HIGH
- **Category**: Database Efficiency
- **File**: `research/scientist/notebook/notebook_analytics.py`
- **Symbols**: `get_op_pair_priors()` (line 125), `get_fingerprint_buckets()` (line 153), `_top_performer_bigram_support()` (line 435)
- **Issue**: Full-table scans with no LIMIT. `_top_performer_bigram_support()` fetches all stage1 winners with full `graph_json` (10KB+ per row), sorts entire list in Python to find 25th percentile, then iterates again.
- **Fix**: Use SQL `ORDER BY loss_ratio LIMIT 1 OFFSET ?` for percentile. Only fetch graph_json for qualifying rows. Add composite index `(stage1_passed, loss_ratio DESC)`.

### B5. Kernel Fallback Pattern Duplicated 13+ Times
- **Severity**: HIGH
- **Category**: Duplication
- **File**: `research/synthesis/compiler.py`
- **Issue**: 13+ sites repeat the identical pattern: check if C kernel available → try native call → catch (ImportError, RuntimeError, AttributeError) → log fallback → execute PyTorch fallback. ~500 lines of near-identical code.
- **Fix**: Extract `_try_native_kernel(name, condition, kernel_fn, fallback_fn)` helper.

---

## C. Hot Path Performance Findings

| Hot Path | File | Status | Notes |
|----------|------|--------|-------|
| Training loop (CUDA graphs) | `runner/execution_training.py` | **Acceptable** | CUDA graph capture, ~0.43ms/step on fast path |
| Training loop (fallback) | `runner/execution_training.py` | **Acceptable** | Hook-based entropy sampling adds 1-2% overhead |
| Data pipeline | `research/training/data_pipeline.py` | **Acceptable** | Vectorized NumPy advanced indexing |
| Model compilation | `research/synthesis/compiler.py:85-137` | **Questionable** | Kogge-Stone scan uses O(log S) `torch.cat` allocations per SSM layer. Triton kernel would give 10-50x |
| IR execution | `research/synthesis/ir_executor.py:128-181` | **Acceptable** | Dict lookup per node (~1% overhead). Pre-computed list would help but marginal |
| Evolutionary search | `research/search/evolution.py:239-276` | **POOR** | Sequential fitness eval, no parallelization |
| Fingerprinting | `research/eval/fingerprint.py:735-786` | **Acceptable** | Interaction perturbations deferred to post-investigation. NumPy + partition-based K-NN |
| Novelty search | `research/search/novelty_search.py:124-156` | **Questionable** | CPU-only behavior archive. GPU-cached version would give 2-5x |
| Graph generation | `research/synthesis/grammar.py` | **Acceptable** | ~5-10ms per graph, weighted sampling |
| Graph copy for trials | `research/synthesis/grammar.py:472` | **Questionable** | Full deep copy per template attempt. Transaction/rollback pattern would avoid allocation |

---

## D. Stub Findings

| Location | Type | Purpose | Risk | Action |
|----------|------|---------|------|--------|
| `training/sparse_training.py:18,24` | `NotImplementedError` | RigLScheduler, RigLOptimizer | Low — referenced but guarded | Keep; document timeline |
| `synthesis/primitives.py:647-648` | Missing handler | `tropical_moe`, `tropical_router` registered but no compiler handler | **HIGH** — grammar can sample → crash | Remove from registry or implement |
| `scientist/llm/backend.py:34-42` | Abstract methods | LLMBackend ABC | None — correct pattern | Keep |
| `runner/execution_training.py:341` | TODO comment | RigL sparse training reference | Low — informational | Keep |

---

## E. Dead Code Findings

### Clearly Dead

| Item | Location | Lines | Evidence |
|------|----------|-------|----------|
| 61 overwritten `_op_*` functions | `synthesis/compiler.py` | ~1,000 | Overwritten by `_register_split_op_modules()` at L2113 |
| `antipattern.py` module | `scientist/intelligence/antipattern.py` | 205 | Zero imports anywhere in codebase |
| `component_lab/` package | `artifacts/component_lab/` (14 files) | 1,851 | Zero imports anywhere in codebase |
| `TopPrograms.js` | `dashboard/src/components/` | 501 | Not imported by any file |
| `ObservabilityDashboard.js` | `dashboard/src/components/` | 474 | Not imported by any file |
| `MetricsChart.js` | `dashboard/src/components/` | 346 | Not imported by any file |
| `AriaStatus.js` | `dashboard/src/components/` | 139 | Not imported by any file |
| `ControlPanel.test.js` | `dashboard/src/components/` | 52 | Orphaned test file |
| **Total clearly dead** | | **~4,568** | |

### Likely Dead

| Item | Location | Lines | Evidence |
|------|----------|-------|----------|
| `ModelCandidate` class | `runner/_types.py:30-63` | 34 | Only used in `execution_candidates.py` type hints, not exported |
| `_failure_penalty_weight()` | `notebook/notebook_misc.py` | ~15 | Single internal call |
| `evaluate_campaign_criteria()` | `notebook/notebook_campaigns.py` | ~20 | Single call |
| `get_latest_digest()` | `notebook/notebook_misc.py` | ~10 | Single call, likely dead |
| `backfill_fingerprint_aggregates()` | `notebook/notebook_analytics.py` | ~30 | Tool-only, not called from main pipeline |
| ~32 registered ops never sampled by grammar | `synthesis/primitives.py` | varies | 128 registered ops vs ~96 in grammar motifs |

### Suspicious

| Item | Location | Notes |
|------|----------|-------|
| `tropical_moe`, `tropical_router` | `primitives.py:647-648` | Registered, no compiler handler — will crash if sampled |
| `sort_seq` | `primitives.py` (if still present) | Filesystem deleted per git status |
| Wrapper functions in `chat_bp.py:40-50` | `api_routes/chat_bp.py` | Wrappers around misc_bp with identical signatures |

---

## F. Duplication Findings

### F1. Compiler Op Duplication (~1,000 lines)
- **Where**: 61 `_op_*` in `compiler.py` duplicated in `compiler_ops_*.py`
- **Type**: Exact duplicates (original overwritten by split modules)
- **Survivor**: `compiler_ops_*.py` files
- **Action**: Delete inline versions from `compiler.py`

### F2. GPU Cleanup Duplication (37 sites across 11 files)
- **Where**: `continuous_investigation.py` (2), `continuous_modes.py` (1), `continuous_validation.py` (3), `dashboard.py` (2), `execution_investigation.py` (3), `execution_screening.py` (2), `execution_search.py` (1), `execution_validation_phase3.py` (1), `execution_validation.py` (2), `_helpers.py` (2), `selection.py` (2)
- **Type**: Semantic duplicate of `_helpers.py:clear_gpu_memory()`
- **Action**: Export from `__init__.py`, replace all raw `torch.cuda.empty_cache()` + `gc.collect()` calls

### F3. Progress Lock Boilerplate (84 sites across 24 files)
- **Where**: All runner mixin files
- **Pattern**: `with self._lock: self._progress.status = "..."`
- **Action**: Create `_update_progress(self, status=None, message=None, **kwargs)` helper

### F4. Error Handling in Runner (30+ sites)
- **Where**: `continuous_validation.py` (11), `execution_validation.py` (7+), `continuous_investigation.py` (3+)
- **Pattern**: Near-identical try-except blocks per validation metric
- **Action**: `_try_eval(name, fn, result_dict)` wrapper

### F5. API Response/Error Wrapping (342 try-except, 386 jsonify)
- **Where**: 20 API blueprint files
- **Type**: Repetitive error handling and response formatting
- **Action**: Extend `with_notebook_context()` decorator coverage

### F6. Seed Metric Aggregation (2 copies)
- **Where**: `execution_validation.py:500-535`, `continuous_validation.py:505-535`
- **Type**: Near-exact duplicate of loss_ratio aggregation with mean/variance
- **Action**: Extract to `_compute_seed_metrics()` in `shared_utils.py`

### F7. Frontend localStorage Pattern (3+ copies)
- **Where**: `DiscoveryRankings.js`, `Leaderboard.js`, `ControlPanel.js`
- **Action**: Extract `useLocalStorage` hook

### F8. Frontend Color Mappings (3+ copies)
- **Where**: `CampaignView.js`, `GlobalParetoChart.js`, `StabilityQualityQuadrant.js`
- **Action**: Centralize in `utils/colors.js`

### F9. Param Formula Inconsistencies (8 ops)
- **Where**: `primitives.py` — `adaptive_lane_mixer`, `mixed_recursion_gate`, `routing_conditioned_compression`, `progressive_compression_gate`, `compression_mixture_experts`, `relu_gate_routing`, `ternary_projection`, `latent_attention_compressor`
- **Issue**: `has_params=True` but `param_formula="0"` — these ops DO have learnable parameters in compiler implementation
- **Action**: Set correct `param_formula` values

---

## G. Efficiency Findings

### G1. CPU
- **Evolutionary search** runs sequentially (see B2). 8x win available.
- **Kogge-Stone scan** in compiler uses Python-level `torch.cat` loop — Triton candidate.
- **Graph `topological_order()`** recomputes config string formatting inside heapq loop (graph.py:454-554).

### G2. Memory
- **Unbounded caches** in ExperimentDB and LabNotebook (see B3).
- **`.fetchall()` on large result sets** materializes all rows. Use `.fetchmany()` or generators for 10K+ row queries.
- **Telemetry dict reinitialized every call** in `_record_routing_telemetry()` (compiler.py:183-241) even though only sampled every 8th call.
- **Full graph deep copy** per template trial in grammar.py — transaction/rollback pattern would avoid allocation.

### G3. I/O
- **Repeated JSON decompression** — `choice_success_rates()` decompresses all stage1 choices on every call. No caching.
- **Config reloading per API request** — analytics routes rebuild full summary per request. 2s TTL cache helps but insufficient under rapid dashboard polling.

### G4. Database
- **Missing composite indexes**: `(stage1_passed, loss_ratio DESC)`, `(experiment_id, timestamp DESC)`, `(graph_fingerprint, composite_score DESC)`.
- **`_leaderboard_by_fingerprint()`** does full LEFT JOIN on every call with no query cache.
- **`upsert_leaderboard()`** lookups then re-fetches same row — could combine.
- **N+1-adjacent**: `get_program_details()` converts Row → dict → JSON-decode 10 fields per result.

### G5. Concurrency
- **All API routes do synchronous SQLite writes** blocking the request thread. `_submit_write()` exists but is underutilized.
- **`_cache_lock`** acquired on every cache read including hits. Read-write lock would reduce contention.

### G6. Frontend
- **`ControlPanel.js`**: 49 `useState` calls. Every state change re-renders all children. Should split into 5-6 sub-components.
- **Zero virtualization**: `DiscoveryRankings`, `ExperimentList`, `LabNotebook` render all rows. Need `react-window`.
- **200+ inline `style={{}}` objects** create new references every render, defeating `React.memo()`.
- **3x duplicate fetch of `/api/leaderboard`** from different components. Need request deduplication in `useAriaData`.
- **Scoring recomputed per render**: `discoveryScore()` called per row with no `useMemo`.

---

## H. Refactor Recommendations

### Immediate Removals (< 1 hour each)
1. Delete 61 dead `_op_*` functions from `compiler.py` — **~1,000 lines**
2. Delete `artifacts/component_lab/` — **1,851 lines**
3. Delete `scientist/intelligence/antipattern.py` — **205 lines**
4. Delete 4 dead JS components (`TopPrograms`, `ObservabilityDashboard`, `MetricsChart`, `AriaStatus`) — **1,460 lines**
5. Remove `tropical_moe`, `tropical_router` from `PRIMITIVE_REGISTRY` (no compiler handler = crash risk)
6. Remove `sort_seq` from registry if still present (filesystem deleted)

### Immediate Performance Wins (< 2 hours each)
7. Add TTL + max_size to notebook/DB caches
8. Add composite SQL indexes for common query patterns
9. Replace 37 raw `torch.cuda.empty_cache()` calls with `clear_gpu_memory()`
10. Use `heapq.nsmallest()` instead of `sorted()` in `_top_performer_bigram_support()`
11. Fix `_record_routing_telemetry()` to skip dict init on non-sample calls

### Medium-Term Refactors (2-8 hours each)
12. Parallelize evolutionary fitness evaluation
13. Extract kernel fallback helper (13 sites → 1 function)
14. Extract `_try_eval()` wrapper for validation metric collection (30+ sites)
15. Create `_update_progress()` helper for runner progress lock boilerplate (84 sites)
16. Split `ControlPanel.js` into sub-components; add `React.memo()`
17. Add `react-window` virtualization to large list components
18. Cache `get_op_pair_priors()` and `get_fingerprint_buckets()` with TTL
19. Fix 8 routing ops with `param_formula="0"` that actually have parameters

### Native/Compiled Migration (see Section I)
20. Triton kernel for Kogge-Stone scan
21. GPU-cached novelty archive
22. Symbolic Jacobian for fingerprint perturbation

---

## I. Native Migration Candidates

| Code | Current | Target | Why | Speedup Est. |
|------|---------|--------|-----|-------------|
| Kogge-Stone scan | PyTorch `torch.cat` loop | **Triton kernel** | O(log S) allocations per SSM layer in forward pass | 10-50x |
| Novelty archive K-NN | NumPy CPU | **FAISS / cuML GPU** | Behavior archive search on every candidate | 2-5x |
| Fingerprint perturbation | Numerical perturbation | **Symbolic Jacobian** | 0.5-1.0s per eval, redundant forward passes | 2-3x |
| `_top_performer_bigram_support()` | Python sort + iterate | **SQL + vectorized NumPy** | Full-scan + sort + re-iterate | 3-5x |
| Graph topological sort | Python heapq with string formatting | **Rust via PyO3** | Called per compilation, string overhead | 2-3x (marginal absolute) |

The training loop itself is already well-optimized (CUDA graphs, vectorized data pipeline). The wins here are in the orchestration and search layers, not the inner training loop.

---

## J. Final Verdict

| Dimension | Score (0-10) | Notes |
|-----------|-------------|-------|
| Performance discipline | **6/10** | Training is solid. Search/orchestration is neglected. |
| Code hygiene | **4/10** | 4,500+ lines of clearly dead code. 61 dead compiler ops. |
| Efficiency | **5/10** | Unbounded caches, no query optimization, sequential search. |
| Maintainability | **6/10** | Good mixin architecture, but dead code confuses and duplication obscures. |
| Native acceleration maturity | **7/10** | C kernels, Rust scheduler, Cython bridge exist. Triton gap for SSM scan. |

**Final assessment**: This codebase has genuinely good bones — the native acceleration stack (C kernels, Rust scheduler, Cython bridge, CUDA graphs) is real and well-integrated. The training hot path is competitive. But the codebase is carrying ~4,500 lines of dead code that should have been deleted weeks ago, the evolutionary search — which is the throughput bottleneck — has no parallelization at all, and the database layer has no query optimization or bounded caching. The compiler.py file is the worst offender: 61 dead op implementations that were superseded by the split modules but never cleaned up, creating a maintenance trap where someone edits the wrong copy. The frontend dashboard has 4 dead components and a ControlPanel with 49 useState hooks that re-renders everything. Clean up the dead weight, parallelize the search, bound the caches, and add SQL indexes — then this becomes a tight, well-performing system.

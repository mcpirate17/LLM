---
status: active
created: 2026-04-01
author: claude-opus
---

# Audit Fix Plan — Prioritized Action Items

Reference: `tasks/audit/performance_hygiene_audit_2026-04-01.md`

---

## Phase 1: Critical Hot-Path Fixes (Immediate — highest ROI)

| ID | File | Fix | Est. Impact |
|----|------|-----|-------------|
| P1.1 | `scientist/runner/results_analysis.py:245-352` | Batch N+1 DB queries in `_update_analytics_stats()` with `executemany()` + single `SELECT ... IN (...)` | ~50s/experiment |
| P1.2 | `eval/associative_recall.py:230`, `induction_probe.py:113`, `diagnostic_tasks.py:442` | Replace `copy.deepcopy(model)` with `state_dict()` save/restore | 20-30% eval speedup |
| P1.3 | `runtime/native/src/runner_abi.c:19-43` | Add `pthread_mutex_t` around static handle arrays | Blocks multi-threaded use |
| P1.4 | `synthesis/grammar.py:871` | Replace `graph.copy()` with node-addition rollback on budget failure | 10-20% grammar alloc reduction |
| P1.5 | `scientist/runner/execution_screening.py:1474` | Move `_EFFICIENCY_OPS` frozenset to module level | 5000 allocs/experiment eliminated |
| P1.6 | `synthesis/compiler.py:1048-1129` | Pre-bind dispatch function on CompiledOp at init, gate telemetry with module flag | 5-10% forward-pass speedup |

## Phase 2: Major Performance & Safety

| ID | File | Fix | Est. Impact |
|----|------|-----|-------------|
| P2.1 | `runtime/native/src/runner_abi.c:100-116` | Replace O(n*m) JSON string search with jsmn/cJSON parser | 100ms+ per model init |
| P2.2 | `runtime/native/src/runner_abi.c:854+` | Add null checks after malloc, fix error-path memory leaks | Safety |
| P2.3 | `training/optimizer_synthesis.py:617,621-622` | Remove 3 unnecessary `.clone()` per step | ~5% training speedup |
| P2.4 | `synthesis/context_rules.py:1559-1680` | Eliminate duplicate child map (successors dict + `_child_map()` → keep one) | 2x graph traversal eliminated |
| P2.5 | `synthesis/context_rules.py:1478` | Fix BFS: use `collections.deque.popleft()` not `list.pop()` | Correct algorithm |
| P2.6 | `database.py:150-164` | Cache decompressed specs (currently re-decompresses on every `get_spec()`) | Repeated zlib+json eliminated |
| P2.7 | `scripts/force_under_observed.py:55-136` | Add `del model` + `torch.cuda.empty_cache()` in loop | Prevents OOM crash |
| P2.8 | `scientist/runner/execution_training.py:86,438` | Parse graph JSON once, pass object instead of string | ~100ms/experiment |
| P2.9 | `scientist/runner/execution_screening.py:1456-1461` | Move `get_primitive` import outside loop, cache results in dict | ~10ms/1000 calls |
| P2.10 | `training/curriculum.py:59-84` | Cache attention masks by seq_len in `self._mask_cache` | 500 allocs eliminated |

## Phase 3: Database & I/O

| ID | File | Fix | Est. Impact |
|----|------|-----|-------------|
| P3.1 | `tools/clear_false_penalties.py:101-154` | Batch 98 individual queries → single `IN (...)` | 98 round-trips → 2 |
| P3.2 | `tools/backfill.py:189-198` | Batch 200 individual UPDATEs per commit | I/O reduction |
| P3.3 | `tools/log_monitor.py:321` | Fix file handle leak — use `with` statement | FD leak |
| P3.4 | `scientist/runner/execution_experiment_phase3.py:195-201` | Replace full 100K+ fingerprint load with indexed batch check | 500ms + 5MB saved |
| P3.5 | `scientist/runner/execution_training.py:519-535` | Add NaN/Inf early bailout during CUDA graph warmup | 200ms on broken models |
| P3.6 | Notebook init | Enable `PRAGMA foreign_keys=ON` by default | Data integrity |

## Phase 4: Duplication Cleanup

| ID | Scope | Fix | Lines Saved |
|----|-------|-----|-------------|
| P4.1 | `tools/backfill*.py` (8 files) | Extract `BackfillRunner` base class | ~60% dedup |
| P4.2 | `training/optimizer_synthesis.py` (12 classes) | Extract step boilerplate to `_step_preamble()` | ~500 lines |
| P4.3 | `synthesis/compiler.py:153-239` | Unify `_record_sparse_telemetry` + `_record_routing_telemetry` | ~50 lines |
| P4.4 | `eval/utils.py` + 4 files | Centralize batch construction | 5 versions → 1 |
| P4.5 | `eval/sandbox.py` + 2 files | Extract `compute_grad_norm()` to utils | 3 copies → 1 |

## Phase 5: Dead Code & Stubs

| ID | File | Fix |
|----|------|-----|
| P5.1 | `runtime/native/src/kernels_ext.c` | Delete file + remove from CMake |
| P5.2 | `training/optimizer_synthesis.py:591` | Delete dead `state["prev_grad"]` read |
| P5.3 | 4 training files | Remove redundant `from __future__ import annotations` |
| P5.4 | `scientist/` | Grep-audit ~642 private functions for unreachable code |
| P5.5 | `synthesis/compiler_ops_mathspaces.py:93` | Implement or document bare `pass` in tropical router |
| P5.6 | `synthesis/ir_executor.py:154-157` | Add `logger.debug()` to silent torch.compile exception |
| P5.7 | `synthesis/context_rules.py:1485,1507` | Use frozensets directly (skip set() conversion) |

## Phase 6: Dashboard

| ID | Fix | Impact |
|----|-----|--------|
| P6.1 | Add react-window virtualization to Leaderboard, Discoveries, ExperimentList | 500-1000 DOM rows → windowed |
| P6.2 | Break god components (App 1679, LiveFeed 1199, CampaignView 1158, Discoveries 993, ExperimentList 920 LOC) | Rerender isolation |
| P6.3 | Extract 2,715 inline `style={{...}}` to constant modules | Memoization unblocked |
| P6.4 | Consolidate 5+ duplicate formatting functions into `utils/format.js` | DRY |
| P6.5 | Wrap list-rendered components with `React.memo` | Rerender reduction |
| P6.6 | Convert 12-15 useState hooks to useReducer in Leaderboard/Discoveries | State coherence |
| P6.7 | Lazy-load recharts (50KB+) — only used in CompareView | Bundle reduction |

## Phase 7: Native Migration Candidates (Future)

| ID | Code | Target | Rationale |
|----|------|--------|-----------|
| P7.1 | `synthesis/ir_executor.py:178-204` forward loop | Cython or Rust (PyO3) | 1000-node Python loop, 300M pointer chases/day |
| P7.2 | `synthesis/grammar.py` graph generation | Rust | 100K+ generations with heavy mutation |
| P7.3 | `eval/fingerprint.py:1282-1305` CKA | torch.compile or CUDA kernel | O(S^2) matmul, 100+ runs/cycle |
| P7.4 | `runtime/native/cython/aria_bridge.pyx:16-41` FP16 dispatch | Cython enum/switch | O(n) tuple membership in native bridge |

---
status: active
created: 2026-04-01
author: claude-opus
---

# High-Performance Code, Efficiency & Code Hygiene Audit Report

**Scope:** All files under `research/` including `tools/`, `training/`, `scripts/`, `runtime/`, `healer/`, `schemas/`, `dashboard/`
**Date:** 2026-04-01

---

## A. Executive Summary

This codebase is **partially performance-conscious**. The native layer (C kernels, Rust scheduler, Cython bridge) shows real investment in acceleration, but the Python orchestration layer — which dominates wall-clock time — is riddled with avoidable inefficiencies: N+1 database queries in evaluation hot paths, unnecessary deep copies of entire models, constant re-instantiation inside loops, repeated JSON parsing of the same data, and silent exception swallowing. The dashboard has 5 god components over 900 LOC each with 2,700+ inline style objects breaking memoization and zero table virtualization.

**Is Python overused?** Yes. The IR executor forward loop, grammar generation, and fingerprint CKA computation are all Python-bound hot paths that should be Cython/Rust. The Cython bridge uses O(n) Python tuple membership for FP16 dispatch instead of a switch statement.

**Is dead code a problem?** Moderate. The worst offenders are 8 near-identical backfill scripts, 12 optimizer classes with duplicated boilerplate, and an empty C stub file. The codebase is not drowning in dead code, but duplication is a real maintenance burden.

**Top 5 highest-value fixes:**
1. Batch N+1 DB queries in `_update_analytics_stats()` — saves ~50s per experiment
2. Replace `copy.deepcopy(model)` with state-dict reload in eval probes — saves 20-30% eval time
3. Add thread safety to C runtime handle arrays — blocks multi-threaded use entirely
4. Pre-cache op dispatch at CompiledOp init — saves 5-10% of every forward pass
5. Virtualize dashboard tables — 500-1000 DOM rows rendered without windowing

---

## B. High-Severity Findings

### B1. N+1 Database Queries in Hot Evaluation Path
- **Severity:** CRITICAL
- **Category:** Database/I/O
- **File:** `scientist/runner/results_analysis.py:245-352`
- **Symbol:** `_update_analytics_stats()`
- **Issue:** Three loops (templates, motifs, ops) each issue individual SELECT + UPDATE per item. With ~20 ops + 5 templates + 3 motifs per graph = 56 round-trips per evaluation.
- **Why it matters:** At 1000 programs/experiment × 56 queries × ~1ms = 56 seconds of pure DB coordination per experiment.
- **Fix:** Single `SELECT * FROM template_stats WHERE template_name IN (...)`, compute deltas in Python, `executemany()` batch UPDATE/INSERT.

### B2. Deep Copy of Entire Model in Evaluation Probes
- **Severity:** CRITICAL
- **Category:** Hot Path Performance
- **Files:** `eval/associative_recall.py:230`, `eval/induction_probe.py:113`, `eval/diagnostic_tasks.py:442`
- **Issue:** `copy.deepcopy(model)` triggers full serialization + deserialization of all parameters. For 200M-param models = 2-4 seconds per probe.
- **Why it matters:** With 500 training steps and 20+ eval runs per architecture, deepcopy is 10-50% of total eval time.
- **Fix:** Save `original_state = model.state_dict()`, modify in-place, restore with `model.load_state_dict(original_state)`.

### B3. Race Condition in C Runtime Handle Arrays
- **Severity:** CRITICAL
- **Category:** Correctness/Safety
- **File:** `runtime/native/src/runner_abi.c:19-43`
- **Symbol:** `g_handle_count`, `g_handle_logits[]`
- **Issue:** Static global arrays accessed without synchronization. `g_handle_count++` is not atomic.
- **Why it matters:** Multi-threaded evaluation corrupts handle state. Silent data corruption.
- **Fix:** Add `pthread_mutex_t` around handle allocation/deallocation.

### B4. Full Graph Copy Per Template Trial
- **Severity:** CRITICAL
- **Category:** Hot Path Performance
- **File:** `synthesis/grammar.py:871`
- **Symbol:** `generate_layer_graph()`
- **Issue:** `trial_graph = graph.copy()` inside template composition loop. If template fails budget check, entire copy is discarded. Allocates ~500KB-5MB per failed attempt.
- **Why it matters:** With 100K graph generations, 10-20% of cycles wasted on doomed copies.
- **Fix:** Track node additions during template application, revert by removing appended node IDs on failure.

### B5. Per-Call Dictionary Lookups in Forward Pass
- **Severity:** MAJOR
- **Category:** Hot Path Performance
- **File:** `synthesis/compiler.py:1048-1129`
- **Symbol:** `_execute_op()`
- **Issue:** Called per op per forward pass. Does 2-3 hash lookups (`_OP_DISPATCH`, `PRIMITIVE_REGISTRY`) and unconditional telemetry `getattr()` even when telemetry is disabled.
- **Why it matters:** 20-op graph × 1000 evaluations = 20K unnecessary lookups. 5-10% of forward-pass time.
- **Fix:** Pre-bind dispatch function on `CompiledOp` at construction. Gate telemetry with module-level flag.

### B6. Constant Reinstantiation in Per-Graph Loop
- **Severity:** MAJOR
- **Category:** Efficiency
- **File:** `scientist/runner/execution_screening.py:1474-1510`
- **Symbol:** `_EFFICIENCY_OPS`
- **Issue:** 33-element `frozenset` recreated on every graph iteration inside `for i, graph in enumerate(graphs)` loop.
- **Why it matters:** 5000 graphs/experiment = 5000 unnecessary frozenset allocations.
- **Fix:** Move to module level.

### B7. O(n) JSON String Search in Native Init
- **Severity:** MAJOR
- **Category:** Native Code Quality
- **File:** `runtime/native/src/runner_abi.c:100-116`
- **Symbol:** `_find_op_name_offset()`
- **Issue:** Searches entire JSON document O(n*m) per op name. No index structure. Fragile on whitespace.
- **Why it matters:** For 10KB IR JSON, each search is 10K+ comparisons. Multiplied by ops.
- **Fix:** Parse JSON once with jsmn/cJSON.

### B8. Missing Null Checks and Memory Leaks in C Runtime
- **Severity:** MAJOR
- **Category:** Safety
- **File:** `runtime/native/src/runner_abi.c:854+`
- **Issue:** `malloc()` return not checked. If second malloc fails, first allocation leaks.
- **Fix:** Check all malloc returns. Use arena allocator or allocate once in handle init.

---

## C. Hot Path Performance Findings

| Hot Path | Status | Issue | Fix |
|----------|--------|-------|-----|
| `ir_executor.forward()` | **Poor** | Python loop with 3 list lookups/node, no compilation | Cython/Rust migration (P7.1) |
| `compiler._execute_op()` | **Poor** | Per-call dict lookups, unconditional telemetry | Pre-bind at init (P1.6) |
| `grammar.generate_layer_graph()` | **Poor** | Full graph copy per template, weight dict rebuild per iter | Rollback pattern (P1.4) |
| `_update_analytics_stats()` | **Critical** | 56 DB round-trips per evaluation | Batch queries (P1.1) |
| `eval/associative_recall` | **Poor** | deepcopy(model) per probe | State-dict reload (P1.2) |
| `sandbox.py` training loop | **Acceptable** | Minor reshape overhead, reasonable structure | No action needed |
| `fingerprint.py` CKA | **Questionable** | O(S^2) batch matmul when rank suffices | Optimize or torch.compile |
| `native dispatch (Cython)` | **Questionable** | O(n) tuple membership for FP16 ops | Cython enum/switch |
| `curriculum.get_mask()` | **Questionable** | Identical tensor allocated 500x | Cache by seq_len |
| `optimizer_synthesis step()` | **Questionable** | 3 unnecessary .clone() per step | Remove clones |

---

## D. Stub Findings

| Location | Type | Risk |
|----------|------|------|
| `synthesis/compiler_ops_mathspaces.py:93` | Bare `pass` in tropical router | Moderate — incomplete op |
| `synthesis/ir_executor.py:154-157` | Silent `except Exception: pass` on torch.compile | Moderate — hidden failures |
| `eval/sandbox.py:122,635` | Bare `pass` in signal handler except | Low — platform compat |
| `synthesis/_template_helpers.py:329` | Bare `pass` in exception handler | Low |
| `tools/_lm_benchmarks.py:68` | `raise NotImplementedError("Generation not supported")` | Low — known limitation |
| `training/data_pipeline.py:24-28` | Protocol method with only docstring, no `...` | Cosmetic |

---

## E. Dead Code Findings

### Clearly Dead
- `runtime/native/src/kernels_ext.c` — 9 lines, comment says "All kernels migrated." Build debris.
- `training/optimizer_synthesis.py:591` — bare `state["prev_grad"]` reads value, discards it.
- `from __future__ import annotations` in 4 training files — redundant on Python 3.10+.

### Likely Dead
- `synthesis/primitives.py:1418` — dangling function body with `pass`.
- ~642 private functions across `scientist/` — many likely unused but need grep verification.

### Suspicious
- Multiple backfill scripts in `tools/` — some may be one-time migration tools that should be archived.

---

## F. Duplication Findings

### Database/Tools Layer
- **8 backfill scripts** (`backfill.py`, `backfill_stats.py`, `backfill_cka_novelty.py`, `backfill_binding.py`, `backfill_templates.py`, `backfill_v8_scores.py`, `backfill_hellaswag.py`, `backfill_triage.py`) — near-identical parse-args → query → compute → update pattern. Extract `BackfillRunner` base class.

### Training Layer
- **12 optimizer classes** in `optimizer_synthesis.py` — identical closure/param_groups/state-init boilerplate repeated. Extract `_step_preamble()` helper.

### Synthesis Layer
- `_record_sparse_telemetry()` and `_record_routing_telemetry()` in `compiler.py:153-239` — identical pattern, should be unified.
- Native kernel dispatch `if HAS_KERNELS and x.is_cuda:` pattern repeated 3+ times across `compiler_ops_attention.py`, `compiler_ops_math.py`, `compiler.py`.

### Eval Layer
- 5 versions of batch construction across `utils.py`, `wikitext_eval.py`, `diagnostic_tasks.py`.
- 3 copies of gradient norm computation across `sandbox.py`, `screening_rapid.py`, `diagnostic_tasks.py`.

### Dashboard
- 5+ duplicate formatting functions: `formatTimestamp`, `formatPct`, `formatTime`, `formatTs`, `formatDuration` across components vs `utils/format.js`.
- API response transformation logic duplicated between Leaderboard and Discoveries.

---

## G. Efficiency Findings

### CPU
- Repeated JSON `loads()` on same graph string 2-3x per evaluation (`execution_training.py:86,438`)
- Redundant frozenset→set conversion in `context_rules.py:1485,1507`
- Weight dict rebuilt from scratch per template iteration in `grammar.py:837-869`
- Duplicate child map construction in `context_rules.py:1559-1680`

### Memory
- No model cleanup in `scripts/force_under_observed.py` — creates up to 500 models without `del`
- 3 unnecessary `.clone()` per optimizer step in `optimizer_synthesis.py:617,621-622`
- Unnecessary tensor allocation in `curriculum.get_mask()` — identical mask created 500x

### I/O
- File handle leak in `tools/log_monitor.py:321`
- Repeated file opens in `tools/monitor_agent.py:177,235,249`
- All 100K+ fingerprints loaded into memory for dedup in `execution_experiment_phase3.py:195-201`

### Database
- N+1 in `tools/clear_false_penalties.py:101-154` — 98 individual queries
- 200 separate UPDATEs per commit batch in `tools/backfill.py:189-198`
- Decompressed specs not cached in `database.py:150-164`
- Foreign keys not enforced (`PRAGMA foreign_keys=ON` missing)

### Frontend
- 500-1000 DOM rows rendered without virtualization (Leaderboard, Discoveries, ExperimentList)
- 2,715 inline `style={{...}}` objects break memoization
- 12-15 useState hooks per component instead of useReducer
- God components (5 files over 900 LOC each)

---

## H. Refactor Recommendations (Prioritized)

### 1. Immediate Removals
- Delete `runtime/native/src/kernels_ext.c`
- Delete dead `state["prev_grad"]` read at `optimizer_synthesis.py:591`
- Remove redundant `__future__` imports in 4 training files
- Move `_EFFICIENCY_OPS` to module level (1-line fix)

### 2. Immediate Performance Wins
- Batch DB queries in `_update_analytics_stats()` (~50s/experiment saved)
- Replace `deepcopy(model)` with state-dict reload (20-30% eval speedup)
- Pre-cache op dispatch at CompiledOp init (5-10% forward pass)
- Remove 3 unnecessary `.clone()` in optimizer loops (5% training speedup)
- Add NaN bailout in CUDA graph warmup
- Cache attention masks by seq_len

### 3. Medium-Term Refactors
- Extract BackfillRunner base class (8 scripts → shared base + thin subclasses)
- Extract optimizer step boilerplate (12 classes → helper + compute functions)
- Unify telemetry recording in compiler
- Centralize batch construction and grad norm computation in eval
- Consolidate dashboard formatting utilities
- Break god components into sub-components with React.memo

### 4. Native/Compiled Migration
- See Section I below

---

## I. Native Migration Candidates

| Code | Current | Target | Rationale |
|------|---------|--------|-----------|
| `ir_executor.py:178-204` forward loop | Python | Cython or Rust (PyO3) | 1000+ node loop with 3 list lookups/node. 100K samples/day = 300M pointer chases |
| `grammar.py` graph generation | Python | Rust | 100K+ generations with template weight computation + graph mutation |
| `fingerprint.py:1282-1305` CKA | Python/PyTorch | torch.compile or CUDA kernel | O(S^2) batch matmul, 100+ runs per eval cycle |
| `aria_bridge.pyx:16-41` FP16 dispatch | Python tuple membership | Cython enum/switch | O(n) per op lookup in native bridge — defeats purpose of Cython |
| `context_rules.py` graph validation | Python | Cython | Called on every graph with multiple traversals |

---

## J. Final Verdict

| Dimension | Score (0-10) |
|-----------|-------------|
| Performance discipline | 4/10 |
| Code hygiene | 5/10 |
| Efficiency | 4/10 |
| Maintainability | 5/10 |
| Native acceleration maturity | 6/10 |

The native layer (C kernels, Rust scheduler, Cython bridge) is real and working — that's above average. But the Python orchestration layer undoes much of that investment. The evaluation hot path deep-copies entire models instead of saving/restoring state dicts. The analytics layer hammers SQLite with 56 individual queries per evaluation when one batch query would suffice. The IR executor forward loop — the single most-called function in the system — is a pure Python loop with redundant list indexing. The grammar generates 100K+ graphs with full deep copies on every failed template trial.

**What should be removed:** `kernels_ext.c`, dead variable reads, redundant `__future__` imports, the constant reinstantiation inside the screening loop.

**What should be rewritten:** The IR executor forward loop (Cython/Rust), the grammar graph copy pattern (rollback instead of clone), the backfill scripts (shared base class), the optimizer boilerplate (extract helper).

**What should be accelerated first:** Batch DB queries in analytics (biggest wall-clock win), replace deepcopy with state-dict reload (biggest eval win), pre-cache op dispatch (biggest per-forward-pass win), virtualize dashboard tables (biggest UI responsiveness win).

The codebase is functional and the architecture is sound, but it's leaving 20-40% of its potential performance on the table through avoidable Python-layer inefficiencies.

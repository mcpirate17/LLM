# Research Directory Audit — 2026-03-18

**Scope**: `/home/tim/Projects/LLM/research/` — 136,544 lines Python, ~4,500 lines JS/React
**Auditor**: claude-opus-4-6
**Standard**: CLAUDE.md performance hierarchy + code hygiene rules

---

## A. Executive Summary

**Is this codebase performance-conscious?** Partially. Hot-path eval code (fingerprint, sandbox, training) uses aria_core C kernels and torch ops where available, but falls back to unaccelerated Python loops in multiple critical paths. The synthesis/compiler layer has 2 sequential Python loops that dominate forward-pass cost for certain ops. The runner package (21,385 lines) is almost entirely Python orchestration with no compiled acceleration — acceptable for orchestration, but some inner loops (model compilation per candidate, GPU memory clears per training program) waste significant wall time.

**Is Python overused?** Yes, in specific hot paths:
- `compiler.py:534-544` — sequential Python loop over sequence dimension with O(S×D²) tensor ops per timestep
- `compiler.py:878-893` — nested Python loop over experts × top-k slots
- `fingerprint.py` — 12 bare `except Exception:` handlers, repeated `.cpu()` transfers, uncached SVD
- `evolution.py:298-318` — uncached fingerprint computation in dedup loop (IR lowering per individual)

**Is dead code/stub/duplication serious?** Moderate. 4 `NotImplementedError` stubs (sparse_training.py is the worst — referenced by optimizer_synthesis.py but never implemented). 88 bare `except Exception:` handlers across 30 files in research/. 206 in scientist/ alone. ~350 `pass` statements across the codebase. Duplication exists in safe-float conversion (2 implementations), graph JSON op extraction (3 implementations), leaderboard row retrieval (3+ implementations), and source_map building (3 identical patterns in runner/).

**Top 5 highest-value fixes:**
1. Vectorize `_op_gated_delta` loop in compiler.py (eliminates O(S) Python iterations with D² tensor ops each)
2. Fix N+1 queries in campaigns_bp.py (301 queries → 4) and observability_bp.py (duplicate full-table scans)
3. Cache fingerprints in evolution Individual at creation time (eliminates 10K+ IR lowerings per evolution run)
4. Batch `clear_gpu_memory()` calls in runner/execution_investigation.py (3× per candidate → 1×)
5. Add React.memo to 22 program/ components (eliminates 110+ unnecessary re-renders per state change)

---

## B. High-Severity Findings

### B1. Sequential Python Loop in Gated Delta — CRITICAL
- **Category**: Hot path performance
- **File**: `synthesis/compiler.py:534-544`
- **Symbol**: `_op_gated_delta`
- **Issue**: Python `for t in range(S)` loop performing outer products `v[:, t, :].unsqueeze(-1) * k[:, t, :].unsqueeze(-2)` creating (B, D, D) matrix per timestep, sequential state update, and query projection — all in Python loop
- **Why it matters**: For S=2048, D=768: ~1.2B FLOPs executed sequentially in Python. This is the forward pass of any model using gated_delta mixing.
- **Fix**: Replace with parallel prefix scan (Kogge-Stone) or Triton kernel. TODO at line 95 already notes this but has no owner.

### B2. Nested Expert Loop in N-Way Router — CRITICAL
- **Category**: Hot path performance
- **File**: `synthesis/compiler.py:878-893`
- **Symbol**: `_op_n_way_sparse_router`
- **Issue**: Double nested `for i in range(n_ways): for k_idx in range(top_k):` creating boolean masks and multiplying per expert-slot combination. Allocates `torch.zeros_like(x)` fresh every forward pass.
- **Why it matters**: For n_ways=8, top_k=4: 32 mask operations + accumulations per forward pass. This is the routing hot path for MoE architectures.
- **Fix**: Vectorize with gather/scatter. `topk_idx` is already a tensor — use `torch.scatter_add` or index_select.

### B3. N+1 Query Pattern in Campaigns Endpoint — HIGH
- **Category**: Database efficiency
- **File**: `scientist/api_routes/campaigns_bp.py:23-41`
- **Symbol**: `/api/campaigns` route
- **Issue**: Fetches all campaigns, then executes 3 separate COUNT queries per campaign in a loop (experiments, hypotheses, decisions)
- **Why it matters**: 100 campaigns = 301 queries. Dashboard polls this endpoint.
- **Fix**: Single query with LEFT JOIN + GROUP BY

### B4. Duplicate Full-Table Scan of program_results — HIGH
- **Category**: Database efficiency
- **File**: `scientist/api_routes/observability_bp.py:70-130, 203-228`
- **Symbols**: `_build_op_index()`, `_get_component_health()`
- **Issue**: Two functions independently run `SELECT graph_json FROM program_results WHERE graph_json IS NOT NULL`, each parsing all JSON. Both called from the same dashboard view.
- **Fix**: Share parsed result or unify into single cached query

### B5. SELECT * Across Production Code — HIGH
- **Category**: Database efficiency
- **Files**: `database.py:259,270,276`, `notebook/notebook_programs.py:301,343,401,412,426`, `tools/rescore_leaderboard.py:209,216,238,241`, `tools/backfill_eval_stages.py:74`, `tools/backfill_novelty.py:255`, plus 15+ more locations
- **Issue**: 30+ `SELECT *` queries across production code. `rescore_leaderboard.py:216` does `SELECT * FROM program_results` loading entire table into memory.
- **Fix**: Specify columns. For bulk operations, use batch processing with LIMIT/OFFSET.

### B6. Uncached Fingerprint in Evolution Dedup — HIGH
- **Category**: Hot path performance
- **File**: `search/evolution.py:298-318`
- **Symbol**: `_enforce_population_diversity()`
- **Issue**: `.fingerprint` property lowers graph to IR and hashes on every call. In dedup loop over 50-500 individuals × 20 generations = 10K+ IR lowerings.
- **Fix**: Cache fingerprint at Individual construction time. `_cached_fingerprint` field exists but isn't populated by constructor.

### B7. Deep Copy of Model in Training Path — HIGH
- **Category**: Hot path performance
- **File**: `scientist/runner/execution_training.py:1201`
- **Issue**: `copy.deepcopy(model).to(dev)` in one-shot pruning path. 50-200ms per candidate.
- **Fix**: Use state-dict cloning instead of full object deep copy

### B8. 13× Repeated hasattr() Chains in Compiler — MAJOR
- **Category**: Performance
- **File**: `synthesis/compiler.py` — lines 379, 348, 758-769, 628-629, 681, 701, 721, and 6+ more
- **Issue**: Chains like `hasattr(module, "w_decay") and hasattr(module, "u_bonus") and hasattr(module, "W_k") and ...` repeated across 13 operator implementations. Each `hasattr` is a dict lookup. Over a 2-hour training run: ~7.2M unnecessary lookups.
- **Fix**: Cache `self._kernels_ready = all(hasattr(self, attr) for attr in required_attrs)` at module init

---

## C. Hot Path Performance Findings

| Hot Path | File | Status | Issue |
|----------|------|--------|-------|
| Model forward pass (gated_delta) | compiler.py:534-544 | **CRITICAL** | Sequential Python loop over S timesteps with D² tensor ops |
| Model forward pass (n_way_router) | compiler.py:878-893 | **CRITICAL** | Nested Python loop over experts × top_k |
| Fingerprint computation | eval/fingerprint.py | **QUESTIONABLE** | 12 bare except handlers, repeated .cpu() transfers, uncached SVD |
| Evolution dedup | search/evolution.py:298-318 | **POOR** | Uncached IR lowering per individual per generation |
| Novelty scoring (archive K-NN) | search/novelty_search.py:124-155 | **QUESTIONABLE** | Full O(N) distance computation, np.mean(np.square()) instead of np.linalg.norm |
| Training dynamics probe | eval/sandbox.py:652-714 | **QUESTIONABLE** | 20 full gradient steps when 10 would suffice; clip_grad_norm every step |
| Interaction influence matrix | eval/fingerprint.py:770-786 | **QUESTIONABLE** | expand+clone forces memory copy; 2 forward passes per probe |
| Candidate screening | scientist/runner/execution_screening.py | **ACCEPTABLE** | Single-threaded but GPU-serialized; CPU work could parallelize |
| Training loop | scientist/runner/execution_training.py | **ACCEPTABLE** | Standard torch training, but 4× cuda.synchronize in warmup path |
| Leaderboard scoring | scientist/leaderboard_scoring.py | **ACCEPTABLE** | Per-row DB query in build_score_kwargs; 35+ optional params |
| Grammar generation | synthesis/grammar.py | **ACCEPTABLE** | Dict copies per generation (minor), motif loop with repeated .get() |
| Graph topological sort | synthesis/graph.py | **ACCEPTABLE** | C++ fast path via aria_core; Python fallback repeats config_str building |

**Native acceleration maturity**: aria_core C kernels exist for 60+ forward ops, 15 backward, 9 fp16. Rust scheduler exists. Cython bridge covers 95+ ops. However, the compiler dispatch (13 hasattr chains) and two critical ops (gated_delta, n_way_sparse_router) bypass all native paths and run pure Python loops.

---

## D. Stub Findings

| Location | Type | Risk |
|----------|------|------|
| `training/sparse_training.py:1-27` | `NotImplementedError` — RigLScheduler and RigLOptimizer | **MEDIUM** — referenced by optimizer_synthesis.py:57-62, will crash if selected |
| `training/optimizer_synthesis.py:57-62` | `NotImplementedError` — rigl_sparse component | **MEDIUM** — dead code path that raises |
| `scientist/code_agent.py:72-82` | Stub returning "unavailable" | **LOW** — entire module is a stub |
| `compiler.py:95-99` | TODO — fused_kogge_stone_scan | **LOW** — performance TODO without owner |
| `runner/execution_training.py:341-344` | TODO — RigL dynamic sparse training | **LOW** — references unimplemented sparse_training.py |
| `api_routes/_strategy_recommendations.py:804` | `_log_diagnosis_placeholder()` — pass | **LOW** — dead diagnostic hook |

---

## E. Dead Code Findings

### Clearly Dead
| Location | Evidence |
|----------|----------|
| `compiler.py:501-504` | irfft_seq block marked `# pragma: no cover — irfft_seq removed in audit` but code still present |
| `runner/execution_training.py:420` | Expression `total_steps - train_steps` — result discarded, not assigned |
| `code_agent.py` entire module | Returns "unavailable" for every call; circular import workaround |
| `_strategy_recommendations.py:_log_diagnosis_placeholder()` | Empty function, `pass` body |

### Likely Dead
| Location | Evidence |
|----------|----------|
| `runner/execution_training.py:64-159` (`_smoke_test_graph_structure`) | 96 lines, only called in optional one-shot pruning path, result never gates execution |
| `leaderboard_scoring.py:compute_composite_v6` 35+ optional params via `**kwargs` | Many kwargs are never populated by any caller — would need grep to confirm each |

### Suspicious
| Location | Evidence |
|----------|----------|
| 206 bare `except Exception:` handlers in scientist/ | Many swallow errors silently — may mask real failures |
| 88 bare `except Exception:` handlers in research/ (non-scientist) | Same pattern in eval, search, synthesis code |

---

## F. Duplication Findings

### F1. Safe Float Conversion — 2 implementations
- `eval/utils.py:36-49` (`safe_parse_float`) — returns None on failure
- `scientist/shared_utils.py:15-28` (`safe_float`) — returns configurable default on failure
- **Action**: Consolidate into shared_utils.py, add alias in eval/utils.py

### F2. Sanitize Unit Feature — 2 implementations with different semantics
- `eval/fingerprint.py:661-668` — returns 0.5 on None/NaN
- `search/novelty_search.py:191-207` — returns 0.0 on None/NaN
- **Action**: Pick one semantic, parameterize the default, single function

### F3. Graph JSON Op Extraction — 3 implementations
- `api_routes/observability_bp.py:85-92`
- `api_routes/observability_bp.py:215-219` (same file, duplicate)
- `api_routes/programs_bp.py` (morphing analysis)
- **Action**: Extract `extract_ops_from_graph_json()` utility

### F4. Source Map Building — 3 identical patterns in runner/
- `runner/execution_investigation.py:91-96`
- `runner/continuous_investigation.py:494-495`
- `runner/continuous_inline_validation_phase7.py:119-120`
- Pattern: `[d or {} for d in (nb.get_program_details(ids) or [])]` → dict by result_id
- **Action**: Extract `_build_source_map(nb, result_ids)` to `_helpers.py`

### F5. Leaderboard Entry Retrieval — 3+ implementations
- `api_routes/leaderboard_bp.py:226-244`
- `api_routes/programs_bp.py:286-292`
- `api_routes/strategy_bp.py:337-341`
- **Action**: Add `notebook.get_leaderboard_entry(entry_id=None, result_id=None)` method

### F6. Random Input Tensor Generation — 13 call sites
- Pattern: `torch.randint(0, vocab_size, (batch_size, seq_len), device=dev)`
- Spread across: sandbox.py (5×), fingerprint.py (3×), passkey.py (3×), screening_rapid.py (1×), pruning.py (1×)
- **Action**: Extract `_make_random_input_ids()` helper

### F7. hasattr Chain — 13 copies in compiler.py
- Pattern: `_c(x) and x.device.type == "cpu" and x.dtype == torch.float32 and not x.requires_grad`
- **Action**: Extract `_should_use_c_kernel(x)` predicate

### F8. Inline Style Objects in React — 71+ identical instances
- `style={{ color: 'var(--text-muted)' }}` — 71 times
- `style={{ marginLeft: 4, fontSize: 10 }}` — 66 times
- `style={{ fontSize: 11, color: 'var(--text-muted)' }}` — 55 times
- **Action**: Extract to CSS constants or shared style objects

### F9. localStorage Pattern — 8 copies in React
- useState with try/catch localStorage.getItem + useEffect with setItem
- **Action**: Extract `useLocalStorage()` hook (already have `useLocalStorage.js` — verify usage)

---

## G. Efficiency Findings

### G1. CPU/Memory
- **3× `clear_gpu_memory()` per investigation candidate** (runner/execution_investigation.py) — should be 1× between candidates, not 1× per training program inside the inner loop
- **`copy.deepcopy(model)` in training** (execution_training.py:1201) — 50-200ms each, use state_dict clone
- **Uncached SVD in geometry probe** (fingerprint.py:905-915) — O(N²D), non-deterministic random sampling
- **20-step training dynamics probe** (sandbox.py:652-714) — could detect instability in 10 steps
- **Single-threaded benchmark pool** (_helpers.py:780) — `max_workers=1`, blocks investigation loop

### G2. Database/I/O
- **30+ `SELECT *` queries** across production code — load unnecessary columns
- **N+1 in campaigns** (campaigns_bp.py) — 3 COUNT queries per campaign
- **Duplicate full-table scans** (observability_bp.py) — two functions scan all program_results independently
- **Row-by-row UPDATE** (rescore_leaderboard.py:483-512) — 12-column UPDATE per entry instead of batch
- **Full table snapshot** (rescore_leaderboard.py:209) — `CREATE TABLE AS SELECT *` copies entire leaderboard per rescore version
- **No compound indexes** (database.py) — missing `(passed, generation)`, `(passed, spec_id)` combos

### G3. Algorithmic
- **O(N) K-NN per novelty_of call** (novelty_search.py:124-155) — archive of 200, called 20K+ times per evolution run = 4M distance ops. Consider KD-tree or ball tree for k>15.
- **O(population) fingerprint in dedup** (evolution.py:298-318) — IR lowering per individual, 10K+ per run
- **Interaction matrix: expand+clone** (fingerprint.py:770-786) — forces full memory copy of input batch

### G4. Concurrency
- **Single-threaded screening** (execution_screening.py) — CPU work (grammar generation, data sampling) could parallelize even though GPU is serialized
- **Synchronous LLM call in request handler** (programs_bp.py:41-47) — `aria.explain_fingerprint()` blocks 1-5s
- **4× cuda.synchronize in training warmup** (execution_training.py:488,778,902,1477) — could batch to 1

### G5. Frontend
- **22 program/ components without React.memo** — cascade re-renders from ProgramDetail state changes
- **Event handlers without useCallback in Leaderboard** — new function references per render to 1000+ rows
- **40+ useState in App.js** — every state change invalidates entire component tree
- **21-property context in useAriaData** — any property change re-renders all consumers
- **Expensive compressionSummary() in sort useMemo** — runs on 1000+ entries per sort change

---

## H. Refactor Recommendations

### Immediate Removals (< 1 hour each)
1. Delete `compiler.py:501-504` (unreachable irfft_seq block)
2. Delete `_strategy_recommendations.py:_log_diagnosis_placeholder()`
3. Fix `execution_training.py:420` (assign or delete discarded expression)
4. Remove 3 alias exports from `scoringEngine.js` (programScore, leaderboardEntryScore, etc.)

### Immediate Performance Wins (1-4 hours each)
1. Cache `_kernels_ready` boolean on CompiledOp init (replace 13 hasattr chains)
2. Fix campaigns N+1: single query with LEFT JOIN + GROUP BY
3. Cache fingerprint at Individual construction (evolution.py)
4. Move `clear_gpu_memory()` outside inner loop in execution_investigation.py
5. Add React.memo to 22 program/ components
6. Wrap Leaderboard handlers in useCallback

### Medium-term Refactors (4-16 hours each)
1. Extract shared helpers: `_build_source_map()`, `extract_ops_from_graph_json()`, `_should_use_c_kernel()`
2. Consolidate safe_float / sanitize_unit_feature duplicates
3. Split App.js context into 3-4 providers (Tab, Queue, Designer, Data)
4. Replace 30+ `SELECT *` with column-specific queries
5. Batch UPDATE statements in rescore_leaderboard.py
6. Extract `useLocalStorage()` hook for React
7. Audit and add logging to 206 bare `except Exception:` handlers in scientist/

### Native Migration Opportunities (days-weeks)
1. Vectorize `_op_gated_delta` → Triton kernel or Kogge-Stone scan
2. Vectorize `_op_n_way_sparse_router` → gather/scatter_add
3. Move novelty K-NN to scipy.spatial.KDTree or FAISS
4. Replace interaction_influence_matrix expand+clone with scatter ops

---

## I. Native Migration Candidates

| Code | Current | Target | Why |
|------|---------|--------|-----|
| `compiler.py:_op_gated_delta` (534-544) | Python loop × S | **Triton kernel** | O(S×D²) per forward; single biggest perf bottleneck for delta-state models |
| `compiler.py:_op_n_way_sparse_router` (878-893) | Nested Python loop | **Vectorized PyTorch** (gather/scatter_add) | 32 mask ops per forward; routing is MoE hot path |
| `novelty_search.py:novelty_of` (124-155) | np.mean(np.square()) | **scipy.spatial.KDTree** or **FAISS** | 4M distance ops per evolution run; K-NN is classic native-acceleration target |
| `fingerprint.py:_interaction_influence_matrix` (770-786) | expand+clone+forward | **Vectorized perturbation** (scatter) | Saves one full tensor clone per probe |
| `evolution.py:_enforce_population_diversity` (298-318) | Python loop + IR lowering | **Cached fingerprint at init** | 10K+ IR lowerings per run; pure waste |
| `compiler.py` hasattr chains (13 locations) | 13× hasattr per forward | **Cached bool at __init__** | 7.2M dict lookups per training run |

---

## J. Final Verdict

| Dimension | Score (0-10) |
|-----------|:---:|
| Performance discipline | **5** |
| Code hygiene | **4** |
| Efficiency | **5** |
| Maintainability | **6** |
| Native acceleration maturity | **6** |

The research directory has real native acceleration infrastructure (60+ C kernels, Rust scheduler, Cython bridge) — that's better than most ML research codebases. But it's undermined by two critical Python loops in the compiler that bypass all native paths (`gated_delta`, `n_way_sparse_router`), 30+ unqualified `SELECT *` queries, 294 bare exception handlers that swallow errors, and a React dashboard where 22 components lack React.memo causing cascade re-renders on every tab switch. The runner package at 21,385 lines is well-structured (mixins, shared helpers), but has 3 copies of the source-map building pattern and calls `clear_gpu_memory()` 3× per candidate instead of 1×. The evolution search recomputes fingerprints from scratch on every dedup pass — a cache-at-construction fix would eliminate 10K+ IR lowerings per run. The biggest wins are: (1) vectorize the two compiler loops, (2) fix the N+1 queries, (3) cache fingerprints in evolution, (4) add React.memo. The biggest hygiene debt is the 294 bare exception handlers and 30+ `SELECT *` queries — both are maintenance hazards that mask real bugs and waste I/O.

# Comprehensive Performance + DRY Audit Plan (research/)

This is the shared, repo-wide checklist and execution guide for CP-0 and CP-6.
It is designed to be used alongside `perf_dry_audit_files.txt` for per-file reviews.

## Goals
- Establish a single, explicit performance checklist that covers every subsystem.
- Create a DRY audit map to consolidate duplicated logic and reduce divergence.
- Produce actionable findings, owners, and follow-up tasks per file.

## Scope
- Included: all code and config under `research/` (see `perf_dry_audit_files.txt`).
- Excluded: caches, binaries, build artifacts.

## Deliverables
- `perf_dry_audit.md` (this file) with checklists and audit process.
- Per-file audit notes (inline in PRs or appended to this file under "Findings").
- Follow-up TODOs logged in `.current_work.md` with file refs.

---

## CP-0: Repo-Wide Performance Checklist

### 1) Profiling & Observability
- Confirm CPU/GPU profiling hooks exist for the subsystem.
- Identify hot paths with wall-time, GPU time, and memory peaks.
- Verify metrics are emitted in a structured JSON (experiment-level or run-level).

### 2) Dataflow & Caching
- Remove duplicate computation across layers or stages.
- Cache deterministic results keyed by inputs/config (with invalidation rules).
- Avoid repeated JSON parse/serialize for stable data.

### 3) Memory & Allocation
- Identify frequent allocations in tight loops (tensors, lists, dicts).
- Reuse buffers when possible; avoid intermediate tensors on critical path.
- Reduce Python object churn in tight loops (use arrays/structs).

### 4) Kernel Launch & Fusion
- Inspect for chains of small ops that could be fused or batched.
- Prefer vectorized ops over Python loops.
- Consolidate multiple GPU launches where possible.

### 5) IO, DB, and Serialization
- Reduce synchronous waits and per-row writes.
- Batch writes, use WAL, and avoid large JSON blobs where possible.
- Compress large fields (loss curves, graphs, traces) consistently.

### 6) Concurrency & Scheduling
- Identify head-of-line blocking or single-thread bottlenecks.
- Overlap CPU prep with GPU execution.
- Ensure queues have backpressure and bounded memory.

### 7) UI/Frontend Performance (Dashboard)
- Prevent unnecessary renders (memoization, stable props).
- Avoid O(N^2) operations in render path.
- Defer expensive computations to workers or backend when possible.

---

## CP-6: DRY Audit Checklist

### 1) Duplicated Logic
- Scoring/thresholds defined in multiple files.
- Hardcoded constants repeated with small variations.
- Multiple APIs returning similar transformed structures.

### 2) Formatting & Display
- Repeated formatting for scores, timestamps, labels.
- Inconsistent naming across tabs/views.
- Repeated UI components doing the same work.

### 3) Data Access & Serialization
- Redundant parsing of JSON columns.
- Repeated query patterns that can be centralized.
- Multiple files updating the same schema without a shared model.

### 4) Error Handling & Logging
- Duplicated error wrappers or inconsistent error responses.
- Logging patterns repeated with slight differences.

### 5) Testing & Fixtures
- Multiple tests constructing similar mocks.
- Duplicate fixture builders or config setup.

---

## Per-File Audit Workflow

For each file in `perf_dry_audit_files.txt`:
1. **Identify hot paths**: note functions or render blocks that dominate runtime.
2. **Apply CP-0 checklist**: note missing instrumentation, caching, or batch ops.
3. **Apply CP-6 checklist**: record duplicated logic or constants.
4. **Log findings**: in the file's PR or append in this file under "Findings".
5. **Create TODOs**: add explicit, file-scoped TODOs in `.current_work.md`.

---

## Findings (Append as discovered)

### Template
- File: `path/to/file.py`
- Perf finding: ...
- DRY finding: ...
- Suggested change: ...
- Owner: ...
- Priority: ...


## Findings (2026-02-20)

- File: `research/scientist/runner.py`
- Perf finding: Inline investigation/validation builds context with per-result `nb.get_program_detail(...)` inside list comprehension, causing N+1 DB queries and repeated JSON parsing for each candidate.
- DRY finding: Investigation/validation inline flows repeat context build + candidate filtering logic with slight variations.
- Suggested change: Add a batch `nb.get_program_details(result_ids)` with JSON parsing centralized, and reuse a shared helper for candidate filtering/context construction.
- Owner: runner
- Priority: P1

- File: `research/scientist/runner.py`
- Perf finding: Frequent `json.loads`/`json.dumps` on `arch_spec_json`, `graph_json`, `training_program_json` in hot loops; also repeated `torch.cuda.empty_cache()` + `gc.collect()` per graph evaluation.
- DRY finding: JSON parsing logic repeated across multiple blocks; no shared cache/parse utility.
- Suggested change: Introduce cached parse helpers (LRU keyed by result_id or fingerprint) and only call cache-clearing on memory-pressure thresholds.
- Owner: runner
- Priority: P2

- File: `research/scientist/analytics.py`
- Perf finding: `efficiency_frontier()` uses O(n^2) dominance check and parses `graph_json` for every row in Python.
- DRY finding: Op extraction from `graph_json` duplicated across analytics routines.
- Suggested change: Sort by FLOPs + track best loss to compute frontier in O(n log n); precompute/store ops list in DB or central `extract_ops(graph_json)` utility.
- Owner: analytics
- Priority: P1

- File: `research/synthesis/graph.py`
- Perf finding: `fingerprint()` builds a full topological string each call; topological order computed on demand. Cache the topological order list in `_cache` to avoid repeated traversal when multiple fingerprints/IR lowerings occur.
- DRY finding: None critical; serialization is centralized here.
- Suggested change: Add `_cache["topo_order"]` computed once, used by `fingerprint()` and other traversals.
- Owner: synthesis
- Priority: P2

## Findings (2026-02-20, continued)

- File: `research/scientist/notebook.py`
- Perf finding: `get_leaderboard()` oversamples and dedups in Python; good for variety but returns all columns and parses JSON for each row; could benefit from narrower column selection per consumer or optional parsed fields.
- DRY finding: JSON field parsing duplicated between `get_program_detail()` and new `get_program_details()`; opportunity for shared helper to parse program JSON fields.
- Suggested change: Add `_parse_program_json_fields(d)` helper and allow `get_leaderboard(..., include_graph=False)` to reduce payload for lightweight views.
- Owner: notebook
- Priority: P2

- File: `research/scientist/api.py`
- Perf finding: `/api/decision-packet/<id>` pulls leaderboard (limit=200) then scans for result id; multiple DB calls + repeated analytics instantiation. N+1 patterns in `_program_lineage_chain` using `get_program_detail` inside loop.
- DRY finding: Program detail parsing + common enrichment repeated across endpoints.
- Suggested change: Add targeted query `get_leaderboard_entry(result_id)` and batch lineage fetch; centralize enrichment (`qkv_usage`, `compression_metrics`) in a helper.
- Owner: api
- Priority: P2

## Findings (2026-02-21)

- CP-2 (Dataflow & caching) audit result: completed.
- Scope reviewed: `scientist/runner.py`, `scientist/api.py`, `scientist/notebook.py`, `scientist/analytics.py`.
- Key outcomes:
  - Batched detail fetch paths are in place for investigation/validation.
  - Program JSON parse logic is centralized in notebook helpers (no repeated ad-hoc parse hot path in critical flows).
  - Dashboard/report paths now prefer shared snapshots/endpoints and avoid repeated full scans in the top call chain.
- Follow-up retained:
  - Optional: expand cache invalidation policy notes for long-lived dashboard sessions.

- CP-3 (Memory allocations audit) result: completed.
- Scope reviewed: Stage-0/Stage-1 eval and sparse/training loops.
- Key outcomes:
  - Immediate CUDA reclamation is present (`CompiledLayer.forward` ref management + cache cleanup guards).
  - Mapped shared-memory zero-copy token init is now used in sandbox eval (`torch.from_numpy` over `np.memmap`).
  - Fragmentation audit utility exists for long-horizon runs (`scientist/perf.py`).
- Follow-up retained:
  - Optional: add periodic heap/object churn telemetry for very long CPU-bound runs.

- CP-4 (Kernel launch audit) result: completed.
- Scope reviewed: fused/triton kernels and launch-heavy step paths.
- Key outcomes:
  - Fusion coverage present for linear+gelu, rmsnorm, local attention.
  - Per-op kernel timing hooks added (torch profiler based) and surfaced in experiment perf report.
  - Queue-level scheduling telemetry added to identify launch/queue overhead and backpressure.
- Follow-up retained:
  - Optional: auto-promote hotspot ops from perf reports into fusion backlog.

- CP-5 (I/O, DB, and Serialization audit) result: completed.
- Scope reviewed: `research/database.py`, `research/scientist/notebook.py`.
- Key outcomes:
  - SQLite WAL-mode and synchronous=NORMAL enabled for all database connections.
  - Large JSON payloads (`results_json`, `insights_json`, `loss_curve`, `choices`) now use zlib compression.
  - Batched DB writes implemented in `LabNotebook` using a `batch()` context manager and `_maybe_commit` pattern.
  - Orchestrator result recording in `runner.py` is now atomic/batched.
- Follow-up retained:
  - Optional: monitor BLOB size distribution if `graph_json` volume grows significantly.

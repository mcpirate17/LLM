# Task: Rewrite LabNotebook SQLite Layer — Native-First

## STATUS: COMPLETED (2026-04-13)

**Fix:** Built `aria_db` Rust/PyO3 module — single WAL connection that never closes.
**Result:** 31-minute clean run, 66 screening passes, 0 errors, 2 experiments completed, 8 S1 survivors.
**Files:** `research/runtime/native/rust/aria-db/`, `research/scientist/notebook/native_conn.py`, `research/scientist/notebook/shared_conn.py`
**Build:** `cd research/runtime/native/rust/aria-db && maturin develop --release`

---

## Context

The `LabNotebook` class (`research/scientist/notebook/`) is the central data store for the neural architecture search pipeline. It wraps a 567MB SQLite database (`lab_notebook.db`) with 58 tables, 143 public methods, and ~1.5M rows in the largest table (`training_curves`).

**The current implementation is broken.** Continuous runs fail within seconds due to a SQLite WAL/SHM corruption bug that has resisted every fix attempt. The system cannot run experiments until this is resolved.

## The Bug (What You're Fixing)

**Root cause:** SQLite WAL mode requires a shared memory file (`-shm`) managed via `mmap()`. In the Flask dashboard process, the SHM gets deleted mid-run, corrupting the process-wide mmap state. Every subsequent `sqlite3.connect()` call in the process then fails with `SQLITE_IOERR_SHORT_READ` (extended error code 522). The only recovery is killing the process.

**Why it happens:** 30+ code paths across `intelligence/`, `analytics/`, and `runner/` open raw `sqlite3.connect()` calls to `lab_notebook.db`. When any of these connections closes as the last WAL reader, SQLite auto-checkpoints and deletes the SHM. The `wal_autocheckpoint=0` pragma was tried on every connection (including a global monkey-patch of `sqlite3.connect`) but the SHM still gets deleted — the deletion mechanism isn't the auto-checkpoint, it's the connection close cleanup, which has no pragma to disable it.

**Why DELETE mode doesn't work either:** Without WAL, SQLite uses file-level exclusive locking. The LabNotebook has a background writer thread (`_writer_loop`) that holds transactions open. In DELETE mode, this blocks all reader threads, starving the screening loop to ~3% GPU utilization (vs normal 80%+). A deadlock between the writer and reader threads makes it completely stuck.

**What was tried and failed:**
- `wal_autocheckpoint=0` on all managed connections — SHM still deleted
- Global monkey-patch of `sqlite3.connect` to set `wal_autocheckpoint=0` — SHM still deleted
- Keepalive connection to prevent SHM teardown — SHM still deleted
- Orphaned WAL file repair — process mmap state already corrupted by then
- DELETE journal mode — deadlocks between writer and reader threads
- DELETE mode + autocommit writer (`isolation_level=None`) — no deadlock but 3% GPU utilization from lock contention
- Connection recovery (detect poisoned connection, reconnect) — can't reconnect because DB header says WAL, every new connection tries WAL init, fails because SHM is gone

## What You Need to Build

A new SQLite connection manager that replaces the current `LabNotebook` connection handling. The existing API (143 methods) must be preserved — only the connection/threading/locking layer changes.

### Architecture Requirements

1. **Native-first.** The connection manager should be in Rust (PyO3) or C (pybind11). Python's `sqlite3` module is the problem — its WAL/SHM handling is broken in multi-threaded Flask processes. A native module can:
   - Use SQLite's C API directly with `SQLITE_OPEN_NOMUTEX` or `SQLITE_OPEN_FULLMUTEX`
   - Control the VFS layer to prevent SHM deletion
   - Use `sqlite3_wal_hook()` to intercept checkpoints
   - Disable the problematic `unlink()` of SHM files entirely

2. **Single connection, single thread.** The current architecture (multiple LabNotebook instances each with their own writer thread, all competing for the same DB) is the root cause of both the WAL corruption and the DELETE deadlock. The replacement should:
   - Use ONE connection to `lab_notebook.db` per process, shared by all threads
   - Serialize writes through a single writer (channel/queue, not a Python thread holding a transaction open)
   - Allow concurrent reads via WAL mode (reads don't need locks in WAL)
   - Never close the connection during the process lifetime (prevents SHM teardown)

3. **All 30+ raw `sqlite3.connect()` calls must go through the new manager.** The intelligence, analytics, and runner modules currently open their own connections. These are the ones that trigger SHM deletion. They need to use the shared connection or a connection pool that the manager controls.

### Files to Scan (Current Implementation)

The notebook is a mixin-based class split across these files:

```
research/scientist/notebook/
├── __init__.py            # LabNotebook composition + re-exports
├── _shared.py             # ExperimentEntry, schema constants, helpers
├── notebook_core.py       # __init__, _configure_sqlite_connection, _writer_loop, migrations
├── notebook_experiments.py # start/complete/fail/cancel experiment, _direct_db_conn
├── notebook_advanced_analytics.py  # learning_log, failure analysis, report snapshots
├── notebook_healer.py     # healer tasks/events
├── notebook_knowledge.py  # knowledge base, insights, digests
├── notebook_leaderboard.py # leaderboard CRUD, composite scoring
├── notebook_misc.py       # chat, metrics, op stats, misc queries
├── notebook_programs.py   # program_results, training curves, graph features
├── program_query_views.py # read-only leaderboard/program queries
└── graph_features.py      # graph feature extraction (calls Rust scheduler)
```

**Critical internals to preserve:**
- `_submit_write(sql, params)` → async write queue
- `flush_writes()` → drain queue, wait for commit
- `_ThreadSafeConnectionWrapper` → thread-safe connection access
- `_direct_db_conn()` → fallback connection for critical lifecycle writes
- `batch()` context manager → group writes
- Schema migration system (`_migrated_paths`, `_ensure_schema_bootstrap`)
- `ExperimentEntry` dataclass in `_shared.py`

**Raw `sqlite3.connect()` call sites to redirect (grep for these):**
```
research/scientist/intelligence/predictor.py (2 calls)
research/scientist/intelligence/temporal_bayesian.py (1)
research/scientist/intelligence/ml_corpus.py (7)
research/scientist/intelligence/graph_segments.py (1)
research/scientist/intelligence/interaction_model.py (2)
research/scientist/intelligence/interaction_analysis.py (2)
research/scientist/intelligence/op_embeddings.py (2)
research/scientist/intelligence/gnn_predictor.py (2)
research/scientist/analytics/model_strength.py (1)
research/scientist/runner/control_actions.py (1 — vacuum)
research/scientist/runner/_helpers.py (1)
research/scientist/api_routes/_observability_core.py (1 — profiling_db, different file)
research/scientist/runtime_events/bootstrap.py (1 — projector connection)
```

Note: some of these connect to `profiling_db` (a different database), not `lab_notebook.db`. Only the `lab_notebook.db` connections need to go through the shared manager. The `profiling_db` connections are fine as-is.

### Database Stats

- 58 tables
- 567 MB total
- Largest: `training_curves` (1.58M rows), `program_graph_pairs` (179K rows), `program_graph_ops` (139K rows)
- Hot tables during screening: `program_results` (19K), `leaderboard` (2.6K), `experiments` (1.1K), `learning_log` (2.6K)

### Event Bus Integration

The event bus (`research/scientist/runtime_events/`) has its own `projector_conn` that writes to `lab_notebook.db`. This connection must also go through the shared manager. Currently in `bootstrap.py:65`.

## Implementation Plan

1. **Scan:** Read every file listed above. Document every public method, every SQL query, every `sqlite3.connect()` call. Build a complete API surface map.

2. **Build the native connection manager:**
   - Rust (PyO3) preferred, C (pybind11) acceptable
   - Single `ConnectionManager` class exposed to Python
   - Methods: `execute(sql, params)`, `executemany(sql, params_list)`, `fetchone(sql, params)`, `fetchall(sql, params)`, `submit_write(sql, params)`, `flush_writes()`, `close()`
   - Internal: single WAL connection, never closes, no SHM teardown possible
   - Thread-safe: reads can happen from any thread concurrently (WAL allows this), writes are serialized through an internal channel

3. **Build `LabNotebook2`** (new name, don't touch old one yet):
   - Same mixin structure, same 143 public methods
   - Replace `self.conn` and `_submit_write` with the native connection manager
   - Remove `_writer_loop`, `_ThreadSafeConnectionWrapper`, `_direct_db_conn`
   - Remove all `PRAGMA journal_mode` / `wal_autocheckpoint` / WAL recovery code

4. **Redirect raw `sqlite3.connect()` calls:**
   - For `lab_notebook.db` connections: use `ConnectionManager.get_reader()` or similar
   - For `profiling_db` connections: leave as-is (different database, no WAL issues)

5. **Test:**
   - All 71 existing notebook tests must pass with `LabNotebook2`
   - All 44 event bus tests must pass
   - Start a continuous run: must see screening progress (Rapid screening PASSED, routing fast lane) within 30 seconds
   - Run for 10+ minutes without disk I/O errors
   - Verify GPU utilization >50% during screening

6. **Swap:**
   - Only after all tests pass and continuous run works
   - `LabNotebook = LabNotebook2` in `__init__.py`
   - Delete old mixin files only after verifying the swap works

7. **Monitor:**
   - Run continuous for 30+ minutes
   - Watch for: disk I/O errors, deadlocks (76 threads 0 running), segfaults, screening stalls
   - Fix any issues that arise

## What NOT to Change

- The 143 public method signatures — callers depend on them
- The database schema (58 tables) — data must be preserved
- The event bus modules (`runtime_events/`) — these are new and working
- The `_shared.py` file (ExperimentEntry, constants)
- Any runner/screening/synthesis code — only the notebook connection layer

## Key Files to Read First

1. `research/scientist/notebook/notebook_core.py` — the `__init__`, `_configure_sqlite_connection`, `_writer_loop`
2. `research/scientist/notebook/notebook_experiments.py` — `start/complete/fail_experiment`, `_direct_db_conn`
3. `research/scientist/runtime_events/bootstrap.py` — projector connection
4. `research/CLAUDE.md` — project standards (correct > minimal > fast, Rust/C for hot paths)
5. `research/scientist/notebook/__init__.py` — mixin composition

## Environment

```bash
source /home/tim/venvs/llm/bin/activate
cd /home/tim/Projects/LLM
pytest research/tests/test_notebook.py -x --tb=short -q    # notebook tests
pytest research/tests/test_runtime_events.py -x --tb=short -q  # event bus tests
python -m research --mode=dashboard --port 5000              # start dashboard
curl -X POST http://localhost:5000/api/experiments/start \
  -H "Content-Type: application/json" \
  -d '{"mode":"continuous","max_experiments":2,"max_time_minutes":5}'  # start run
```

## Success Criteria

1. Zero `disk I/O error` messages during continuous runs
2. Screening programs in <2 seconds each (not minutes)
3. GPU utilization >50% during screening
4. All 71 notebook tests + 44 event bus tests pass
5. 30+ minute continuous run without crashes, deadlocks, or I/O errors

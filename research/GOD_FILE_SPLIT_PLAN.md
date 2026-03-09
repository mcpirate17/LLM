# God File Split Plan

**Date**: 2026-03-08
**Goal**: Split files >2,000 lines into focused child modules. Create children first, then update parents to re-export, then archive originals.
**Constraint**: C/C++/Rust/Cython first. Python must use high-perf patterns (numpy, numba, `__slots__`, threading, caching, dispatch tables).

---

## Priority Order

| # | File | Lines | Language | Effort | Impact |
|---|------|-------|----------|--------|--------|
| 1 | `scientist/runner.py` | 15,795 | Python | **NONE** — already split, delete dead file | HIGH |
| 2 | `scientist/api.py` | 10,623 | Python | [DONE] | CRITICAL |
| 3 | `scientist/notebook.py` | 6,275 | Python | [DONE] | HIGH |
| 4 | `scientist/runner/execution.py` | 4,563 | Python | [DONE] | HIGH |
| 5 | `scientist/analytics.py` | 4,429 | Python | [DONE] | MEDIUM |
| 6 | `scientist/persona.py` | 3,346 | Python | [DONE] | MEDIUM |
| 7 | `scientist/runner/continuous.py` | 3,106 | Python | [DONE] | HIGH |
| 8 | `synthesis/compiler.py` | 2,229 | Python+C | [DONE] | MEDIUM |
| 9 | `scientist/native_runner.py` | 2,640 | Python+C+Rust | [DONE] | LOW |
| 10 | `scientist/runner/results.py` | 2,004 | Python | [DONE] | MEDIUM |
| 11 | `runtime/native/rust/executor.rs` | 1,976 | Rust | LOW | LOW (well-structured) |

---

## Phase 0: Dead Code Removal

### 0.1 Delete `scientist/runner.py` (15,795 lines)
- **Status**: Dead. `scientist/runner/__init__.py` is the active package.
- Python resolves `from .runner import ExperimentRunner` → `runner/__init__.py`.
- **Action**: `git rm scientist/runner.py` + verify no file references it by path.
- **Risk**: NONE — confirmed Python resolves to package.

### 0.2 Delete `scientist/api_routes/read.py.bak` (117,798 bytes)
- Backup file, no code references it.

---

## Phase 1: `scientist/api.py` → Flask Blueprints (10,623 → ~800 core + 12 blueprints)

Partial work exists in `scientist/api_routes/` (aria.py, control.py, designer.py, deps.py, frontend.py, read.py). Finish the migration.

### Strategy
1. **Create child blueprint modules** in `scientist/api_routes/`:

| Blueprint | Routes | Est. Lines | Source Lines |
|-----------|--------|------------|--------------|
| `analytics_bp.py` | 23 `/api/analytics/*` | ~600 | Cluster of pure-query routes |
| `experiments_bp.py` | 15 `/api/experiments/*` | ~500 | Experiment CRUD + lifecycle |
| `programs_bp.py` | 11 `/api/programs/*` | ~400 | Program eval, lineage, backfill |
| `leaderboard_bp.py` | 4 `/api/leaderboard/*`, `/api/discoveries` | ~200 | Leaderboard + pins |
| `native_bp.py` | 5 `/api/native-profile/*`, `/api/native-runner/*` | ~200 | Native profiling |
| `campaigns_bp.py` | 8 `/api/campaigns/*` | ~250 | Campaign lifecycle |
| `knowledge_bp.py` | 3 `/api/knowledge/*` | ~150 | Knowledge extraction |
| `actions_bp.py` | 4 `/api/actions/*` | ~120 | Action queue |
| `diagnostics_bp.py` | 2 `/api/diagnostics/*` | ~80 | Fingerprint + cache |
| `config_bp.py` | 3 `/api/config`, `/api/llm/config` | ~100 | Config endpoints |
| `events_bp.py` | SSE + progress + live-feed | ~200 | Server-sent events |
| `misc_bp.py` | Remaining ~33 routes | ~400 | Strategy, reports, status, etc. |

2. **Each blueprint** gets shared state via `deps.py` (already exists — holds notebook, runner, persona refs).
3. **`api.py` becomes a thin shell**: `create_app()` creates Flask app, registers 12 blueprints, sets up SSE/CORS/error handlers. Target: <800 lines.
4. **Archive**: `git mv api.py api_monolith.py.bak` (temporary, delete after tests pass).

### Performance rules for blueprints
- All query routes use `notebook.conn.execute()` directly — no ORM overhead
- Response serialization uses `json_utils.SafeJSONEncoder`
- Heavy analytics routes use `@functools.lru_cache` with TTL pattern
- SSE routes use generator-based streaming (already in place)

### Execution order
```
a) Extract analytics_bp.py (23 routes, most self-contained)
b) Extract experiments_bp.py (15 routes)
c) Extract programs_bp.py (11 routes)
d) Extract campaigns_bp.py + knowledge_bp.py + actions_bp.py (small, fast)
e) Extract native_bp.py + diagnostics_bp.py + config_bp.py
f) Extract events_bp.py (SSE plumbing)
g) Extract misc_bp.py (remaining routes)
h) Slim api.py to shell, register all blueprints
i) Delete api_monolith.py.bak
```

---

## Phase 2: `scientist/notebook.py` → Domain DAL Modules (6,275 → ~900 core + 7 modules)

### Strategy
`LabNotebook` is a 157-method god class. All methods are SQLite queries. Split by domain into mixin classes, compose in `LabNotebook`.

| Module | Methods | Est. Lines | Domain |
|--------|---------|------------|--------|
| `notebook_core.py` | `__init__`, `_migrate`, `_writer_loop`, `_submit_write`, `flush_writes`, `close`, `batch`, `_maybe_commit`, `_sanitize_numeric`, `_compress`, `_decompress` | ~400 | Core infrastructure |
| `notebook_experiments.py` | `start_experiment`, `complete_experiment`, `get_resumable_experiment`, `cleanup_stale_experiments`, preregistration methods | ~500 | Experiment lifecycle |
| `notebook_programs.py` | `record_program_result`, `add_entry`, `get_entry`, `purge_junk_programs`, fingerprint methods | ~600 | Program results |
| `notebook_leaderboard.py` | `upsert_leaderboard`, `promote_to_tier`, `get_tiers_for_result_ids`, scaling summary | ~500 | Leaderboard + tiers |
| `notebook_campaigns.py` | Campaign CRUD, hypothesis chain, decisions, selection decisions/insights | ~600 | Campaigns + hypotheses |
| `notebook_knowledge.py` | `add_knowledge`, `get_knowledge`, `search_knowledge`, `validate_knowledge`, digests | ~300 | Knowledge base |
| `notebook_healer.py` | `create_healer_task`, `update_healer_task`, `add_healer_event`, healer queries | ~200 | Self-repair tracking |
| `notebook_chat.py` | `save_chat_message`, `get_chat_history`, `mark_messages_compacted`, `compact_old_chat` | ~150 | Chat persistence |

### Composition pattern
```python
# notebook_core.py
class _NotebookCore:
    __slots__ = ('_conn', '_db_path', '_write_queue', '_writer_thread', ...)
    def __init__(self, db_path): ...
    def _migrate(self): ...

# notebook_experiments.py
class _ExperimentsMixin:
    """Requires self._conn, self._submit_write from _NotebookCore."""
    def start_experiment(self, ...): ...

# notebook.py (slim)
from .notebook_core import _NotebookCore
from .notebook_experiments import _ExperimentsMixin
from .notebook_programs import _ProgramsMixin
# ...

class LabNotebook(_NotebookCore, _ExperimentsMixin, _ProgramsMixin,
                  _LeaderboardMixin, _CampaignsMixin, _KnowledgeMixin,
                  _HealerMixin, _ChatMixin):
    """Full notebook — composed from domain mixins."""
    pass
```

### Performance rules
- All mixins use `self._submit_write()` for async writes (existing pattern)
- Query methods return raw `sqlite3.Row` dicts — no dataclass wrapping on hot paths
- `__slots__` on `_NotebookCore` for attribute access speed
- Mixin methods are plain `def` — no `@property` for query methods

---

## Phase 3: `scientist/runner/execution.py` → Extract God Functions (4,563 lines)

Three functions >500 lines need extraction:

### 3.1 `_run_validation_thread()` (1,014 lines → ~200 + 3 helpers)
Extract into:
- `_eval_candidate_batch()` — candidate evaluation loop
- `_aggregate_validation_results()` — result collection + scoring
- `_report_validation_progress()` — SSE progress reporting

### 3.2 `_execute_experiment()` (771 lines → ~200 + 3 helpers)
Extract into:
- `_run_screening_loop()` — screening phase iteration
- `_run_investigation_phase()` — investigation evaluation
- `_finalize_experiment()` — cleanup + result recording

### 3.3 `_micro_train()` (582 lines → ~150 + 2 helpers)
Extract into:
- `_setup_training()` — optimizer, scheduler, data prep
- `_training_step()` — single step with gradient clipping + loss tracking

### File organization
```
runner/
  execution.py          (slim: ~500 lines, orchestration only)
  _validation.py        (~400 lines, validation thread helpers)
  _training.py          (~350 lines, micro-train helpers)
  _screening.py         (existing, may absorb screening loop)
```

### Performance rules
- Training loop helpers use `torch.no_grad()` context managers
- Loss tracking uses pre-allocated numpy arrays, not Python lists
- Progress reporting batched (existing `_TRAINING_STEP_SSE_EVERY` constant)

---

## Phase 4: `scientist/analytics.py` (4,429 lines)

### Strategy
Split into focused analysis modules:

| Module | Est. Lines | Domain |
|--------|------------|--------|
| `analytics_core.py` | ~400 | `ExperimentAnalytics.__init__`, data loading, caching |
| `analytics_ops.py` | ~600 | Op success rates, failure patterns, grammar weight analysis |
| `analytics_trends.py` | ~500 | Learning trajectory, efficiency frontiers, regression detection |
| `analytics_strategy.py` | ~500 | Strategy backtesting, control comparison, experiment clustering |
| `analytics_insights.py` | ~400 | Insight generation, negative results, compression opportunities |

### Performance rules
- Heavy aggregations use `numpy` vectorized ops (already partially done)
- Result caching via `@functools.lru_cache` with notebook generation counter as cache key
- `__slots__` on `ExperimentAnalytics` class
- Pareto frontier computation uses scipy if available: `from scipy.spatial import ConvexHull`

---

## Phase 5: `scientist/persona.py` (3,346 lines)

### Strategy
Extract rule-based fallback methods into separate module:

| Module | Est. Lines | Domain |
|--------|------------|--------|
| `persona.py` | ~1,500 | Aria class: LLM methods, reactive responses, analysis |
| `persona_rules.py` | ~1,200 | All `_rule_based_*()` methods (mode recommendation, hypothesis, critique, summary) |
| `persona_llm.py` | ~400 | LLM backend management (`_get_llm`, `configure_llm`, `_track_cost`) |

### `_rule_based_mode_recommendation()` (563 lines)
Split decision tree into named branches:
```python
# persona_rules.py
def _check_escalation_ready(stats) -> Optional[Recommendation]: ...
def _check_compression_guardrail(stats, config) -> Optional[Recommendation]: ...
def _check_sparsity_guardrail(stats, config) -> Optional[Recommendation]: ...
def _check_digest_overrides(stats, digest) -> Optional[Recommendation]: ...
def _data_driven_fallback(stats, op_rates) -> Recommendation: ...

def rule_based_mode_recommendation(stats, config, digest, op_rates) -> Recommendation:
    """Dispatch through decision tree branches in priority order."""
    for checker in [_check_escalation_ready, _check_compression_guardrail,
                    _check_sparsity_guardrail, _check_digest_overrides]:
        result = checker(stats, config)  # each takes relevant args
        if result:
            return result
    return _data_driven_fallback(stats, op_rates)
```

---

## Phase 6: `synthesis/compiler.py` (2,229 lines)

### 6.1 `_init_params()` → Dispatch Table (285 lines → ~30 + table)
```python
# compiler_init.py — param initializer registry
from typing import Callable, Dict
import torch.nn as nn

InitFn = Callable[['CompiledOp', 'OpNode', dict, tuple], None]
_INIT_TABLE: Dict[str, InitFn] = {}

def register_init(op_name: str):
    def decorator(fn: InitFn):
        _INIT_TABLE[op_name] = fn
        return fn
    return decorator

@register_init("linear_proj")
def _init_linear_proj(self, op, config, input_shape):
    d_in = input_shape[-1]
    d_out = config.get('d_model', d_in)
    self.weight = nn.Parameter(torch.randn(d_out, d_in) * (d_in ** -0.5))
    self.bias = nn.Parameter(torch.zeros(d_out))

# ... 60 more @register_init entries
```

### 6.2 Hot-path: `forward()` already dispatches to C via `_native_wrapper`
- Verify overhead: if `_native_wrapper` is None, `_execute_op()` fallback should be tight
- Consider: `__slots__` on `CompiledOp` (check no dynamic attrs)

---

## Phase 7: Runner Mixin Files (~2,000-3,000 lines each)

### `runner/continuous.py` (3,106 lines)
- Extract `_run_inline_validation()` (1,025 lines) into `runner/_inline_validation.py`
- Keep orchestration in `continuous.py`

### `runner/results.py` (2,004 lines)
- Extract `_auto_escalate()` (487 lines) into `runner/_escalation.py`
- Keep result recording in `results.py`

---

## Phase 8: Native Code (C/Rust/Cython) — Already Well-Structured

| File | Lines | Status |
|------|-------|--------|
| `runtime/native/src/runner_abi.c` | 1,234 | OK — single concern (ABI entry) |
| `runtime/native/rust/executor.rs` | 1,976 | OK — could split backward pass to `backward.rs` if it grows |
| `runtime/native/cython/aria_bridge.pyx` | 991 | OK — auto-generates to 37K C lines |

No native splits needed now. The Rust executor could benefit from extracting the backward pass (~400 lines) into `backward.rs` if it grows past 2,500 lines.

---

## Execution Rules

1. **Create children first** — never modify the parent until all children compile (`python -m py_compile`)
2. **Parent becomes re-export shim** — imports from children, re-exports public API. Zero behavior change.
3. **Archive originals** — `git mv old.py old.py.bak`, delete after integration tests pass
4. **One phase at a time** — complete each phase before starting the next
5. **Claim in `.current_work.md`** before starting any phase
6. **No pytest until all phases complete** — only `python -m py_compile` per file

---

## Impact Summary

| Metric | Before | After |
|--------|--------|-------|
| Files >2,000 lines (Python) | 10 | 2 (native_runner.py, executor.rs) |
| Largest Python file | 10,623 (api.py) | ~900 (notebook.py core) |
| Dead code removed | 0 | 15,795 (runner.py) + 117KB (read.py.bak) |
| God functions >500 lines | 7 | 0 (all decomposed into <300-line helpers) |

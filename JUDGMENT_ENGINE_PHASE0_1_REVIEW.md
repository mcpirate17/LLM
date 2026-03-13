# Judgment Engine — Review & Implementation Plan

**Date:** 2026-03-12
**Agents:** claude-opus, codex

---

## Status Summary

| Phase | Status | Key Deliverable |
|---|---|---|
| 6 (Grammar Overhaul) | **COMPLETE** | 55+ motifs, 10 templates, C++ smoke test, grammar rewritten |
| 1 (Research Aggregates) | **COMPLETE** | 4 signal types in API, but signals underutilized downstream |
| 0 (Refinement Guardrails) | **85%** | Quality gate works, but mutations still mostly random |
| 2 (Judgment Engine) | **NOT STARTED** | `judgment.py` doesn't exist |
| 3-5 | **NOT STARTED** | Blocked on Phase 2 |

---

## What's Done Well

### Phase 6 — Grammar Overhaul (all wired and integrated)
- `synthesis/op_roles.py`: 111 ops → 10 roles with valid-successor transition matrix
- `synthesis/motifs.py`: 55+ motifs mined from top performers, each with lift/support stats
- `synthesis/templates.py`: 10 composable templates (residual, transformer, bottleneck, MoE, etc.), old EXOTIC_TEMPLATES deleted
- `synthesis/grammar.py`: `generate_layer_graph()` composes 1-3 templates filled with motifs. Old `_choose_action`/`_pick_op` random-walk deleted
- `aria_core/src/cpu/smoke_test.cpp`: C++ BFS gradient-flow check, ~0.01ms, pybind11 bindings
- `runner/execution_training.py`: Calls `smoke_test_graph()` pre-training, rejects failures early
- `tools/_motif_mining.py` + `docs/motif_mining_report.md`: Reproducible pipeline with statistical backing

### Phase 1 — Research Aggregates (API complete)
- `notebook_analytics.py`: `get_op_pair_priors()`, `get_fingerprint_buckets()`, `get_lineage_successor_stats()`, `get_failure_risk_signatures()`
- `analytics_bp.py`: `/api/analytics/recommendation-signals` returns all 4 signal types + legacy fields
- Tests: `test_recommendation_aggregates.py` covers all 4 signal types

### Phase 0 — Refinement Guardrails (infrastructure done)
- `intent_parser.py`: `IntentConstraints` dataclass, 5 presets, component taxonomy, parent-tier guardrails
- `mutation.py`: `refine_winner()` accepts `intent` + `parent_scores`, routes by intent keyword
- `main.py`: `refine_winner_endpoint()` fetches parent scores, generates 2x candidates, validates each with `_validate_proposal_quality()` (compilation + smoke test + regression check + op retention)
- `research_signals.py`: Thread-safe cached fetch, shared by main.py and router

### Dead Code Removed (2026-03-12)
- `routers/aria.py::post_refine_winner`: Stale duplicate — didn't pass intent or parent_scores. Main.py has the real endpoint.

---

## Gaps — Ranked by Impact

### GAP 1: Grammar ignores research signals (Phase 6.5) — HIGH
`batch_generate()` never converts `op_pair_priors` into `template_weights`/`motif_weights`. `GrammarConfig` fields exist but default to `{}`. The grammar generates from uniform distribution despite rich signal data.

In `execution_screening.py`, `op_weights` are set on the config but `template_weights` and `motif_weights` are never populated.

### GAP 2: Mutations don't use parent scores directionally (Phase 0.1) — HIGH
Parent scores are fetched and passed to `refine_winner()` but only appended to rationale string. `_build_param_mutation()` picks random nodes with random deltas — no direction from loss_ratio. `_build_replacement_mutation()` picks random ops — no consultation of op_priors.

### GAP 3: No judgment engine (Phase 2) — MEDIUM
`research/scientist/judgment.py` doesn't exist. No `score_candidate()`, no `recommend_components()`, no signal composition. Phases 3-5 are blocked.

### GAP 4: No screening prioritization (Phase 4.2) — MEDIUM
`execution_screening.py` processes candidates in generation order. No ranking by predicted success. All candidates enter Stage 0 immediately.

### GAP 5: No grammar pass-rate tests — MEDIUM
Plan requires ≥70% smoke pass, ≥40% training pass, ≥90 unique fingerprints per 100 seeds. No such test exists.

### GAP 6: Live data quality issues (from prior review)
- Op-pair priors surface generic glue patterns (e.g., `rmsnorm_pre→add`) — need stronger filtering toward informative motifs from high-tier survivors
- Fingerprint buckets too coarse — most graphs collapse into `exotic`
- Lineage successor stats return no rows on live DB — `designer_run_lineage` table may not be populated consistently
- Failure-risk signatures return nothing — positive-evidence requirement is correct but currently too conservative

---

## Implementation Plan

### Task Assignment

| Task | Agent | Lines | Dependencies |
|---|---|---|---|
| T1: Wire signals into grammar | **codex** | ~80 | None |
| T2: Signal-driven mutations | **codex** | ~120 | None |
| T3: Grammar pass-rate test | **claude-opus** | ~100 | None |
| T4: Judgment engine core | **claude-opus** | ~250 | Phase 1 (done) |
| T5: Screening prioritization | **claude-opus** | ~60 | T4 |

---

### T1: Wire research signals into grammar (codex)

**Files:** `research/scientist/runner/execution_screening.py`, `research/synthesis/grammar.py`

**What to do:**
1. In `execution_screening.py`, where grammar config is built (~line 195-225), after computing `op_weights`:
   - Call `nb.get_op_pair_priors(min_support=5, limit=50)` (already available on notebook instance)
   - Convert top op-pair success rates into `motif_weights`: for each motif in `VALIDATED_MOTIFS`, check if any of its op-pair subsequences appear in the priors with success_rate > 0.3. Weight = sum of matching pair success_rates
   - Convert fingerprint bucket top-ops into `template_weights`: boost templates whose slot constraints match the dominant bucket (e.g., if `attention-heavy` bucket dominates, boost `tpl_transformer_block`)
2. Pass `template_weights` and `motif_weights` to `GrammarConfig` construction
3. In `grammar.py:generate_layer_graph()`, the weights already flow to `pick_template()` and `pick_motif()` via their weight params — just confirm they're non-empty

**Constraints:**
- No new files — modify existing code only
- Use `lru_cache` or a module-level cache on the signal-to-weight conversion (signals change slowly, avoid per-candidate recomputation)
- If signals unavailable, fall back to uniform weights silently
- Import `VALIDATED_MOTIFS` from `synthesis.motifs` — don't redefine motif metadata
- No duplication of analytics code — use existing `get_op_pair_priors()`

---

### T2: Signal-driven mutations in refine-winner (codex)

**File:** `aria_designer/api/app/mutation.py`

**What to do:**
1. `_build_param_mutation()`: Use `parent_scores["loss_ratio"]` + `intent_constraints.param_direction` to set delta direction and magnitude. If parent loss_ratio < 0.3 (poor), allow larger deltas (±30%). If loss_ratio > 0.7 (good), cap at ±10%. Delete the pure-random delta logic
2. `_build_replacement_mutation()`: Fetch research signals via `fetch_research_recommendation_signals()` (from `research_signals.py`, already cached). When picking replacement op, prefer ops from `op_pair_priors` that pair well with the target node's neighbors. Fall back to random if no signals
3. `_build_add_layer_mutation()`: Before randomly cloning a node, check if adding any op from `op_pair_priors` would form a high-success pair with an adjacent op. Prefer the highest-success pairing

**Constraints:**
- Import `fetch_research_recommendation_signals` from `.research_signals` (already exists)
- `__slots__` on any new classes
- No new files
- Must still work when research API unreachable (empty signals → current random behavior)
- Keep function lengths under 100 lines

**Verification:** `aria_designer/tests/test_api.py` must pass. Mutation output should vary by intent.

---

### T3: Grammar pass-rate validation test (claude-opus)

**File:** New: `research/tests/test_grammar_pass_rate.py`

**What to do:**
1. Generate 100 graphs with `batch_generate(100, grammar)` using default `GrammarConfig`
2. Run `smoke_test_graph()` from `aria_core` on each (Python structural fallback if aria_core not built)
3. For each smoke-passing graph, attempt `compile_model()` + 10-step micro-train on tiny data (`d_model=64, seq_len=16, batch_size=2`)
4. Assert: ≥70% pass smoke test
5. Assert: ≥40% compile + train without crash/NaN
6. Assert: ≥90 unique fingerprints via `compute_fingerprint()`
7. Mark `@pytest.mark.slow` — not in CI, run manually

**Constraints:**
- 60-second timeout per graph, 5-minute total test timeout
- No new dependencies
- Use existing `compute_fingerprint`, `compile_model`, `safe_eval` from synthesis pipeline

---

### T4: Judgment engine core (claude-opus)

**File:** New: `research/scientist/judgment.py`

**What to do:**
1. Dataclasses with `slots=True`:
   - `JudgmentContext`: fingerprint_bucket, active_op_pairs, parent_fingerprint, parent_scores, intent, matched_insights, novelty_context, performance_context
   - `JudgmentResult`: total_score, signal_breakdown (dict), confidence, risk_flags (list), recommended_action, evidence (list), support_counts (dict)
2. `_SIGNAL_SCORERS: Dict[str, Callable]` — 8 scorer functions registered by name:
   - `_score_op_priors`: per-op success rate from `op_success_rates` table
   - `_score_op_pairs`: pair success from `get_op_pair_priors()`
   - `_score_fingerprint_bucket`: per-bucket performance bonus/penalty
   - `_score_lineage`: boost patterns that historically improved parents
   - `_score_failure_risk`: graduated penalty from `get_failure_risk_signatures()`
   - `_score_insight_interactions`: synergistic pair boosting from insights table
   - `_score_novelty`: reward under-explored fingerprint regions
   - `_score_intent_alignment`: boost candidates matching stated intent
   - Each returns `(score_delta: float, confidence: float, evidence: list[dict])`
3. Two entry points calling the same internal `_run_scoring_pipeline()`:
   - `score_candidate(candidate_graph, ctx, signals) → JudgmentResult`
   - `recommend_components(ctx, signals, candidates) → list[tuple[candidate, JudgmentResult]]`
4. Safeguards:
   - Cap any single signal at ±30% of total
   - When support < threshold, weight toward neutral (0.5)
   - 15% exploration budget for under-sampled combinations
   - Novelty-aware tie-breaking

**Constraints:**
- Single file unless >1250 lines
- `__slots__` everywhere
- Dict dispatch, not if/elif
- Import from `notebook_analytics` for data — no reimplementation
- Lazy import torch/numpy
- Must be importable standalone (no circular deps)

---

### T5: Screening prioritization (claude-opus, after T4)

**File:** `research/scientist/runner/execution_screening.py`

**What to do:**
1. After `batch_generate()`, build `JudgmentContext` for each candidate graph
2. Call `judgment.score_candidate()` on each
3. Sort by `total_score` descending
4. Skip candidates with hard failure-risk flags (log skip reason to experiment metadata)
5. Reserve 15% of slots for low-support candidates regardless of score (exploration budget)
6. Process sorted candidates through existing Stage 0/1 pipeline

**Constraints:**
- Scoring must be <10ms per candidate (runs on 50-200 per batch)
- If judgment module import fails, silently fall back to current order
- No new files

---

## Execution Order

```
Parallel (Week 1):
  codex   → T1 (wire signals into grammar)
  codex   → T2 (signal-driven mutations)
  claude  → T3 (grammar pass-rate test)
  claude  → T4 (judgment engine)

Sequential (Week 2, after T4):
  claude  → T5 (screening prioritization)
```

T1, T2, T3, T4 have zero dependencies on each other — all run in parallel.
T5 imports `judgment.score_candidate()` from T4.

---

## Verification (after all tasks)

```bash
# Research tests (includes new grammar test)
cd /home/tim/Projects/LLM && python -m pytest research/tests/ -x --tb=short -q

# Designer tests
cd /home/tim/Projects/LLM/aria_designer && python -m pytest tests/ --ignore=tests/test_aria_features.py -x -q

# Grammar pass-rate (slow, manual)
cd /home/tim/Projects/LLM && python -m pytest research/tests/test_grammar_pass_rate.py -v --timeout=300

# aria_core build check
cd /home/tim/Projects/LLM/aria_core && python setup.py build_ext --inplace
```

## Success Criteria

1. `GrammarConfig.motif_weights` populated from research signals when DB has data
2. Refine-winner uses parent loss_ratio for direction, op_priors for replacement selection
3. Grammar test: ≥70% smoke pass, ≥40% train, ≥90 unique fingerprints over 100 seeds
4. `judgment.score_candidate()` returns `JudgmentResult` with signal breakdown + evidence
5. Screening candidates sorted by judgment score before training
6. All existing tests pass — zero regressions
7. Zero dead code, zero duplication, zero new Python fallbacks for native code

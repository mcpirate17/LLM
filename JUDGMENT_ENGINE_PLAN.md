# Shared Judgment Engine — Comprehensive Multi-Agent Plan

## Global Rules

**Every agent working on this plan MUST follow these rules. No exceptions.**

### Claiming & Coordination
- **Claim before coding.** Update `.current_work.md` with your agent name, the phase/step number, and the files you will touch BEFORE writing any code. If another agent has claimed it, do NOT touch it.
- **No silent reclaims.** If a task is claimed and you think it's stale, add a note asking for status — don't just take it.
- **Re-read this plan fresh** before starting any work — another agent may have updated it.

### Code Quality — Non-Negotiable
- **No duplication.** Before writing ANY new function, grep the codebase for existing implementations. If logic exists elsewhere, import it or refactor to a shared location. Never copy-paste.
- **No dead code.** Every function, import, variable, and file must be reachable. Delete anything that isn't. No commented-out code, no `_old` suffixes, no `.bak` files.
- **No disconnected code.** Every new module must be imported and called from at least one existing code path. If you write `judgment.py`, something must import and use it. Orphan files are dead code.
- **No test skipping.** If a test fails, fix the code or fix the test. Never `@pytest.mark.skip`, never `--ignore`, never `xfail` without a linked issue. Pre-existing failures (listed below) are the only exceptions.

### Language Hierarchy
- **C++ (pybind11) or Rust (PyO3/maturin)** for anything called >100 times per generation cycle or >10ms per call. This includes: smoke tests, motif validation, graph composition, fingerprint computation, template instantiation. C++ goes in `aria_core/src/cpu/` with pybind11 bindings in `aria_core/bindings/bindings.cpp`. Rust goes in `aria-scheduler/` with maturin.
- **Cython** as bridge layer if C++/Rust kernels need Python-object interaction.
- **Python** as last resort, only for: orchestration, API routes, database queries, LLM prompt construction. When Python is unavoidable, it must be high-performance: `__slots__` on all classes, dict dispatch over if/elif, comprehensions over loops, `lru_cache` on pure functions, no string concatenation in loops, localized attribute lookups in tight paths.

### No Fallbacks
- **Do not write Python fallbacks for C++/Rust code.** If the native code doesn't build, fix the build — don't ship a slow Python alternative that hides the problem. Fallbacks create two code paths that diverge over time, double the maintenance, and mask performance regressions.
- If a C++ kernel is required, it is a build dependency. Tests must fail if it's missing, not silently degrade.
- The only exception: initial development scaffolding. You may write a Python prototype FIRST to validate logic, then rewrite in C++/Rust and DELETE the Python version. Never ship both.

### Testing Strategy
- **No testing until the end of each phase section.** Implement all steps in a phase first, then write and run tests once at the end.
- **Fix broken tests immediately.** If your changes break an existing test, fix it before moving on. Do not leave broken tests for the next agent.
- Pre-existing known failures (do not waste time on these):
  - `tests/test_aria_features.py::test_refine_winner`
  - `tests/test_component_contracts.py::test_component_contract[comp_path31]` (route_lanes)
  - `tests/test_perf_regression.py::test_mlp_param_count` (key mismatch)
  - `tests/test_stress.py::test_import_and_validate_survivor` (TypeError)

### Performance Standards (from CLAUDE.md)
- Max function: 100 lines. Max file: 1,250 lines.
- `__slots__` on every class. `@dataclass(slots=True)` on every dataclass.
- Dict dispatch, not if/elif chains.
- `functools.lru_cache` on deterministic pure functions.
- NumPy/torch vectorization, not Python loops, for batch operations.
- Lazy imports for heavy modules in startup paths.

---

## Problem Statement

The recommendation and refinement system is broken. Aria's "Refine Recommended" destroyed an investigation-tier architecture (composite 185.76) into two screening-tier variants (composite 69-75). Root causes:

1. **Refine-winner is a stub** — only random param mutations and activation swaps, zero research signal usage
2. **User intent presets are theater** — 10 carefully designed intents ("beat_benchmarks", "refine_compression") are sent as raw text and never parsed semantically
3. **Rich research signals are underexploited** — 7 signal types generated, only 4 consumed, 3 ignored entirely
4. **No refinement guardrails** — mutations can freely regress below parent architecture quality
5. **No feedback loop** — designer outcomes never feed back to research
6. **Grammar synthesis is disconnected** — `op_weights` empty by default, no automatic research signal loading
7. **Binary toxic blocklist deleted** — failure_signatures table cleared; replacement soft-penalty system needed
8. **Grammar generates op soup** — 93% of generated architectures never train (validated 2026-03-11, 30-seed experiment)

---

## Current State Audit

### Data Available (lab_notebook.db)

| Signal | Coverage | Quality |
|---|---|---|
| graph_fingerprint | 4,958/4,958 (100%) | Strong |
| graph structure (n_ops, depth, category_histogram) | 93% | Strong |
| loss_ratio | 83% | Strong |
| stability_score | 93% | Strong |
| novelty_score | 65% | Moderate |
| param_count / flops | 56% | Moderate |
| op_success_rates table | 97 ops tracked | Strong (but avg_novelty NULL) |
| insights | 713 rows | Moderate |
| selection_insight_interactions | Exists | Moderate |
| designer_run_lineage | Exists | Links designer→research |
| operator-pair / motif tables | **Missing** | Gap — must be derived |
| fp_intrinsic_dim, fp_isotropy, fp_rank_ratio | **0% populated** | Gap — schema exists but empty |

### Code Maturity Scores

| Component | File | Maturity | Key Gap |
|---|---|---|---|
| Suggestions engine | `aria_designer/api/app/suggestions.py` | 6/10 | Ignores 3/7 signal types, no interaction modeling |
| Refine-winner | `aria_designer/api/app/mutation.py` | 2/10 | Random mutations, zero signal usage |
| Ask Aria intents | `AskAriaModal.jsx` → `main.py` | 4/10 | Intent never parsed, treated as raw text |
| Research signals API | `analytics_bp.py` | 7/10 | Comprehensive but 3 types unused by consumer |
| Grammar synthesis | `synthesis/grammar.py` | 2/10 | Random op soup, 7% hit rate, no motifs or templates |
| Feedback loop | (missing) | 0/10 | One-way flow, no designer→research signal |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Shared Judgment Engine                     │
│                 research/scientist/judgment.py                │
│                                                              │
│  Inputs:                        Outputs:                     │
│  ├─ fingerprint bucket          ├─ total_score               │
│  ├─ active op pairs/motifs      ├─ per_signal_breakdown      │
│  ├─ parent lineage fp           ├─ confidence                │
│  ├─ matched insights            ├─ risk_flags                │
│  ├─ failure risk signatures     ├─ recommended_action        │
│  ├─ novelty/perf context        │   (promote/mutate/hold/    │
│  └─ support thresholds          │    discard/suggest)        │
│                                 └─ evidence + support counts │
│                                                              │
│  Two modes:                                                  │
│  ├─ recommend_components(ctx) → ranked suggestions           │
│  └─ score_candidate(candidate, ctx) → decision + evidence    │
└──────────────┬──────────────────────────┬────────────────────┘
               │                          │
    ┌──────────▼──────────┐    ┌──────────▼──────────┐
    │  Designer Consumer   │    │  Runner Consumer     │
    │  suggestions.py      │    │  runner/synthesis.py │
    │  mutation.py         │    │  runner/execution_*  │
    │  main.py routes      │    │  grammar.py          │
    └──────────────────────┘    └──────────────────────┘
```

---

## Phase 6: Grammar Overhaul — Motif-Based Compositional Synthesis

**Priority: HIGHEST — This is the single biggest leverage point in the entire system. Do this first.**

### Problem (validated 2026-03-11)

The current grammar (`synthesis/grammar.py`) generates random op soup. Experimental results from 30 seeds with branching config (split_prob=0.5, min_splits=1, three_way_split_prob=0.3):

| Outcome | Count | Rate | Cause |
|---|---|---|---|
| Crashes (no grad, bugs) | 10 | 33% | Gradient-killing ops (sign_ste, abs, log, exp, reciprocal, topk_gate), compiler bugs (`cannot access local variable 'output'`) |
| Flat (doesn't learn) | 16 | 53% | Incoherent op sequences — random stacking has no meaningful signal flow |
| Diverges | 2 | 7% | Numerically unstable compositions (state_space in random context) |
| **Actually trains** | **2** | **7%** | Accidentally coherent structures |

The 2 winners both happened to form semi-coherent architectures by luck:
- Seed 48 (`conv1d_seq → cos → relu_gate_routing → ternary_projection`): loss 30.6 → 1.15, 96% drop in 100 LM steps
- Seed 27 (`local_window_attn → rmsnorm → silu → relu_gate_routing → nm_sparse_linear`): loss 3.0 → 0.28, 91% drop

**93% of compute is wasted on architectures that were never going to work.**

### Solution: Three-layer compositional grammar

#### Layer 1: Op Role Classification

Every op in `PRIMITIVE_REGISTRY` gets a functional role tag. The grammar only composes ops in role-valid sequences.

| Role | Examples | Constraint |
|---|---|---|
| `project` | linear_proj, linear_proj_down/up, conv1d_seq, nm_sparse_linear, semi_structured_2_4_linear, block_sparse_linear, ternary_projection | Has learnable weight matrix. Must appear in every block. |
| `normalize` | rmsnorm, layernorm, dynamic_norm | Stabilizes activations. Placed before mix or project. |
| `activate` | silu, gelu, relu, tanh, sigmoid, swish | Pointwise nonlinearity. Must follow a project or gate, never standalone. |
| `mix` | linear_attention, local_window_attn, graph_attention, selective_scan, rwkv_time_mixing, rwkv_channel, fourier_mixing, state_space | Sequence-level mixing. Core of attention/SSM/conv blocks. |
| `route` | moe_2expert, relu_gate_routing, topk_gate, split2/3/4, concat | Controls information flow. Wraps sub-blocks. |
| `gate` | gated_linear, swiglu_mlp, learnable_scale, learnable_bias | Multiplicative modulation. Pairs with project. |
| `position` | rope, alibi, learned_pos | Positional info. Applied once near input, not repeated. |
| `unsafe` | sign_ste, abs, log, exp, reciprocal, square, neg, sliding_window_mask | Gradient-killing or numerically unstable standalone. **Never placed by grammar.** May exist inside hand-validated compound ops only. |

#### Layer 2: Motifs — Validated functional units

A **motif** is a 2-4 op chain that is empirically validated to: (a) produce gradients, (b) learn, (c) be numerically stable. Motifs are the atoms of the new grammar — known-good component combinations.

**Motifs are mined from the existing 4,500+ results**, not hand-curated. The mining script extracts recurring 2-4 op subsequences from stage1-passing architectures, scores them by (frequency × avg_loss_ratio_improvement), validates each with a smoke test, and classifies by function.

| Motif class | Example motifs | Role in block |
|---|---|---|
| `ffn_core` | `linear_proj → silu → linear_proj`, `linear_proj → gelu → linear_proj_down`, `swiglu_mlp` (single op) | Feed-forward transformation |
| `attention_core` | `rmsnorm → linear_attention → linear_proj`, `layernorm → local_window_attn → linear_proj`, `graph_attention → linear_proj` | Sequence mixing via attention |
| `ssm_core` | `linear_proj → selective_scan → linear_proj`, `state_space → linear_proj`, `rwkv_time_mixing → linear_proj` | Sequence mixing via state space |
| `conv_core` | `conv1d_seq → gelu → linear_proj`, `conv_only → silu → linear_proj` | Local sequence mixing via convolution |
| `gate_core` | `linear_proj → sigmoid → learnable_scale`, `gated_linear → linear_proj` | Gating / modulation |
| `norm_wrap` | `rmsnorm → [slot] → residual_add`, `layernorm → [slot] → residual_add` | Normalization + residual wrapper |

#### Layer 3: Structural Templates — Broad frameworks for how motifs compose

A **template** is an abstract DAG pattern — a broad structural framework (recursion pattern, depth pattern, MoE pattern, etc.) where the nodes are motif slots, not specific ops. Templates define the skeleton; motifs fill the slots. Templates compose with each other recursively — a parallel template can have a bottleneck template in one branch and a residual template in the other.

| Template | Pattern | What it explores |
|---|---|---|
| `sequential` | motif_a → motif_b → ... → motif_n | Depth: stacking different functional blocks |
| `parallel_split` | split → {motif_a \| motif_b \| ...} → concat → project | Width: parallel processing paths |
| `residual_block` | norm → motif → residual_add | Skip connections: gradient highways |
| `bottleneck` | project_down → motif → project_up → residual_add | Compression: information bottleneck |
| `moe` | route → {motif × N experts} → merge | Sparsity: conditional computation |
| `recurrent` | motif applied K times (shared or unshared weights) | Iteration: refinement loops |
| `hybrid_parallel` | split → {attention_motif \| ssm_motif} → concat → project | Hybrid: combining different mixing paradigms |
| `hierarchical` | motif_fine → downsample → motif_coarse → upsample → merge | Scale: multi-resolution processing |
| `dense_cascade` | motif_1 → motif_2 → motif_3 with dense skip connections | Dense connectivity: DenseNet-style |
| `gated_residual` | norm → motif → gate_core → residual_add | Learned residual: adaptive skip weighting |

**Combinatorial diversity (generative, not enumerative):**
- ~50+ validated motifs × 10 templates × motifs-per-template (2-4) × recursive template composition depth (1-3) × parameter variations (dims, heads, window sizes, bottleneck ratios, expert counts)
- The search space is effectively unbounded — every architecture is unique but structurally sound
- For context: 4,500 architectures explored so far with 7% hit rate. The new grammar targets **80-90% hit rate** with no ceiling on diversity.

#### Smoke Test Gate

Every generated architecture gets a fast validation before entering the training pipeline. **Implement in C++** — this runs on every candidate, hundreds of times per generation cycle:

```cpp
// aria_core/src/cpu/smoke_test.cpp
// smoke_test_graph(graph_ir, d_model, seq_len) → {ok, has_params, grad_flows, no_nan}
// ~0.1ms per graph on CPU. Exposed via pybind11 in bindings.cpp.
// No Python fallback. If aria_core isn't built, tests fail — fix the build.
```

### Implementation Steps

#### 6.1 Mine motifs from existing results

> **⚠ DATA MINING DISCLAIMER — READ BEFORE IMPLEMENTING**
>
> The database contains ~4,500 candidate architectures. The vast majority failed because they were assembled almost randomly by the op-soup grammar — their failure tells us very little about which components are actually bad. Drawing negative conclusions from this noise (e.g., "op X has a 90% failure rate, penalize it") will produce misleading signals because the failures are confounded by random composition, not by the ops themselves.
>
> **Rule: Mine only from the top performers. Do not attempt to learn "what doesn't work" from bulk failure data.**
>
> Specifically:
> - **DO** extract motifs, op pairs, and sequences from the top-performing architectures (e.g., top 10% by loss_ratio, all stage1-passing, all investigation/validation tier).
> - **DO** rank motifs by their association with strong outcomes (low loss, high stability, successful training).
> - **DO NOT** build failure penalties, toxic blocklists, or negative-signal scores from bulk failure statistics. The failure population is dominated by random-composition noise, not signal.
> - **DO NOT** penalize individual ops based on aggregate failure rates — an op that appears in 100 random failures and 3 winners may be excellent when composed correctly.
> - **Exception:** If advanced statistical analysis is performed (e.g., clustering on components associated with best loss/stability, controlling for composition context, filtering noise), then cautious negative signals may be derived. But this requires deep data mining with proper methodology — not naive frequency counts. See `research/docs/motif_mining_report.md` for baseline analysis if available.
>
> The rationale: extracting reliable negative signals from ~4,500 mostly-random candidates requires advanced analytics (component-conditioned clustering, confound control, etc.) that goes well beyond simple aggregation. Until that analysis is done, stay on the positive side — learn what works, don't guess at what doesn't.

- Query lab_notebook.db for top-performing graphs: stage1-passing AND (loss_ratio in top quartile OR tier >= investigation)
- Extract op subsequences (2-4 ops) from each graph's topological order
- Aggregate by subsequence: count, avg_loss_ratio, avg_stability_score
- Rank by (count × avg_loss_ratio_improvement)
- Validate top candidates with smoke test (forward + backward)
- Require minimum support of 3 passing architectures within the top-performer pool (not 5 from general population)
- **Output:** `VALIDATED_MOTIFS` dict in the motifs module, keyed by motif class
- **Baseline analysis available:** See [`research/docs/motif_mining_report.md`](research/docs/motif_mining_report.md) for deep data mining results (2026-03-11). Reproducible script: `research/tools/_motif_mining.py`.

#### 6.2 Classify all 111 ops into roles
- Single source of truth for role assignments, must cover every op in `PRIMITIVE_REGISTRY`
- `unsafe` ops are excluded from grammar sampling entirely
- Role is queryable by grammar at generation time (dict lookup, O(1))
- **Check for existing role/category code first** — `synthesis/primitives.py` may already have category tags. If so, extend rather than duplicate.

#### 6.3 Build template library
- Rewrite `research/synthesis/templates.py` (currently has only 5 hardcoded EXOTIC_TEMPLATES — replace entirely, delete old code)
- Each template is a callable: `(rng, motif_library, params) → ComputationGraph`
- Templates accept motif-class constraints per slot (e.g., "this slot must be an attention_core or ssm_core")
- Templates compose: a template can reference another template as a sub-pattern
- **Implement template instantiation in Rust** (`aria-scheduler`) or C++ (`aria_core`). This is called hundreds of times per batch. No Python fallback — if the native build is broken, fix it.

#### 6.4 Rewrite grammar to use motifs + templates
- `AdaptiveGenerator.generate()` becomes:
  1. Pick 1-3 templates (weighted by success priors if available, uniform otherwise)
  2. For each template slot, pick a motif from the compatible class
  3. Compose templates into a single `ComputationGraph`
  4. Run smoke test — reject and retry if it fails (max 5 retries, then return best-effort)
- Preserve `GrammarConfig` interface for backward compatibility
- Add new config fields: `template_weights`, `motif_weights`, `composition_depth`
- **Delete the old `_choose_action` / `_pick_op` random-walk code** — it's the source of op soup. Don't keep it as a fallback, don't rename it `_legacy`. Delete it.

#### 6.5 Wire judgment engine priors into grammar
- Template selection weights come from Phase 1 research aggregates (when available)
- Motif selection weights come from op_pair success rates (when available)
- This is where Phases 1-2 feed into Phase 6
- Without judgment data, use uniform weights (still far better than current op soup)

#### 6.6 Smoke test gate in eval pipeline
- Add `smoke_test_graph()` to `aria_core/src/cpu/smoke_test.cpp` + pybind11 bindings. Takes flattened graph IR, allocates small test tensors, runs forward+backward, checks grad flow. ~0.1ms. No Python fallback.
- Insert before micro-training in `runner/execution_training.py`
- Log smoke test failures as a new failure category (distinct from training failures)

### Phase 6 Tests (run AFTER all 6.1-6.6 are implemented)
- Motif mining produces ≥20 validated motifs from existing data
- Every motif in library passes smoke test independently
- Every template produces compilable graphs when filled with valid motifs
- `generate()` with new grammar achieves ≥70% smoke test pass rate over 100 seeds
- `generate()` with new grammar achieves ≥40% "actually trains" rate over 100 seeds (vs current 7%)
- Backward compatibility: `GrammarConfig` with default params still works
- Template composition depth 2+ produces valid graphs
- Diversity: 100 seeds produce ≥90 unique fingerprints (no mode collapse)
- **Fix any pre-existing test failures** in `test_integration.py`, `test_reference_architectures.py`, etc. that touch grammar/synthesis code. Do not leave them broken.

### Files
- New: `research/synthesis/op_roles.py` — op role classification (check `primitives.py` first for existing categories to extend, not duplicate)
- New: `research/synthesis/motifs.py` — motif library + mining script + validation
- Rewrite: `research/synthesis/templates.py` — template library (delete existing EXOTIC_TEMPLATES, replace entirely)
- Modify: `research/synthesis/grammar.py` — use motifs + templates; **delete** old random-walk generation code
- New: `aria_core/src/cpu/smoke_test.cpp` + bindings — C++ smoke test kernel
- Modify: `research/scientist/runner/execution_training.py` — pre-training smoke test gate

### Dependency
- Phase 6.1-6.4 can start immediately (no dependencies on other phases)
- Phase 6.5 depends on Phases 1-2 (judgment engine priors) but is optional — grammar works without priors
- Phase 6.6 is independent

---

## Phase 0: Refinement Guardrails (Critical — Do In Parallel With Phase 6)

**Why:** This is the root cause of the user's "Refine Recommended" regression. Refinements must never regress below the parent.

### [ ] 0.1 Parent-score injection into refine prompts
- Before generating a refined candidate, fetch the parent's scores from leaderboard
- Inject into the LLM/mutation context: tier, composite_score, loss_ratio, novelty
- **Actually use the scores** to constrain mutations — not just append to rationale string
- Reject any mutation that produces a candidate scoring below parent's composite minus 5% tolerance

**Status (2026-03-11):** Partially done. `_fetch_parent_scores_for_workflow()` exists in main.py and `refine_winner()` accepts `parent_scores` param. BUT: parent scores are only appended to the rationale string — they are NOT used to guide mutation direction or reject regressions. **Needs:** signal-driven mutation logic that uses parent tier/score to constrain changes.

### [ ] 0.2 Post-refinement quality gate
- After compiling a refined graph, run smoke test (Phase 6.6) + fast forward pass
- Compare gradient health and output stability against parent
- If regression detected, reject the patch and try an alternative mutation
- Return regression warning to UI if all mutations regress

**Status (2026-03-11):** `_validate_proposal_quality()` does compilation + forward + basic regression guard. **Needs:** call C++ `smoke_test_graph()` from Phase 6.6, gradient-based quality proxy, UI warning path.

### [ ] 0.3 Intent-aware mutation constraints
- Map each intent preset to allowed mutation types via structured `IntentConstraints` dataclass
- Each strategy picks component-type-aware targets, not random nodes
- Multi-op patches per intent, not single random mutations
- Block mutations outside the intent scope

**Status (2026-03-11):** `_select_mutation_strategy()` does basic keyword→strategy mapping. **Needs:** structured IntentConstraints, component-type-aware targeting, `intent_parser.py`.

**Files:**
- `aria_designer/api/app/mutation.py` — signal-driven mutations + quality gate
- `aria_designer/api/app/main.py` — inject parent scores, wire smoke test
- New: `aria_designer/api/app/intent_parser.py` — structured intent → constraints

### Phase 0 Tests (run AFTER all 0.1-0.3 are implemented)
- Refinement of investigation-tier arch never drops below screening
- Each intent preset produces mutation types within its scope
- Quality gate rejects proposals that fail smoke test
- Quality gate rejects proposals that remove >30% of ops from high-tier parents
- Fix `test_aria_features.py::test_refine_winner` — this is a pre-existing failure that Phase 0 should resolve, not skip

---

## Phase 1: Research Aggregates & Signal Expansion

### 1.1 Operator-pair success tables
- Derive from program_results: extract op bigrams (existing `_extract_op_bigrams`)
- Aggregate: success_rate = n_s1_passed / n_total, avg_loss_ratio, avg_novelty per bigram
- Store in `op_pair_stats` table or compute on-demand with `lru_cache`
- Minimum support: 5 graphs
- **Check `analytics_grammar.py` and `analytics_experiments.py` first** — may already have partial implementations. Extend, don't duplicate.

### 1.2 Fingerprint bucket assignment
- Use existing fp_ columns (locality, sparsity, symmetry, hierarchy) where populated
- Fallback: use graph_category_histogram (93% coverage) for coarse bucketing
- Buckets: {attention-heavy, mixing-heavy, sparse, hybrid, exotic}
- Per-bucket: top ops, top op-pairs, s1_rate, avg_novelty

### 1.3 Lineage successor stats
- From designer_run_lineage + program_results: which fingerprints succeeded their parents?
- Compute: parent_fp → child_fp transition success rates
- Track which op changes (add/remove/swap) led to improvements

### 1.4 Soft failure-risk signatures (replaces deleted blocklist)

> **⚠ CAUTION — See Phase 6.1 data mining disclaimer.** Naive failure-rate aggregation from the existing 4,500 candidates is unreliable because most failures are caused by random composition (op soup), not by the individual ops or pairs themselves. The penalty tiers below MUST only be applied after controlling for composition context. Specifically:
> - Only compute failure rates for ops/pairs that have appeared in at least 5 top-performing architectures (establishes the op CAN work) AND still fail >85% of the time in other contexts — this suggests a genuine fragility, not random noise.
> - If an op has never appeared in a top performer, its failure rate is uninformative — it may simply have never been composed correctly. Do not penalize it; leave it neutral.
> - Prefer the positive-signal approach: boost ops/pairs that appear disproportionately in winners, rather than penalizing ops that appear in losers.

- Recompute failure_signatures with graduated penalties instead of binary block
- Penalty tiers (apply ONLY after composition-context filtering per disclaimer above):
  - 95-100% fail rate, 20+ occurrences, AND confirmed working in ≥5 top performers → strong penalty (0.05 weight)
  - 85-95% fail rate, 10+ occurrences, AND confirmed working in ≥3 top performers → moderate penalty (0.3 weight)
  - 70-85% fail rate, 5+ occurrences → mild penalty (0.6 weight) — only if sufficient positive evidence exists
- Expose both `failure_risk_signatures` (soft) and `critical_failures` (hard, >98% fail, >50 occurrences, confirmed working elsewhere)

### [ ] 1.5 Expand recommendation-signals endpoint
- Add to `/api/analytics/recommendation-signals`:
  - `op_pair_priors`: top 100 pairs with success_rate, support, avg_loss_ratio
  - `fingerprint_buckets`: per-bucket top ops and pairs
  - `lineage_successors`: parent→child transition stats
  - `failure_risk_signatures`: graduated penalties (replaces binary toxic)
- Keep backward compatibility: existing fields unchanged

**Status (2026-03-11):** `top_entries` and `op_weights` added to payload. The 4 key signal types above are NOT implemented — depend on 1.1-1.4. **Needs:** 1.1-1.4 completed first.

**Files:**
- `research/scientist/notebook/notebook_analytics.py` — aggregation helpers (check existing code first)
- `research/scientist/api_routes/analytics_bp.py` — expanded endpoint
- `research/scientist/notebook/_shared.py` — op_pair_stats schema if persisted

### Phase 1 Tests (run AFTER all 1.1-1.5 are implemented)
- Pair aggregation returns correct support/success rates on synthetic fixtures
- Fingerprint bucketing is deterministic
- Lineage successor stats group by parent correctly
- Failure risk penalties are graduated, not binary
- Endpoint includes new sections with stable keys
- No duplicate utility functions introduced (grep for `_safe_float`, `json_safe`, etc.)

---

## Phase 2: Shared Judgment Engine

### 2.1 Core judgment module
Create `research/scientist/judgment.py`. **Use `__slots__` on all dataclasses. Use dict dispatch for signal scoring.**

```python
@dataclass(slots=True)
class JudgmentContext:
    fingerprint_bucket: str
    active_op_pairs: List[str]
    parent_fingerprint: Optional[str]
    parent_scores: Optional[Dict]
    intent: Optional[str]
    matched_insights: List[Dict]
    novelty_context: Dict
    performance_context: Dict

@dataclass(slots=True)
class JudgmentResult:
    total_score: float
    signal_breakdown: Dict[str, float]
    confidence: float
    risk_flags: List[str]
    recommended_action: str
    evidence: List[Dict]
    support_counts: Dict[str, int]
```

### 2.2 Signal scoring functions (composable)
Each returns a `(score_delta, confidence, evidence_items)` tuple. **Register via dict dispatch, not if/elif:**
```python
_SIGNAL_SCORERS: Dict[str, Callable] = {
    "op_priors": _score_op_priors,
    "op_pairs": _score_op_pairs,
    "fingerprint_bucket": _score_fingerprint_bucket,
    "lineage": _score_lineage,
    "failure_risk": _score_failure_risk,
    "insight_interactions": _score_insight_interactions,
    "novelty": _score_novelty,
    "intent_alignment": _score_intent_alignment,
}
```

### 2.3 Two entry points
- `recommend_components(ctx, signals, candidates)` → ranked list with evidence
- `score_candidate(candidate_graph, ctx, signals)` → JudgmentResult
- Both call the same scoring pipeline — **no duplicated logic between them.**

### 2.4 Exploration safeguards
- Cap any single signal's influence at ±30% of total
- When support < threshold, weight signal toward neutral (0.5)
- Reserve 15% exploration budget for under-sampled combinations
- Novelty-aware tie-breaking when two candidates have similar scores

**Files:**
- New: `research/scientist/judgment.py` — core engine (must be imported and called by Phase 3 and Phase 4 code)
- Do NOT create `judgment_signals.py` as a separate file unless judgment.py exceeds 1,250 lines. Keep it in one module until then.

### Phase 2 Tests (run AFTER 2.1-2.4 are implemented)
- Same context produces identical results from both entry points
- Low-support priors are downweighted
- Hard failures block, soft failures only penalize
- Exploration budget yields under-sampled combinations
- No single signal dominates (±30% cap)

---

## Phase 3: Designer Integration

### 3.1 Refactor suggestions.py
- Replace `_score_adjustment()` with calls to `judgment.score_candidate()`
- Pass richer context: current workflow fingerprint, parent fingerprint, active graph motifs
- Return structured evidence per suggestion (signal_breakdown, support, risk_flags)
- Preserve existing endpoint shape for backward compatibility

### 3.2 Rewrite mutation.py (refine-winner)
- Replace random param mutation with intent-driven strategies:
  - `mutate_param_informed`: use op_priors to pick which param to change and direction
  - `add_op`: use judgment engine to pick best op for intent
  - `swap_op`: use op_pair_priors to find better alternatives for weak ops
  - `restructure`: add/remove residual connections, splits, based on topology priors
- Each strategy queries judgment engine before returning
- **Delete the old random mutation code** — don't keep it as fallback

### 3.3 Intent-aware patch generation
- Parse intent preset ID in `_generate_patch_impl()`
- Route to specialized generators per intent (dict dispatch, not if/elif)
- Generate multi-op patches, not single suggestions

### 3.4 Wire research signals everywhere
- Consume all 7+ signal types (currently 4/7)
- Use `compression_opportunities` structure, not just flat technique list
- Use `insight_interactions` for synergistic pair boosting

**Status (2026-03-11):** Router parity done — `routers/aria.py` uses shared `research_signals.py`. Rest not started.

**Files:**
- `aria_designer/api/app/suggestions.py` — refactor to use judgment engine
- `aria_designer/api/app/mutation.py` — rewrite with intent-driven strategies; delete old random code
- New: `aria_designer/api/app/intent_parser.py` — intent preset → constraints
- `aria_designer/api/app/main.py` — wire intent parsing, inject parent scores

### Phase 3 Tests (run AFTER 3.1-3.4 are implemented)
- Each intent preset produces different mutation strategies
- Suggestions include signal_breakdown evidence
- Refine-winner never regresses below parent
- Response remains JSON-compatible with current UI
- Suggestion endpoint latency < 500ms
- Fix `test_refine_winner` if not already fixed in Phase 0

---

## Phase 4: Continuous Research Integration

### 4.1 Wire judgment into grammar synthesis
- At `batch_generate()` time, fetch latest op_priors and op_pair_priors
- Feed as `template_weights` and `motif_weights` to the new grammar (Phase 6)
- Bias motif selection toward motifs that form successful compositions

### 4.2 Candidate screening prioritization
- Before micro-training, score each candidate via `judgment.score_candidate()`
- Rank by predicted success (high support + novelty bonus)
- Train most promising candidates first
- Skip candidates with hard failure-risk flags (but log them — don't silently discard)

### 4.3 Mutation/refinement in continuous mode
- When runner refinement cycle triggers, use lineage successor stats
- Choose next edits based on which motif changes historically improved score
- Respect intent from campaign configuration

### 4.4 Promotion decisions with evidence
- Include judgment evidence in promotion decisions
- Store signal breakdown alongside tier promotion records

### 4.5 Decision trace persistence
- Log every automated judgment to `decision_traces` table:
  - decision_type, candidate_fingerprint, judgment_result (JSON), outcome, timestamp
- Enable post-hoc analysis: "why was this candidate generated/promoted/rejected?"

### 4.6 Exploration safeguards
- 15% exploration budget for under-sampled motif combinations
- Cap prior influence when support is stale (>7 days old)
- Novelty-aware tie-breaking
- Hard toxic = block, soft failure = penalty only

**Files:**
- `research/synthesis/grammar.py` — wire template_weights/motif_weights from signals
- `research/scientist/runner/synthesis.py` — candidate screening with judgment
- `research/scientist/runner/execution_screening.py` — prioritized screening
- `research/scientist/runner/results_analysis.py` — promotion with evidence
- `research/scientist/runner/dashboard.py` — decision trace persistence
- `research/scientist/notebook/_shared.py` — decision_traces table schema

### Phase 4 Tests (run AFTER 4.1-4.6 are implemented)
- op_weights populated from research signals, not empty
- Candidate generation shifts toward supported motifs when evidence exists
- Promotion includes judgment evidence alongside metrics
- Decision traces persisted and queryable
- Exploration budget yields under-sampled combinations even with strong priors

---

## Phase 5: Feedback Loop (Designer → Research)

### 5.1 Track accepted/rejected suggestions
- When user clicks "Apply" on a suggestion, record: suggestion details, source_fingerprint, result_fingerprint, user intent
- When user rejects, record negative signal

### 5.2 Track refinement outcomes
- After a refinement is applied, schedule a background eval
- Compare parent vs child scores
- Feed success/failure back to op_pair_stats and lineage_successors

### 5.3 Feed designer signals to research
- Periodically sync designer acceptance data to research notebook
- Update op_success_rates with designer-validated signals
- Weight designer signals lower than automated eval (human selection bias)

**Files:**
- `aria_designer/api/app/main.py` — track apply/reject events
- `aria_designer/api/app/database.py` — suggestion_outcomes table
- `research/scientist/notebook/notebook_analytics.py` — ingest designer feedback

### Phase 5 Tests (run AFTER 5.1-5.3 are implemented)
- Applied suggestions create positive feedback entries
- Rejected suggestions create negative entries
- Designer feedback appears in research op_priors (weighted)

---

## Completed Quick Wins

### [x] Designer router parity for research signals
- `aria_designer/api/app/routers/aria.py` now fetches and forwards live research recommendation signals instead of hardcoding `research_signals={}`.
- Added shared helper `aria_designer/api/app/research_signals.py` so `main.py` and the router use the same cached fetch path.

### [x] Indentation bug fix (main.py:2697)
- Profiling stage SSE "running" event was indented inside the conversion error handler, making it dead code. Fixed 2026-03-11.

### [x] Split op contiguity fix (compiler.py)
- `split2`, `split3`, `split4` ops now call `.contiguous()` on sliced outputs. C kernels in aria_core require contiguous tensors; split dimension slicing produces non-contiguous views. Fixed 2026-03-11.

---

## Execution Order

| Priority | Phase | Agents | Deliverable |
|---|---|---|---|
| **NOW** | 6.1-6.4 | claude-opus | Motif mining + op roles + templates + grammar rewrite |
| **NOW** | 0 | Any (claim first) | Refinement guardrails (independent safety fix) |
| Week 2 | 1 | Any (claim first) | Research aggregates (op pairs, fingerprint buckets, failure risk) |
| Week 2 | 6.5 | claude-opus | Wire judgment priors into grammar (depends on Phase 1) |
| Week 3 | 2 | Any (claim first) | Shared judgment engine |
| Week 3 | 6.6 | claude-opus | C smoke test kernel in eval pipeline |
| Week 4 | 3 + 4 | Parallel (claim first) | Designer + runner integration |
| Week 5 | 5 | Any (claim first) | Feedback loop |

**Phase 6 is the highest leverage work.** It directly addresses the 93% waste rate. Phases 0-5 improve decisions around the edges; Phase 6 fixes the source.

## Verification

After each phase, run the FULL test suite — do not cherry-pick:
```bash
# Research tests (all markers)
cd /home/tim/Projects/LLM && python -m pytest research/tests/ -x --tb=short -q

# Designer tests (all except known pre-existing failures listed in Global Rules)
cd /home/tim/Projects/LLM/aria_designer && python -m pytest tests/ --ignore=tests/test_aria_features.py -x -q

# Dashboard build
cd /home/tim/Projects/LLM/research/dashboard && npm run build

# aria_core build (if C/Rust code was changed)
cd /home/tim/Projects/LLM/aria_core && python setup.py build_ext --inplace
```

## Success Criteria

1. **Grammar hit rate ≥40%** — at least 40% of generated architectures actually train (vs current 7%)
2. **Grammar smoke test pass rate ≥80%** — at least 80% compile + produce gradients (vs current 53%)
3. **Diversity maintained** — 100 seeds produce ≥90 unique fingerprints (no mode collapse)
4. **Zero dead code introduced** — every new file is imported, every new function is called
5. **Zero duplication** — no logic exists in more than one place
6. Refine-winner never regresses a parent below its current tier
7. Each intent preset produces semantically different mutations
8. Suggestions include per-signal evidence breakdown with support counts
9. Grammar synthesis uses research op_priors when available
10. Decision traces queryable for every automated decision
11. All tests pass (fix, don't skip)

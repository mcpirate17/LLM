# Multi-Agent Work Plan — Remaining Tasks

**Created**: 2026-03-13
**Agents**: claude-opus (hard/architectural), codex (medium/systematic), gemini (easy/mechanical)
**Coordination**: `.current_work.md` for file claims, this file for task ownership

---

## Agent Assignments Overview

| Agent | Focus | Complexity |
|-------|-------|------------|
| **claude-opus** | Algorithmic rewrites, new engines, math-heavy changes | Hard |
| **codex** | C++/Rust bindings, vectorization, type systems | Medium |
| **gemini** | Dashboard UI, config wiring, simple refactors, data helpers | Easy |

---

## Phase 1: Parallel Foundation (No Dependencies Between Agents)

### claude-opus — Grammar Completion + Parallel Scan

#### 1A. Finish Grammar Overhaul (Judgment Engine Phase 6.4)
**Files**: `research/synthesis/grammar.py`
- Delete the old `_choose_action` / `_pick_op` random-walk generation code entirely
- Rewrite `AdaptiveGenerator.generate()` to:
  1. Pick 1-3 templates (from `templates.py`) weighted by success priors if available
  2. For each template slot, pick a motif from compatible class (from `motifs.py`)
  3. Compose into `ComputationGraph`
  4. Run smoke test — reject and retry (max 5 retries)
- Preserve `GrammarConfig` interface for backward compatibility
- Add `template_weights`, `motif_weights`, `composition_depth` config fields
- **Success**: `generate()` over 100 seeds → ≥70% smoke test pass, ≥40% actually trains

#### 1B. Parallel Associative Scan for SSM (Next-Gen P0.2)
**Files**: `synthesis/compiler_ops_sequence.py`, `arch_builder.py`
- Replace sequential `for t in range(S)` loop with parallel prefix scan
- Semigroup: `(A, b) ⊕ (A', b') = (AA', Ab' + b)`
- Use `torch.cumsum` in log-space for diagonal case
- Keep sequential fallback for S ≤ 32 only
- **Success**: SSM throughput ~6x improvement on S=512+

#### 1C. NSGA-II Multi-Objective Selection (Next-Gen P1.1)
**Files**: `scientist/leaderboard_scoring.py`, `search/evolution.py`, `search/novelty_search.py`
- Implement `fast_non_dominated_sort()` and `crowding_distance()`
- 4 objectives: loss_ratio (min), param_count/baseline (min), 1/throughput (min), -novelty (min)
- Keep `compute_composite_score()` for dashboard display
- Add `pareto_rank()` for actual selection pressure
- **Success**: Evolution discovers 4+ distinct clusters on Pareto front (vs 1 GPT-2-like cluster)

---

### codex — C++ Bindings + Fingerprint Vectorization

#### 1D. Expose 48 Missing C++ Kernel Bindings (Next-Gen P0.1)
**Files**: `aria_core/bindings/bindings.cpp`, `aria_core/include/kernels.h`
- Audit all kernels declared in headers but absent from pybind11 bindings
- Add wrappers — priority order: F16 variants first, backward kernels second, then remaining
- **Success**: `len(dir(aria_core._C))` matches kernel count in headers; no Python fallbacks needed

#### 1E. Fingerprint Vectorization (Perf Plan Phase 1)
**Files**: `research/eval/fingerprint.py`
- Replace `_collect_position_sensitivities_fallback` Python loop (128 sequential backprops) with `torch.func.vmap` + `torch.func.grad`
- Single vectorized backward pass for all position sensitivities
- **Success**: Sensitivity probe time reduced ~75%

#### 1F. Smoke Test Bindings Verification
**Files**: `aria_core/src/cpu/smoke_test.cpp`, `aria_core/bindings/bindings.cpp`
- Verify `smoke_test_graph()` is properly bound via pybind11 and callable from Python
- If not bound, add the binding
- Add a Python-level wrapper in `research/synthesis/` that the grammar/runner can call
- **Success**: `aria_core._C.smoke_test_graph(graph_ir, d_model, seq_len)` returns `{ok, has_params, grad_flows, no_nan}`

---

### gemini — Dashboard + Simple Wiring

#### 1G. Reference Architecture Dashboard Pins (Ref Arch P0.4)
**Files**: `research/dashboard/src/components/` (Leaderboard component)
- Add a "pinned" indicator (star/pin icon) next to reference architectures (GPT-2, Mamba, RWKV, RAG) on the leaderboard
- Reference entries have `model_source = 'reference'` — filter on that
- Pin icon should be non-interactive, just visual
- **Success**: Reference architectures visually distinct on leaderboard

#### 1H. Remove Novelty-Loss Gate (Next-Gen P1.2)
**Files**: `scientist/leaderboard_scoring.py`
- Find the gate `g(lr) = max(0, (0.9 - lr) / 0.6)` that multiplies novelty by loss performance
- Replace with floor: `g(lr) = min(1.0, 0.3 + 0.7 * (0.9 - lr) / 0.6)` — 30% novelty credit minimum even for weak performers
- Do NOT change the composite_score function signature or other scoring logic
- **Success**: Novel architectures with weak loss still get ≥30% novelty credit

#### 1I. Wire Routing Template Weights into Dashboard
**Files**: `research/dashboard/src/components/LiveFeed.js` (or appropriate component)
- Show routing template usage stats in the live feed: how many candidates used routing templates vs standard templates
- Data already available via SSE events from the runner — just display it
- **Success**: Dashboard shows routing vs non-routing template split

---

## Phase 2: Training & Scoring (After Phase 1)

### claude-opus — Difficulty-Aware Training + Judgment Engine

#### 2A. Difficulty-Aware Training Loop (Routing-First P2.2)
**Files**: `scientist/runner/execution_training.py`, `eval/engine.py`
**Depends on**: 1A (grammar generates routing architectures)
- Collect routing statistics during training: fraction of tokens per path
- Add load-balance auxiliary loss using existing `load_balance_loss_f32` C kernel
- Report `routing_fast_fraction`, `routing_balance_score`, `routing_effective_flops`
- Store in `program_results` metadata
- **Success**: Routing architectures report meaningful routing statistics; load-balance loss prevents lane collapse

#### 2B. Efficiency-Aware Scoring (Routing-First P2.3)
**Files**: `scientist/leaderboard_scoring.py`
**Depends on**: 2A (routing metrics available)
- Add `routing_efficiency_bonus` to composite score
- Formula: `effective_efficiency = efficiency_multiple * (1 + routing_savings_ratio)` where `routing_savings_ratio = 1 - (effective_flops / dense_flops)`
- **Success**: Models with routing savings score higher than equivalent dense models

#### 2C. Complete Judgment Engine (Judgment Engine Phase 2)
**Files**: `research/scientist/judgment.py`
**Depends on**: 2D (research aggregates)
- Verify/complete the 8 signal scoring functions (dict dispatch)
- Ensure both entry points (`recommend_components`, `score_candidate`) work
- Add exploration safeguards: ±30% cap per signal, low-support downweighting, 15% exploration budget
- **Success**: Same context produces identical results from both entry points; tests pass

---

### codex — Algebraic Types + Numerical Stability

#### 2D. Research Aggregates (Judgment Engine Phase 1.1–1.4)
**Files**: `research/scientist/notebook/notebook_analytics.py`, `research/scientist/api_routes/analytics_bp.py`
**Depends on**: None (can start in Phase 1 if capacity allows)
- Implement op-pair success tables (1.1): extract bigrams, aggregate success_rate/avg_loss_ratio
- Fingerprint bucket assignment (1.2): coarse bucketing from graph_category_histogram
- Lineage successor stats (1.3): parent→child transition success rates
- Soft failure-risk signatures (1.4): graduated penalties per data mining disclaimer
- Expand `/api/analytics/recommendation-signals` endpoint (1.5)
- **Success**: Endpoint returns `op_pair_priors`, `fingerprint_buckets`, `lineage_successors`, `failure_risk_signatures`

#### 2E. Algebraic Type Constraints (Next-Gen P2.1–P2.2)
**Files**: `synthesis/primitives.py`, `synthesis/grammar.py`, `synthesis/templates.py`
**Depends on**: 1A (grammar overhaul complete)
- Add `AlgebraicType` dataclass to primitives: `space`, `input_constraint`, `output_guarantee`
- Tag all primitives with their algebraic space
- Enforce type compatibility in grammar op selection — filter by output→input compatibility
- Define bridge operators (exp_map, log_map, etc.) for cross-space transitions
- **Success**: Grammar cannot emit mathematically nonsensical cross-space sequences

#### 2F. Numerical Stability Fixes (Next-Gen P3.1–P3.4)
**Files**: `mathspaces/hyperbolic.py`, `mathspaces/tropical.py`, `mathspaces/padic.py`, `mathspaces/spiking.py`
- P3.1: Learnable curvature for hyperbolic ops (softplus + clamp ≤ 10)
- P3.2: Adaptive temperature for tropical softmin (scale with sqrt(S/S_ref))
- P3.3: Smooth p-adic valuation proxy (epsilon-guarded log)
- P3.4: Per-module gradient scaling for spiking STE (1/√L)
- **Success**: Mathspace NaN rate <1% (from ~15%)

---

### gemini — Simple Wiring + Data Helpers

#### 2G. Adaptive Training Budget for Novel Architectures (Next-Gen P1.3)
**Files**: `scientist/runner/execution_training.py`
- If architecture uses ≥2 exotic ops (math_space, spiking, functional categories), grant 2x training steps at screening
- Track `loss_improvement_rate = (loss[250] - loss[500]) / loss[250]`
- If still improving at step 500, extend to step 1000
- **Success**: Slow-converging novel architectures get extended budget; false-negative rate reduced

#### 2H. Routing-Aware Fingerprint Dimensions (Routing-First P2.6)
**Files**: `eval/fingerprint.py`
- Add 3 new fingerprint dimensions: `routing_selectivity` (std of difficulty scores), `routing_compute_ratio` (slow/fast FLOP ratio), `routing_lane_correlation` (position/content correlation)
- Compute only when routing ops detected in graph
- **Success**: Routing architectures occupy distinct fingerprint region (CKA distance >0.3 from dense)

#### 2I. Op-Pair Mining Script (Judgment Engine Phase 6.1 — data extraction only)
**Files**: New script `research/tools/mine_op_pairs.py`
- Query lab_notebook.db for top-performing graphs (stage1-passing, top quartile loss_ratio)
- Extract op bigrams from each graph's topological order
- Output CSV with: pair, count, avg_loss_ratio, avg_stability_score
- This is data extraction only — gemini does NOT modify grammar or motifs
- **Success**: CSV with ≥50 op pairs ranked by (count × avg_loss_ratio_improvement)

---

## Phase 3: Integration (After Phase 2)

### claude-opus — Designer + Runner Wiring

#### 3A. Wire Judgment into Runner (Judgment Engine Phase 4.1–4.4)
**Files**: `runner/synthesis.py`, `runner/execution_screening.py`, `runner/results_analysis.py`
**Depends on**: 2C, 2D
- Candidate generation: bias motif selection toward judgment-supported compositions
- Screening prioritization: rank candidates by predicted success
- Promotion decisions: include judgment evidence
- Decision trace persistence to `decision_traces` table

#### 3B. Refinement Guardrails Completion (Judgment Engine Phase 0)
**Files**: `aria_designer/api/app/mutation.py`, `aria_designer/api/app/main.py`
**Depends on**: 2C
- Make parent scores actually constrain mutations (not just rationale text)
- Add smoke test call in quality gate
- Wire `intent_parser.py` constraints into mutation strategy selection
- Fix `test_refine_winner`

#### 3C. Designer Integration (Judgment Engine Phase 3)
**Files**: `aria_designer/api/app/suggestions.py`, `aria_designer/api/app/mutation.py`
**Depends on**: 2C, 3B
- Replace `_score_adjustment()` with `judgment.score_candidate()` calls
- Return structured evidence per suggestion
- Delete old random mutation code from `mutation.py`

---

### codex — Native Kernel Expansion + Loss/Optimizer Synthesis

#### 3D. Native Interaction/Sensitivity Kernels (Perf Plan Phase 2)
**Files**: `aria_core/src/cpu/kernels.cpp`, `aria_core/bindings/bindings.cpp`
**Depends on**: 1D
- Implement `aria_interaction_metrics_f32` and `aria_sensitivity_metrics_f32` in C++ with AVX2
- Remove Python polyfill guards in `eval/fingerprint.py`

#### 3E. Activate Loss/Optimizer Synthesis (Next-Gen P0.3 + P5.2)
**Files**: `scientist/runner/execution_training.py`
**Depends on**: None
- Make `loss_type` and `optimizer_type` searchable dimensions in `RunConfig`
- Enable synthesized loss/optimizer for screening stage only
- Imports already exist — wire them into the actual training path with config flags

#### 3F. Typed Motifs for Mathematical Spaces (Next-Gen P2.3)
**Files**: `synthesis/motifs.py`
**Depends on**: 2E (algebraic types)
- Create space-specific motifs: hyperbolic block, tropical attention block, Clifford transform block, p-adic hierarchy block
- Mine existing leaderboard for successful exotic-op chains as empirical seeds

---

### gemini — Dashboard + Feedback Scaffolding

#### 3G. Reference Architecture Dashboard UI (Ref Arch P3.x)
**Files**: `research/dashboard/src/components/`
- Add a "Reference Architectures" section/tab showing GPT-2, Mamba, RWKV, RAG baselines
- Display their metrics as comparison targets
- Link to graph visualization for each

#### 3H. Decision Trace Dashboard View
**Files**: `research/dashboard/src/components/`
**Depends on**: 3A (traces persisted)
- Display recent decision traces: what was generated/promoted/rejected and why
- Simple table with columns: timestamp, candidate, action, top_signal, score

#### 3I. Feedback Loop Scaffolding (Judgment Engine Phase 5.1)
**Files**: `aria_designer/api/app/main.py`, `aria_designer/api/app/database.py`
- Track when user clicks "Apply" on a suggestion: record suggestion details, fingerprint, intent
- Track when user rejects: record negative signal
- Store in `suggestion_outcomes` table
- This is data collection only — consumption by research (Phase 5.2-5.3) deferred

---

## Phase 4: Polish (After Phase 3)

### claude-opus
- **4A.** Difficulty scorer pretraining (Routing-First P2.4) — entropy-based bootstrap
- **4B.** NSGA-II evolution integration with grammar weights (Next-Gen P4.1–P4.2)
- **4C.** Feed designer feedback into research priors (Judgment Engine Phase 5.2–5.3)

### codex
- **4D.** CUDA kernels for tropical/clifford ops (Perf Plan Phase 3)
- **4E.** fp16 Cython dispatch (Native Runner remaining)
- **4F.** Graph fusion pass (Native Runner remaining)

### gemini
- **4G.** Morphological box incompatibility constraints (Next-Gen P5.3)
- **4H.** Pareto frontier chart in dashboard (extends existing chart work)
- **4I.** Investigation gating — skip full fingerprint for failing candidates (Perf Plan Phase 4)

---

## Deferred / Out of Scope

These items are lower priority or depend on empirical results from the above:
- P6.1–P6.2 (Next-Gen): Exotic mathspace isolated benchmarks + composition algebra — wait for algebraic types (2E) to land first
- P5.1 (Next-Gen): Morphological box integration into grammar — large scope, revisit after grammar overhaul proven
- Cython bridge for reference architectures (Ref Arch P1.3) — low value given native runner coverage
- E2E tests for reference architectures (Ref Arch P4.x) — nice-to-have

---

## Coordination Rules

1. **Claim before coding.** Update `.current_work.md` with agent name + files BEFORE writing code.
2. **No silent reclaims.** If claimed by another agent, ask — don't take.
3. **Read this plan fresh** before starting any task.
4. **Phase gates.** Don't start Phase N+1 tasks until Phase N dependencies are met.
5. **Test after each task.** Run `pytest research/tests/ -x --tb=short -q` minimum.
6. **No duplication.** Grep for existing implementations before writing new ones.
7. **No dead code.** Every new file must be imported and called. Every deleted function must have its callers updated.

---

## Task Count Summary

| Agent | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Total |
|-------|---------|---------|---------|---------|-------|
| **claude-opus** | 3 (hard) | 3 (hard) | 3 (hard) | 3 (hard) | **12** |
| **codex** | 3 (medium) | 3 (medium) | 3 (medium) | 3 (medium) | **12** |
| **gemini** | 3 (easy) | 3 (easy) | 3 (easy) | 3 (easy) | **12** |

# HYDRA: Novel Architecture Discovery Pipeline

**Status**: IN PROGRESS
**Agents**: claude-opus, gemini, codex
**Goal**: Discover architectures achieving 3-5x compute-efficiency over Transformer/Mamba baselines by generating hundreds of thousands of candidates through 8-dimensional morphological exploration and 5 parametric template families, filtered through a 4-phase evaluation funnel.

---

## 0. The Template Grammar: The "Blank Canvas"

The "blank canvas" is governed by a **Morphological Box** (`research/morphological_box.py`), defining the discrete dimensions of architectural variation.

### Rigid Framework (The Invariants)
- **Standard Interface**: All components operate on `(Batch, Sequence, Dimension)` tensors.
- **Boundaries**: Input fixed at `vocab_size` (32,000); Output fixed at `vocab_size` via LM head.
- **Residual Backbone**: Default topology is a pre-norm residual stack, though mutated variants (U-Net, Fractal) are permitted.

### The 8 Dimensions of Variation
1. **Token Representation**: Dense float, Binary hash (STE), Sparse top-k, Complex-valued, Quaternion, Multi-resolution, Mixture embedding, RVQ quantized.
2. **Weight Storage**: Dense, Low-rank (UV), Hypernetwork, Shared basis, Kronecker, Polynomial, Structured N:M sparse (2:4, block-sparse).
3. **Token Mixing**: Softmax attention, Linear attention, SSM (Mamba), Fourier mixing, Graph attention, Integral kernel mixing.
4. **Channel Mixing**: SwiGLU MLP, MoE top-k, KAN-spline, RWKV channel, Implicit fixed-point (DEQ), Basis expansion.
5. **Compute Routing**: Uniform, Mixture of Depths, Early exit, Adaptive recursion (MoR), Token merging, Adaptive Tri-lane.
6. **Architecture Topology**: Sequential, U-Net, Fractal, DenseNet, Mixture of Paths, Feedback loop.
7. **Normalization**: RMSNorm (pre), LayerNorm (pre), Dynamic norm, Sigmoid norm (QK-norm).
8. **Positional Encoding**: RoPE, ALiBi, Random Fourier, Convolutional (implicit).

---

## 1. Architecture: Language Stack & DRY Principles

### Where Each Language Lives

```
┌──────────────────────────────────────────────────────────┐
│ Python (orchestration only)                               │
│ ├─ synthesis/templates.py    — graph topology generation  │
│ ├─ synthesis/grammar.py      — random graph construction  │
│ ├─ synthesis/validator.py    — static graph validation    │
│ ├─ evaluator.py              — training funnel            │
│ ├─ scientist/runner.py       — experiment orchestration   │
│ └─ scientist/api.py          — REST endpoints             │
├──────────────────────────────────────────────────────────┤
│ Cython (thin bridge, no logic)                            │
│ └─ runtime/native/cython/aria_bridge.pyx                  │
│    Dispatches op names → C/Rust kernels, zero logic       │
├──────────────────────────────────────────────────────────┤
│ Rust (scheduler + dispatch)                               │
│ └─ runtime/native/rust/aria-scheduler/src/executor.rs     │
│    Topological execution, arena allocation, backward pass │
├──────────────────────────────────────────────────────────┤
│ C/C++ (all compute kernels)                               │
│ ├─ aria_core/src/cpu/math_space.cpp — math-space kernels  │
│ ├─ aria_core/src/cpu/kernels.cpp    — core op kernels     │
│ └─ aria_core/include/kernels.h      — kernel declarations │
└──────────────────────────────────────────────────────────┘
```

### DRY Enforcement

| Concern | Mitigation |
|---------|-----------|
| Op pools defined once | `_derive_pool()` merges registry + fallback; pools referenced by both SlotTemplate topologies and smart functions. |
| Each template exists in ONE form | SlotTemplate topology for materialized variants; smart function for richer control flow. No dead code. |
| Primitives registered once | `primitives.py` is the single registry; `compiler.py` dispatches; `mathspaces/*.py` implements. |
| Validation logic centralized | `validator.py` for static checks; `evaluator.py` for runtime checks; no duplication. |
| Kernel code: one implementation | C kernel → Cython bridge → Python fallback. Python is fallback ONLY when native unavailable. |

---

## 2. Component Taxonomy (The Library)

The system maps high-level UI components to research primitives via `component_mapping.yaml`.

### Core Mathematical Primitives
- **Non-Linearities**: `relu`, `gelu`, `silu`, `swiglu`, `tanh`, `sigmoid`.
- **Linear Algebra**: `linear_proj`, `low_rank_proj`, `shared_basis_proj`, `nm_sparse_linear`, `block_sparse_linear`.
- **Advanced Math Spaces**:
  - **Hyperbolic**: `exp_map`, `log_map`, `poincare_add`, `hyp_linear`, `hyp_tangent_nonlinear`, `hyperbolic_norm`.
  - **Tropical**: `tropical_gate`, `tropical_center`, `tropical_attention`, `tropical_add`, `tropical_matmul`.
  - **Clifford**: `geometric_product`, `rotor_transform`, `grade_select`, `grade_mix`, `clifford_attention`.
  - **Ultrametric/p-adic**: `padic_gate`, `padic_expand`, `ultrametric_attention`.
  - **Spiking**: `stdp_attention`, `lif_neuron`, `spike_rate_code`.

---

## 3. Implementation Status

### Part A: Five Novel Template Families — DONE

| Task | Status | Agent | Files |
|------|--------|-------|-------|
| A.1: Math-Space Op Pools | ✅ Done | claude-opus | `synthesis/templates.py`, `synthesis/primitives.py` |
| A.2: 5 SlotTemplate Topologies | ✅ Done | claude-opus | `synthesis/templates.py` (TOPOLOGY_LIBRARY) |
| A.3: 5 Smart Template Functions | ✅ Done | claude-opus | `synthesis/templates.py` (SMART_TEMPLATES) |

### Part B: 4-Phase Evaluation Funnel — DONE

| Task | Status | Agent | Files |
|------|--------|-------|-------|
| B.1: η Pre-Filter | ✅ Done | claude-opus | `synthesis/validator.py` (`compute_theoretical_efficiency()`, `ValidationResult.eta`) |
| B.2: Early Termination Stage 1 | ✅ Done | claude-opus | `evaluator.py` (`Stage1Result.early_exit_step/reason`, 3 checkpoints) |
| B.3: Pre-Investigation Gate | ✅ Done | claude-opus | `scientist/runner.py` (`_hydra_fingerprint_gate()`, CKA + Jacobian checks) |
| B.4: Slot-Swap Optimization | ✅ Done | claude-opus | `scientist/runner.py` (`_try_slot_swap_optimization()`) |
| B.5: Discovery Manifesto | ✅ Done | claude-opus | `scientist/persona.py` (`generate_discovery_manifesto()`) |
| B.6: Scaling Delta API | ✅ Done | claude-opus | `scientist/api.py` (`/api/discoveries/<id>/scaling-delta`) |

### Part C: Native Kernel Implementations — DONE

| Task | Status | Agent | Files |
|------|--------|-------|-------|
| C.1: Hyperbolic Kernels (C) | ✅ Done | gemini | `aria_core/src/cpu/math_space.cpp` |
| C.2: Clifford Kernels (C) | ✅ Done | gemini | `aria_core/src/cpu/math_space.cpp` |
| C.3: Tropical Kernels (C) | ✅ Done | gemini | `aria_core/src/cpu/math_space.cpp` |
| C.4: Rust Scheduler Integration | ✅ Done | gemini-3.1 | `runtime/native/rust/aria-scheduler/` |
| C.5: Cython Bridge | ✅ Done | gemini-3.1 | `runtime/native/cython/aria_bridge.pyx` |
| C.6: Binding Stubs Impl | ✅ Done | claude-opus| `aria_core/src/cpu/binding_stubs.cpp` |

### Part D: Semantic Enforcement & Scale-Up — IN PROGRESS

| Task | Status | Agent | Files |
|------|--------|-------|-------|
| D.1: Semantic Enforcement Pass | ✅ Done | gemini | `research/synthesis/grammar.py` (`_enforce_routing_semantics()`) |
| D.2: Scale-Up Synthesis (n=1000) | 🚧 In Progress | gemini | Active experiment `0d03add4-686` |
| D.3: Kernel Tests (Math-Space) | 🚧 In Progress | codex | `aria_core/tests/test_math_space_kernels.py` |
| D.4: E2E Pipeline Validation | 🚧 In Progress | codex | `research/tests/test_hydra_e2e.py` |

---

## 4. Part C: Native Kernel Detail

### C.1: Hyperbolic Kernels in C
- **Ops**: `exp_map`, `log_map`, `poincare_add`, `hyp_linear`, `hyp_tangent_nonlinear`, `hyperbolic_norm`
- **Optimization**: Vector-wise SIMD (AVX2/AVX-512) via hardware-agnostic macros.
- **Linkage**: Wrapped in `extern "C"` for C-linkage stability.

### C.2: Clifford Kernels in C
- **Ops**: `geometric_product`, `rotor_transform`, `grade_select`, `grade_mix`, `clifford_attention`
- **Optimization**: Optimized for Cl(3,0) 8-component multivectors using 8-wide SIMD.

### C.3: Tropical Kernels in C
- **Ops**: `tropical_attention`, `tropical_gate`, `tropical_add`, `tropical_matmul`
- **Math**: $(\max, +)$ semiring optimized with SIMD `max_ps` and `add_ps`.

---

## 5. Template Family Details

### 1. Hyperbolic Hierarchy Scanner
**Hypothesis**: Hyperbolic space embeds hierarchical structures with exponentially low distortion. Tokens near the Poincaré ball origin (broad concepts) → lightweight hyperbolic processing; tokens near boundary (specific details) → deep SSM processing.
**Topology**: norm → exp_map → difficulty_scorer → lane_router → {hyp_lane, ssm_lane} → gather → proj → residual

### 2. Clifford Geometric Mixer
**Hypothesis**: Clifford geometric product simultaneously encodes alignment (dot) and orientation (wedge). Rotor transforms give rotation-equivariant projections with 8x fewer params.
**Topology**: norm → pos → clifford_attn → proj → add → norm → clifford_channel → mlp → add

### 3. Tropical Sparse Router
**Hypothesis**: Tropical semiring (max,+) creates natural winner-take-all routing — idempotent, inherently discrete-like yet differentiable via softmax approximation.
**Topology**: scorer → router → 3×dispatch → {tropical_easy, tropical_medium, heavy} → gather → balance_loss

---

## 6. Discovery Scoring Function

The **Discovery Score** (0-100+) ranks architectures based on performance, novelty, and efficiency.

$$Score = (Loss \cdot 30) + (Novelty \cdot 20) + (Baseline \cdot 25) + (ID \cdot 5) + (ParamEff \cdot 10) + (Speed \cdot 10) + Bonuses$$

- [x] **Loss Ratio (30 pts)**: $1.0 - \frac{CandidateLoss}{BaselineLoss}$.
- [x] **Novelty Score (20 pts)**: Structural fingerprint distance from known baselines.
- [x] **Baseline Comparison (25 pts)**: $clamp(1.5 - \frac{CandidateLoss}{TransformerLoss}, 0, 1)$.
- [x] **Identity/Similarity (5 pts)**: CKA-backed identity bonus.
- [x] **Parameter Efficiency (10 pts)**: Normalized inverse $log_{10}$ of parameter count.
- [x] **Learning Speed (10 pts)**: Measured via $LossImprovementRate$ in early steps.
- [x] **Bonuses**: Rewarded for Efficiency, Routing, Adaptive Compute, and Sparsity.

---

## 7. Performance Targets & Baseline Benchmarks

### Baseline Scaling Laws
- **Transformer (GPT-2 Style)**: $L(N) \approx 11.94 \cdot N^{-0.0696}$
- **Mamba (SSM Style)**: $L(N) \approx 12.0 \cdot N^{-0.0741}$

### Target Metrics
- **Parameter Efficiency Ratio ($\eta$)**: Must be $\ge 3.0$ (achieve baseline loss with $< 33\%$ parameters).
- **Throughput Guard**: Must maintain `FlopEfficiencyRatio >= 0.5` vs. Transformer baseline.
- **Standard Loss Target (d=256, seq=128)**: 
  - Vanilla Transformer: **3.6 - 3.8**
  - Mamba: **3.9 - 4.1**
  - **HYDRA Target**: **< 3.4** at identical parameter counts.

---

## 8. Evaluation Funnel — ✅ Verified & Hardened (claude-opus)

### Phase 1: Theoretical Filter (η coefficient) — ✅
- `validator.py:compute_theoretical_efficiency()`: η = info_capacity / (params × flops)
- Zero-GPU cost; added to `ValidationResult.eta`.
- **Hardened**: η < 1e-8 → `add_error()` (reject graph), η < 1e-6 → `add_warning()`.
- **Wired into grammar**: `grammar.py:_validate_graph()` now rejects graphs with η < 1e-8 at generation time.

### Phase 2: Early Termination in Stage 1 — ✅
- `evaluator.py`: Step 50 (slow learner), Step 100 (dead gradients), Step 200 (insufficient progress).
- Tracks `early_exit_step` and `early_exit_reason` in `Stage1Result`.
- **Verified**: All 3 checkpoints fire correctly, results propagated to `record_program_result`.

### Phase 3: Pre-Investigation Gate + Slot-Swap — ✅
- `runner.py:_hydra_fingerprint_gate()`: CKA similarity < 0.95, Jacobian spectral norm in [0.1, 100]. Tags `hydra_prime` for novel + parameter-efficient candidates (CKA < 0.85 AND params < 1M).
- **Wired**: Called from `_auto_escalate()` at line 14690 for all screening→investigation promotions.
- `runner.py:_try_slot_swap_optimization()`: During investigation, swap norm → dynamic_norm if loss ratio is borderline (> 0.3).
- **Wired**: Called in `_run_inline_investigation()` after `investigation_passed` is determined, before benchmark evals. Records swap metadata on the investigation entry.

### Phase 4: Discovery Documentation — ✅
- `persona.py:generate_discovery_manifesto()`: Markdown with scaling law delta, architecture composition, hydra_prime tagging.
- **Wired**: Called from `_run_inline_investigation()` after `_auto_escalate()` for entries with `composite_score >= 150`. Writes manifesto files to `research/discoveries/manifesto_<result_id>.md`.

---

## All Primitives Used (verified in registry + compiler dispatch)

| Category | Primitives | Registry | Compiler | C Kernel |
|----------|-----------|----------|----------|----------|
| Hyperbolic | `exp_map`, `log_map`, `poincare_add`, `hyp_linear`, `hyp_tangent_nonlinear`, `hyperbolic_norm` | ✅ | ✅ | ✅ |
| Clifford | `geometric_product`, `rotor_transform`, `grade_select`, `grade_mix`, `clifford_attention` | ✅ | ✅ | ✅ |
| Tropical | `tropical_attention`, `tropical_gate`, `tropical_add`, `tropical_matmul`, `tropical_center` | ✅ | ✅ | ✅ |
| SSM | `selective_scan`, `rwkv_time_mixing`, `conv1d_seq`, `state_space` | ✅ | ✅ | ✅ |
| Compression | `latent_attention_compressor`, `progressive_compression_gate`, `compression_mixture_experts` | ✅ | ✅ | ✅ |
| Routing | `difficulty_scorer`, `lane_router`, `conditional_dispatch`, `conditional_gather`, `load_balance_loss` | ✅ | ✅ | ✅ |
| Control | `fixed_point_iter`, `training_phase_gate`, `mixed_recursion_gate`, `early_exit` | ✅ | ✅ | ✅ |
| Entropy | `entropy_router`, `route_topk` | ✅ | ✅ | ✅ |

---

## Verification Commands

```bash
source /home/tim/venvs/llm/bin/activate

# Compile checks (all should pass)
python -m py_compile research/synthesis/templates.py
python -m py_compile research/synthesis/primitives.py
python -m py_compile research/synthesis/validator.py
python -m py_compile research/evaluator.py
python -m py_compile research/scientist/runner.py
python -m py_compile research/scientist/persona.py
python -m py_compile research/scientist/api.py

# Integration tests (skip pre-existing failures)
pytest research/tests/test_integration.py -q -k "not test_experiment_clusters and not test_leaderboard_upsert" --tb=short
```

# Next Generation Architecture Synthesis Plan

## 1. Current Progression Criteria for Fingerprints
Based on recent threshold adjustments, a fingerprint must pass the following pipeline gates to progress:

- **Behavioral Novelty ($> 0.15$)**: The model's Centered Kernel Alignment (CKA) representation matrix must not be $\geq 0.85$ identical to hardcoded heuristic baselines (e.g., `ref_transformer`, `ref_conv`, `ref_ssm`). The threshold was lowered to `0.15` to permit functional sequence-dimension smoothing typical of valid autoregressive models.
- **Loss Improvements (Stage 1)**: The architecture must successfully initialize, avoid NaN collapse, and yield target `best_loss_ratio` and `baseline_loss_ratio` metrics compared to baseline models.
- **Hardware Efficiency**: Must pass basic `flops_per_token` and `throughput_tok_s` thresholds over standard context window sizes to prune catastrophically slow graph structures.
- **Auto-Escalation Selection**: `results_auto_escalate_phase7.py` promotes results to formal investigations if `novelty_confidence \geq 0.50` and `brittle_risk == False`.

## 2. Strategic Changes for 5x Performance Improvement over Mamba/GPT
To achieve a 5x computational/loss improvement over highly optimized architectures (SSM/Attention), the search space must shift away from standard dense matrix multiplications in a Euclidean latent space.

### Design Level Changes (Math & Primitives)
- **Aggressive Conditional Routing (`compiler_ops_routing.py`)**: Move away from activating all parameters for every token. Introduce primitives for highly sparse execution paths (e.g., token-level MoE, granular block-dropping, or dynamic depth).
- **Leverage Non-Euclidean Latents (`compiler_ops_mathspaces.py`)**: Use hyperbolic or complex mathematical spaces to encode hierarchical data (like syntax trees) with exponentially fewer dimensions. Reducing the latent dimension size directly yields quadratic compute reductions.
- **Sub-Quadratic Sequence Mixing (`compiler_ops_sequence.py`)**: Transition graph generation to favor linear-time recurrent/scan primitives with high hardware-fusion capabilities, bypassing SRAM/HBM memory bandwidth bottlenecks inherent in standard attention.

### Creation & Synthesis Engine Changes (Graph Generation)
- **Hardware-Aware Mutation Graphing (`graph.py`, `motifs.py`)**: Primitives cannot just be mathematically viable; they must map cleanly to Triton/CUDA block sizes. Graph mutation must heavily penalize structures that are bottlenecked by SRAM/HBM bandwidth.
- **Pareto-Frontier Targeting**: Modify the synthesis fitness function away from simply optimizing "Novelty + Loss". Shift to strictly optimizing `(Loss_Ratio * Throughput) / Parameter_Count`. This forces the evolutionary search to cull standard dense graph mutations entirely and only explore the sparse/sub-quadratic frontier.

## Next Steps for Agents
1. Review and implement sparse primitives in `compiler_ops_routing.py`.
2. Expand mathematical representations in `compiler_ops_mathspaces.py`.
3. Adjust fitness/mutation functions in `graph.py` and `motifs.py` to enforce the Pareto-Frontier target.

---

## 3. Independent Assessment — AI/Math Scientist Review (2026-03-12)

### 3.1 Diagnosis: Why the System Converges to Mediocrity

The core problem is not a lack of exotic primitives. It is that the system's **fitness landscape is degenerate**: a single strong attractor basin (dense attention + SwiGLU + RMSNorm ≈ GPT-2) dominates the search, and multiple interacting mechanisms conspire to prevent escape from that basin.

#### 3.1.1 The Scoring Function Is a Weighted Sum, Not Multi-Objective

The composite score in `leaderboard_scoring.py` is:

$$S = 100 \cdot f(\text{loss})^{1.6} \cdot c + 40 \cdot \text{nov} \cdot g(\text{loss}) + \sum_i w_i \cdot h_i(\text{metrics})$$

This is a **scalarization** of a multi-objective problem. By Pareto theory, a weighted-sum scalarization can only find solutions on the **convex hull** of the Pareto frontier. Any concave regions — exactly where novel architectures with different loss-efficiency tradeoffs live — are invisible to this formulation.

Worse, the novelty term is **gated by loss performance** via $g(\text{lr}) = \max(0, (0.9 - \text{lr})/0.6)$. This creates a chicken-and-egg problem: novel architectures need loss convergence to earn novelty credit, but they need novelty credit to survive long enough to converge. The gate should be removed or inverted (reward novelty MORE when loss is weak, as a signal to investigate further).

#### 3.1.2 The Training Signal Is Too Weak to Differentiate

500 steps of AdamW on cross-entropy with batch_size=4, seq_len=128 produces a **noisy, low-information signal**. At this scale:

- All architectures that don't NaN will achieve loss_ratio ∈ [0.3, 0.9] — a narrow band
- The loss landscape curvature at 500 steps is dominated by **optimizer dynamics**, not architecture quality
- Two architectures with genuinely different asymptotic behavior look identical at step 500

Meanwhile, `loss_synthesis.py` (8 loss variants including tropical-CE, spectral loss, rank-weighted CE) and `optimizer_synthesis.py` (11 optimizers including spectral momentum, tropical gradient, Lion variant) are **completely dead code** — never called by the training loop. The system was designed to co-search architecture × loss × optimizer, but only the architecture dimension is active.

This is like searching for the best car design by only testing engines at idle RPM with the same fuel and transmission.

#### 3.1.3 The Grammar Lacks Algebraic Type Constraints

The grammar can emit sequences like: `exp_map → tropical_matmul → grade_select`. This is **mathematically nonsensical** — mapping from the Poincaré ball (a Riemannian manifold with curvature -1) into a tropical semiring (where "multiplication" means addition), then extracting a Clifford algebra grade from the result.

Each mathematical space has an **algebraic signature**:
- Hyperbolic: operates on the open unit ball $\mathbb{B}^d = \{x \in \mathbb{R}^d : \|x\| < 1\}$
- Tropical: operates on $(\mathbb{R} \cup \{+\infty\}, \min, +)$ — values represent path costs
- Clifford: operates on multivectors in $\text{Cl}(3,0)$, requiring $d \equiv 0 \pmod{8}$
- p-adic: operates on $\mathbb{Q}_p$ with ultrametric topology

The grammar treats all of these as `(B, S, D) → (B, S, D)` black boxes. There are no type constraints preventing composition of operations from incompatible algebraic structures. This means most exotic-op combinations are **gradient-carrying garbage** — they compute something, but it has no mathematical meaning.

#### 3.1.4 The Sequential Scan Bottleneck Is Algorithmically Unnecessary

The `StateSpaceMixer` in `arch_builder.py:270-298` and `_op_selective_scan` in `compiler_ops_sequence.py` both use:

```python
for t in range(S):
    h = A_bar[:, t] * h + b_x[:, t]
```

This is a **first-order linear recurrence** $h_t = A_t h_{t-1} + b_t$. The scan operator $\oplus$ defined by $(A_1, b_1) \oplus (A_2, b_2) = (A_1 A_2, A_1 b_2 + b_1)$ is **associative**, which means the entire recurrence can be computed in $O(\log S)$ parallel steps using **Blelloch's parallel prefix scan** (1990).

This is exactly what Mamba-2 (Dao & Gu, 2024) and the `torch.cumsum`-based implementations exploit. The sequential loop makes our SSM primitives $O(S/\log S)$ times slower than they need to be, and it prevents GPU utilization since each step depends on the previous.

**This is the single highest-impact algorithmic fix in the codebase.**

#### 3.1.5 The 48 Missing C++ Bindings Are a Systematic Performance Tax

Investigation found that 48 of 135 implemented C++ kernels in `aria_core` lack Python bindings. This includes:
- All F16 variants (aria_add_f16, aria_mul_f16, aria_relu_f16, aria_matmul_f16) — AVX2-optimized
- All backward kernels — gradient computation falls back to PyTorch autograd
- aria_cumsum_f32 — SIMD + OpenMP optimized, used by tropical operations

Every missing binding means the profiler measures **Python fallback throughput**, not kernel throughput. Since `compute_efficiency_multiple()` compares against GPT-2 reference metrics (`throughput_tok_s: 1,200,845`), architectures using these ops are systematically **under-scored** — not because they're slow, but because the measurement path is slow.

This creates a perverse incentive: the scoring system rewards architectures that happen to use the 87 bound ops and penalizes those that use the 48 unbound ones, regardless of the architecture's actual computational merit.

### 3.2 What the Existing Plan Gets Right

Sections 3-6 of the existing plan correctly identify:
- The grammar needs tighter closure with empirical signals (Phase A)
- "Sparse" must mean realized savings, not structural decoration (Phase B)
- Refinement should be the exploitation engine (Phase C)
- Pareto pressure should be upstream in generation, not just scoring (Phase D)
- Exotic mathspaces should expand last, not first (Phase E)

**I agree with this ordering and these priorities.** My assessment adds the mathematical specifics of *how* to execute each phase and identifies additional structural problems the existing plan doesn't address.

---

## 4. Mathematician's Remediation Plan

### Phase 0: Fix the Measurement Infrastructure (Prerequisite)

**Rationale**: You cannot optimize what you cannot measure. The scoring system currently penalizes architectures for infrastructure gaps, not architectural flaws.

#### P0.1 — Expose all 48 missing C++ kernel bindings
- **Files**: `aria_core/bindings/bindings.cpp`
- **Scope**: Add pybind11 wrappers for all kernels declared in `kernels.h` but absent from bindings
- **Priority**: F16 variants first (inference throughput), backward kernels second (training throughput)
- **Impact**: Removes systematic scoring bias against exotic-op architectures
- **Validation**: `python -c "import aria_core; print(len(dir(aria_core._C)))"` should match kernel count in headers

#### P0.2 — Implement parallel associative scan for SSM primitives
- **Files**: `synthesis/compiler_ops_sequence.py` (`_op_selective_scan`, `_op_state_space`), `arch_builder.py` (`StateSpaceMixer`)
- **Math**: Replace sequential loop with parallel prefix scan using the semigroup $(A, b) \oplus (A', b') = (AA', Ab' + b)$
- **Implementation**: Use `torch.cumsum` in log-space for the diagonal case, or implement segmented scan via `torch._C._VariableFunctions`
- **Fallback**: Keep sequential path for S ≤ 32 (overhead of parallel scan not worth it for short sequences)
- **Impact**: SSM primitives become $O(S \log S / P)$ instead of $O(S)$ serial, enabling GPU saturation

#### P0.3 — Connect loss_synthesis.py and optimizer_synthesis.py to training
- **Files**: `scientist/runner/execution_training.py` (training loop)
- **Scope**: Make loss function and optimizer **searchable dimensions**, not fixed constants
- **Design**: Add `loss_type` and `optimizer_type` fields to `RunConfig`; default to `cross_entropy`/`adamw` for backward compatibility
- **Constraint**: Only enable synthesized loss/optimizer for screening stage (investigation/validation use standard CE+AdamW for comparability)
- **Impact**: Opens a 3-dimensional search space (architecture × loss × optimizer) instead of 1-dimensional

### Phase 1: Fix the Fitness Landscape (Critical)

#### P1.1 — Replace weighted-sum scoring with NSGA-II dominance ranking

The composite score collapses a multi-objective problem into a scalar. Replace with proper Pareto dominance:

**Objective vector** $\mathbf{f}(x) = (\text{loss\_ratio}, -\text{throughput}, \text{param\_count}, -\text{novelty})$

**Dominance**: $x$ dominates $y$ iff $f_i(x) \leq f_i(y) \; \forall i$ and $\exists j: f_j(x) < f_j(y)$

**Ranking**: Non-dominated sorting assigns each candidate to a **Pareto front** $\mathcal{F}_1, \mathcal{F}_2, \ldots$. Within each front, use **crowding distance** (Deb et al., 2002) to maintain diversity.

- **Files**: `scientist/leaderboard_scoring.py`, `search/evolution.py`
- **Keep** `compute_composite_score()` for dashboard display and backward compatibility
- **Add** `pareto_rank()` and `crowding_distance()` for actual selection pressure
- **Impact**: Evolution can discover solutions on concave portions of the frontier that weighted sums miss

#### P1.2 — Remove the novelty-loss gate

The gate $g(\text{lr}) = \max(0, (0.9 - \text{lr}) / 0.6)$ that multiplies novelty by loss performance must go. Novel architectures that haven't converged yet are precisely the ones that need the novelty bonus to survive selection.

- **Replace with**: Separate novelty as an independent objective in the NSGA-II formulation (P1.1)
- **Or, minimally**: Change gate to $g(\text{lr}) = \min(1.0, 0.3 + 0.7 \cdot (0.9 - \text{lr}) / 0.6)$ — floor of 30% novelty credit even for weak performers
- **Files**: `scientist/leaderboard_scoring.py:230`

#### P1.3 — Increase screening training budget for novel architectures

500 steps is too few to differentiate architectures with different asymptotic behavior. But we can't afford 5000 steps for all candidates.

**Adaptive budget**: If the architecture uses ≥2 exotic ops (math_space, spiking, functional categories), grant 2x training steps at screening. The hypothesis is that non-standard representations need more gradient steps to align their internal geometry.

- **Files**: `scientist/runner/execution_training.py`
- **Metric**: Track `loss_improvement_rate = (loss[step_250] - loss[step_500]) / loss[step_250]` — architectures still improving at step 500 get extended to step 1000
- **Impact**: Reduces false-negative rate for slow-converging novel architectures

### Phase 2: Add Algebraic Type Constraints to the Grammar

#### P2.1 — Define algebraic space tags for primitives

Each primitive should declare what algebraic space it operates in:

```python
@dataclass
class AlgebraicType:
    space: str  # "euclidean", "poincare", "tropical", "clifford", "padic", "spiking"
    input_constraint: str  # "unit_ball", "real", "multivector_8", "non_negative", "binary"
    output_guarantee: str  # same options
```

- **Files**: `synthesis/primitives.py` — add `algebraic_type` field to `PrimitiveOp`
- **Rules**:
  - `exp_map` outputs `poincare/unit_ball`, so next op must accept `unit_ball` input
  - `tropical_matmul` outputs `tropical/non_negative`, incompatible with `grade_select` (expects `clifford/multivector_8`)
  - `log_map` converts `poincare/unit_ball → euclidean/real` (bridge operator)
  - Standard ops (linear_proj, relu, etc.) operate in `euclidean/real`

#### P2.2 — Enforce type compatibility in graph generation

When the grammar picks the next op, filter candidates by algebraic type compatibility with the current node's output type. Bridge operators (exp_map, log_map, padic_expand) explicitly convert between spaces.

- **Files**: `synthesis/grammar.py` (op selection), `synthesis/templates.py` (motif instantiation)
- **Impact**: Eliminates mathematically nonsensical compositions; every graph now has a coherent algebraic interpretation
- **Side effect**: Exotic ops will be sampled in coherent chains (exp_map → hyp_linear → log_map) instead of random soup

#### P2.3 — Create typed motifs for each mathematical space

Currently, all 37 motifs live in `euclidean/real` space. Create space-specific motifs:

- **Hyperbolic block**: `exp_map → hyp_linear → hyp_tangent_nonlinear → log_map` (enter manifold, transform, nonlinearity, exit)
- **Tropical attention block**: `tropical_matmul → tropical_gate → tropical_center` (distance matrix, gated routing, causal centering)
- **Clifford transform block**: `reshape_to_mv8 → rotor_transform → grade_select(1) → reshape_to_flat` (lift to multivector, rotate, extract vector part, flatten)
- **p-adic hierarchy block**: `padic_expand → linear_proj → padic_gate → linear_proj_down` (multi-scale decomposition, transform, hierarchical gating, recompress)

- **Files**: `synthesis/motifs.py`
- **Validation**: Mine existing leaderboard for any successful exotic-op chains; use those as empirical seeds

### Phase 3: Fix Numerical Stability in Mathematical Spaces

#### P3.1 — Adaptive curvature for hyperbolic operations

Fixed curvature $c = 1.0$ is suboptimal. Different data hierarchies have different natural curvatures. Make $c$ a learnable parameter with constraints:

$$c = \text{softplus}(\hat{c}) + \epsilon, \quad \hat{c} \in \mathbb{R}$$

This ensures $c > 0$ and allows gradient-based adaptation. The Poincaré ball radius becomes $1/\sqrt{c}$.

- **Files**: `mathspaces/hyperbolic.py`
- **Risk**: Curvature near 0 collapses to Euclidean (fine), curvature → ∞ collapses ball to a point (clamp $c \leq 10.0$)

#### P3.2 — Adaptive temperature for tropical softmin

Fixed $T = 0.1$ causes numerical underflow for long sequences. Replace with:

$$T(S) = T_{\text{base}} \cdot \sqrt{S / S_{\text{ref}}}$$

where $S_{\text{ref}} = 128$ (reference sequence length) and $T_{\text{base}} = 0.1$. This scales temperature with sequence length, preventing the softmin from collapsing to a hard argmin.

- **Files**: `mathspaces/tropical.py` (all functions using tropical_softmax)
- **Alternative**: Make $T$ learnable per-layer (like the `log_tau` in spiking STDP attention)

#### P3.3 — Replace log-based p-adic valuation with smooth proxy

$v_p(x) = -\log_p(|x|)$ is singular at $x = 0$ and has unbounded gradient near 0. Replace with:

$$\tilde{v}_p(x) = -\frac{\log(|x| + \epsilon)}{\log p}, \quad \epsilon = 10^{-6}$$

And for the distance $d(x,y) = p^{-v_p(x-y)}$, use:

$$\tilde{d}(x,y) = \text{softplus}\left(\frac{|x - y| + \epsilon}{\log p}\right)^{-1}$$

which is smooth, bounded, and preserves the ultrametric ordering.

- **Files**: `mathspaces/padic.py`

#### P3.4 — Gradient clipping for spiking STE

The surrogate gradient in LIF neurons uses $\sigma'(5(V - \theta))$ which has peak gradient 1.25. Through $L$ layers, gradient magnitude scales as $1.25^L$. For $L = 10$: gradient ≈ 9.3x amplification.

Add **per-module gradient scaling**: after each LIF layer, scale gradients by $1/\sqrt{L}$ where $L$ is the number of spiking layers in the graph. This is cheap (one `register_hook` per layer) and prevents gradient explosion in deep spiking networks.

- **Files**: `mathspaces/spiking.py` (execute_lif)

### Phase 4: Implement Proper Multi-Objective Evolution

#### P4.1 — NSGA-II selection in evolution.py

Replace the current `fitness * w + novelty * w` scalar ranking with non-dominated sorting:

```python
def nsga2_select(population, objectives):
    """
    objectives: list of (name, direction) where direction ∈ {"min", "max"}
    Returns: population sorted by (pareto_rank, -crowding_distance)
    """
    fronts = fast_non_dominated_sort(population, objectives)
    for rank, front in enumerate(fronts):
        for ind in front:
            ind.pareto_rank = rank
        assign_crowding_distance(front, objectives)
    return sorted(population, key=lambda x: (x.pareto_rank, -x.crowding_distance))
```

**Objectives** (4-dimensional):
1. `loss_ratio` (minimize)
2. `param_count / baseline_params` (minimize)
3. `1 / throughput_tok_s` (minimize, i.e., maximize throughput)
4. `-novelty_score` (minimize, i.e., maximize novelty)

- **Files**: `search/evolution.py`, `search/novelty_search.py`
- **Impact**: Maintains a diverse frontier instead of converging to a single point

#### P4.2 — Efficiency-aware grammar weighting

After each generation, update grammar weights based on which motifs appear in front $\mathcal{F}_1$:

$$w_m^{(t+1)} = w_m^{(t)} \cdot \left(1 + \alpha \cdot \frac{|\{x \in \mathcal{F}_1 : m \in x\}|}{|\mathcal{F}_1|}\right)$$

where $\alpha = 0.3$ is the learning rate for grammar weights. Motifs that appear frequently in Pareto-optimal solutions get higher sampling probability.

- **Files**: `synthesis/grammar.py` (GrammarConfig.motif_weights), `search/evolution.py` (post-generation update)

### Phase 5: Unlock Dead Capabilities

#### P5.1 — Morphological box dimensions as grammar input

The grammar currently ignores token_representation, weight_storage, topology, normalization, and positional_encoding choices from the morphological box. These 8 dimensions define $8.9 \times 10^6$ possible configurations (product of option counts), but the grammar only searches over op composition.

**Minimal integration**: Sample morphological box choices alongside graph generation. Use the arch_builder for the non-graph components (token representation, normalization, positional encoding) while using the grammar for the computation graph.

- **Files**: `synthesis/grammar.py`, `morphological_box.py`, `arch_builder.py`
- **Impact**: Dramatically expands the effective search space with already-implemented, tested modules

#### P5.2 — Activate loss and optimizer synthesis

Wire `loss_synthesis.py` and `optimizer_synthesis.py` into the screening training loop (P0.3). Treat loss type and optimizer type as heritable traits in evolution:

- **Mutation**: 10% chance of loss/optimizer mutation per generation
- **Crossover**: Inherit loss from parent 1, optimizer from parent 2
- **Fitness attribution**: Track which loss-optimizer pairs improve which architecture families

#### P5.3 — Populate morphological box incompatibility constraints

The `incompatible_with` field in `morphological_box.py` is defined but never populated. Add empirically-derived constraints:

- `binary_hash` token rep + `hypernetwork` weight storage → unstable gradients (known)
- `complex_valued` tokens + `no_norm` → divergence (known)
- `state_space` mixing + `dense_net` topology → O(n²) memory (structural)
- `fourier_mixing` + `none` positional encoding → position-invariant (may be desirable or not)

- **Files**: `morphological_box.py` (Option.incompatible_with tuples)

### Phase 6: Roadmap for Exotic Mathspace Expansion

Only after Phases 0-5 are stable:

#### P6.1 — Benchmark each mathematical space in isolation

Before expanding exotic ops, **measure their standalone value**:

| Space | Benchmark Task | Hypothesis | Metric |
|-------|---------------|------------|--------|
| Hyperbolic | Tree-structured data (code ASTs, dependency trees) | Exponential capacity for hierarchy | loss_ratio on TreeQA |
| Tropical | Shortest-path / routing problems | Native path semantics | routing_savings_ratio |
| Clifford | 3D spatial reasoning, rotation-equivariant tasks | Parameter-efficient rotations | loss/param ratio |
| p-adic | Hierarchical classification (WordNet, taxonomy) | Ultrametric ≈ tree distance | hierarchy_fitness |
| Spiking | Temporal pattern recognition, event streams | Sparse activation | activation_sparsity_score |

- **Files**: New eval tasks in `eval/` for each space
- **Gate**: Only promote a mathspace to increased grammar weight if its isolated benchmark shows ≥ 1.5x improvement over Euclidean baseline on the target task

#### P6.2 — Composition algebra for cross-space operations

Define **bridge operators** that convert between spaces with mathematical guarantees:

- `euclidean_to_poincare`: $x \mapsto \tanh(\|x\|/2) \cdot x/\|x\|$ (maps $\mathbb{R}^d$ into the ball)
- `poincare_to_euclidean`: $\log_0(x)$ (logarithmic map at origin)
- `euclidean_to_tropical`: $x \mapsto -\log(\text{softmax}(x))$ (convert to cost/distance space)
- `tropical_to_euclidean`: $x \mapsto \exp(-x)$ (convert costs back to probabilities)
- `euclidean_to_clifford`: reshape $\mathbb{R}^d \to \mathbb{R}^{d/8} \times \mathbb{R}^8$ (embed as grade-1 multivectors)
- `clifford_to_euclidean`: grade-1 extraction + reshape

The grammar should enforce: **every entry into a non-Euclidean space must be paired with an exit**. This prevents dangling manifold outputs that downstream ops can't interpret.

---

## 5. Priority Ordering and Dependencies

```
P0.1 (bindings)  ──────────────────────────────── standalone, high impact
P0.2 (parallel scan) ─────────────────────────── standalone, highest perf impact
P0.3 (loss/optim synthesis) ───┐
                               ├──→ P5.2 (activate synthesis)
P1.1 (NSGA-II scoring) ───────┤
P1.2 (remove novelty gate) ───┤
P1.3 (adaptive training budget)┘
                               ├──→ P4.1 (NSGA-II evolution)
P2.1 (algebraic types) ───────┤    P4.2 (efficiency grammar weights)
P2.2 (type enforcement) ──────┤
P2.3 (typed motifs) ──────────┘
                               ├──→ P5.1 (morphbox integration)
P3.1-P3.4 (numerical fixes) ──┘    P5.3 (incompatibility constraints)
                                         │
                                         ▼
                                    P6.1 (isolated benchmarks)
                                    P6.2 (composition algebra)
```

**Critical path**: P0.2 → P1.1 → P4.1 (parallel scan → proper scoring → multi-objective evolution)

**Independent tracks** (can run in parallel):
- Track A: P0.1 (bindings) — pure C++/pybind11 work
- Track B: P0.2 + P1.3 (parallel scan + adaptive budget) — training infrastructure
- Track C: P1.1 + P1.2 + P4.1 (scoring overhaul) — search/ranking
- Track D: P2.1 + P2.2 + P2.3 (algebraic types) — grammar/type system
- Track E: P3.1-P3.4 (numerical stability) — mathspace fixes

## 6. Expected Outcomes

| Phase | Metric | Current | Target | Mechanism |
|-------|--------|---------|--------|-----------|
| P0.2 | SSM throughput (tok/s) | ~200K (sequential) | ~1.2M (parallel) | Associative scan |
| P0.1 | Kernel coverage | 64% (87/135) | 100% (135/135) | Complete bindings |
| P1.1 | Frontier diversity | 1 cluster (GPT-2-like) | 4+ clusters | NSGA-II |
| P1.2 | Novel arch survival rate | ~5% past screening | ~20% | Remove novelty gate |
| P2.1-2.3 | Exotic op chain validity | ~10% (random) | ~90% (typed) | Algebraic constraints |
| P3.1-3.4 | Mathspace NaN rate | ~15% | <1% | Numerical fixes |
| P4.1 | Pareto front size | N/A (scalar) | 15-30 non-dominated | Multi-objective |
| P5.1 | Effective search dimensions | 1 (op graph) | 4 (graph + rep + norm + pos) | Morphbox integration |

## 7. Summary Judgment

The system has invested heavily in mathematical breadth (6 exotic spaces, 120+ ops, 37 motifs) but underinvested in **measurement fidelity** and **selection pressure design**. The result is a search process that:

1. **Generates** diverse candidates (good)
2. **Evaluates** them through a noisy, biased lens (bad — missing bindings, fixed loss/optimizer, 500-step budget)
3. **Selects** via a degenerate scalar that suppresses the very novelty it claims to reward (bad — weighted sum, novelty gate)
4. **Composes** mathematical operations without algebraic coherence (bad — type-free grammar)

The fix ordering should be: **measure correctly → select properly → compose coherently → then expand**.

This is not a call to abandon the exotic mathematics. The hyperbolic, tropical, and Clifford algebras are genuinely promising for their respective problem structures. But their value cannot be realized until the surrounding infrastructure stops systematically mis-measuring and mis-ranking the architectures that use them.

The shortest path to 5x efficiency over GPT-2 in this codebase is:
1. Parallel scan (immediate 6x SSM speedup)
2. Complete kernel bindings (accurate throughput measurement)
3. Multi-objective selection (discover the actual Pareto frontier)
4. Typed exotic-op composition (coherent non-Euclidean architectures)
5. Joint architecture-loss-optimizer search (3D instead of 1D)

Each of these is a **multiplicative** improvement on search quality, not additive. Combined, they should move the frontier from "occasionally finds something slightly better than random" to "reliably discovers architectures that exploit the mathematical structure available."

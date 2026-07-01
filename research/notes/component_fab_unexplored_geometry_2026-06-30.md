# component_fab — Unexplored Geometry, Algebra & Math (2026-06-30)

> **MISSION ALIGNMENT (CLAUDE.md):** the fab exists to find NOVEL non-QKV mechanisms
> that beat softmax/frontier. Every axis below is chosen because it is **not** a
> softmax/attention/semiring/reciprocal twin in disguise. The capability walls the
> project already mapped (compositional/multislot binding, long-range AR, gate
> collapse-proofing) are the *target*, not a "use softmax instead" signal.

## 0. What the fab currently searches — the ceiling

The current exploration vocabulary (drawn from `component_fab/{proposer,improver,inventor,generator}`):

**Algebraic spaces** (`op_algebraic_space` in `code_generator._dispatch_*`):
euclidean, tropical, clifford, spiking, padic, complex (declared in `_NOVEL_SPACES`,
**NOT dispatched**), quaternion, hyperbolic / hyperbolic_poincare, linear_memory,
fast_weight_memory, slot_memory, hierarchical_residual_state, symplectic_residual,
tropical_surprise_memory, semiring_surprise_memory{,_rope}, padic_surprise_memory.

**Math-knob families** (`DEFAULT_MATH_KNOBS` + knob catalog): calculus (finite-diff
integral), linear_algebra (low-rank), sparse_matrix (banded), kernel_methods
(positive random features), multiscale (Haar wavelet), graph_diffusion (path
Laplacian), information_geometry (Fisher attention), spectral_graph (Chebyshev),
tensor_decomp (Tucker).

**Atom kinds** (`research/synthesis/parametric_atoms.ATOM_KINDS`): norm, basis,
scan, mlp. **Stage kinds** (`parametric_ops.StageSpec`): dot / cosine / reciprocal
address × softmax / sharpen score-norm × mean / semiring aggregate.

**Routing kinds** (8): depth_router, site_recursion, sparse_depth, low_info_skip,
difficulty, hash, top_k_moe, none.

**Block templates** (12): latent_compress, three_lane_adaptive, recursive_depth,
gated_parallel, loss_monster_paired, recursive_depth_router, sparse_moe_block,
hetero_moe_block, hyperbolic_bridge, attn_spectral_filter, graph_attention,
top_ar_block.

**Invention blueprints** (9): data_dependent_decay_memory, causal_fast_weight_memory,
causal_slot_router_memory, hierarchical_residual_compressor, symplectic_residual_mixer,
tropical_surprise_memory, semiring_surprise_memory{,_rope}, padic_surprise_memory.

This is ~150 effective blueprints after Cartesian combinations. The known softmax
twins in this set: reciprocal/semiring/phase-lock/sparsemax/tropical families all
share the **same total-order score → weighted-mean aggregate** skeleton — the
novelty is in the *score geometry*, not the *structure*. (This is the 2026-06-14
"softmax-twin" pathology.)

## 1. The big capability walls (target list, not a fallback)

Per `research/notes/novel_mechanism_architecture_redesign_2026-06-14.md` and
`.current_work.md`:
- **(W1) Compositional / multi-slot binding** — every non-QKV lane ≈ chance at 2+ triples.
- **(W2) Long-gap / AR held-pair / AR depth (S4+)** — single-pass gate cannot route deep enough.
- **(W3) Gate collapse** — softmax-router entropy drops to 0; learned scoring degenerates to single-expert.
- **(W4) Loss-monster gap** — fast next-token learners trade capability for low loss; pairing is mandatory.

Each axis below is graded on (novelty vs softmax twin, axis-of-the-wall it could
unlock, effort-to-wire).

---

## 2. Ranked proposal of novel axes (high → lower effort-to-novelty)

### Tier-1 — novel by structure, not by score-geometry. Highest priority.

#### A. **Tsallis / Rényi q-softmax family** — beat softmax/softmax-twin on the *score-norm* axis, not by clever addresses
**Why novel:** softmax = minimizer of KL divergence (Gibbs variational). Replace
the KL free-energy minimizer with the **Tsallis q-entropy** (q ≠ 1) minimizer.
q = 1 recovers softmax. For q > 1 the map becomes the sparse `entmax` family
(sparsemax is the q = 2 member; the support shrinks toward a hard top-k as
q → ∞). For q < 1 it is denser than softmax (uniform limit as q → 0⁺). The full
(q, β) surface is a 2-parameter family where softmax (q=1), sparsemax (q=2),
top-k (q→∞), and the intermediate entmax maps are all specific points — and most
of the surface has **never been instantiated for a sequence mixer.** Currently
the fab has `op_physics_score_norm_family=softmax|sharpen` only.

**Wall it could unlock:** W1 (compositional binding) — sparse, capacity-bounded
score norms naturally produce the per-slot "winner-takes-this-slot" selectivity
that softmax blur cannot; entmax-2 already has literature wins on multi-label /
structured prediction.

**Implementation:**
- New file: `component_fab/generator/primitive_templates/_tsallis_mixer.py`.
- Score-norm transform with `q` (init 1.0 = softmax, trainable) and a learned
  sharpening `β`. q-exponential path:
  `weights_j = [1 + (1-q)*(scores_j - tau)]_+^(1/(1-q))` (normalised) where `tau`
  is the threshold solved by bisection. cuBLAS-friendly, 50 lines.
- **Wiring (correct subsystem):** score-norm is applied in
  `research/synthesis/parametric_ops.py` (`StageSpec` + `SCORE_NORM_FAMILIES`) —
  NOT in `code_generator._dispatch_math_knob`, which routes on `op_math_family`
  and is a separate subsystem that never sees score-norm. Extend
  `SCORE_NORM_FAMILIES` with `("tsallis_q", "renyi")` and add the q-exponential
  branch to the `StageSpec` score-norm forward path.
- Proposer axis: add `tsallis_q`/`renyi` as sampleable values of
  `op_physics_score_norm_family` (currently `{softmax, sharpen}`).
- Repair rule in `dynamic.py::_REPAIR_RULES`: a new `repair_score_norm_spectrum`
  trigger (when `nb_max_accuracy < 0.62` AND no explicit score-norm was tried)
  emitting a 5-spec sweep over `op_physics_score_norm_family` with
  `q ∈ {0.5, 0.9, 1.5, 2.0, 3.0}`.

**Cost:** ~150 LOC + 30 LOC tests. No new axes, just a richer
`op_physics_score_norm_family` vocabulary. Will compose with every existing
address/aggregate combination.

#### B. **Sheaf diffusion mixer** — *the* likely compositional-binding fix
**Why novel:** a sheaf assigns to every open set U of the position poset a vector
`F(U)` and a restriction map `ρ_{U⊆V}: F(V) → F(U)`. Two windows that *overlap*
on tokens must agree on the overlap. Diffusing on the sheaf Laplacian then
**forces the same token to be consistently represented across every window that
contains it.** This is the principled fix for compositional binding: each slot
gets its own *section* of the sheaf; agreement on overlaps prevents the "right
content, wrong slot" failure mode (`novel_mechanism_architecture_redesign_2026-06-14.md`,
Run-1 diagnosis).

Currently the fab has nothing sheaf-shaped. `graph_diffusion` is on a fixed
*causal-path* topology; `hierarchical_residual_compressor` is levels-of-summary,
not local-to-global consistency.

**Wall it could unlock:** W1 (compositional/multislot binding). Likely also W2
(long-gap) because the sheaf Laplacian is exactly the *long-range consistency*
operator.

**Implementation:**
- New file: `component_fab/generator/primitive_templates/_sheaf_mixer.py`.
- `class SheafMixer(nn.Module)`: window-size `w ∈ [2, 8]` learned per layer;
  per-token learned restriction `ρ_t: R^{2D} → R^D` (concatenated to a
  window); per-head sheaf Laplacian `L = D - A` (incidence of the window
  hypergraph) precomputed at init; diffusion `H = (I + αL)^k · x`. Stochastic
  depth on `k` ∈ [1, 4] for cheapness.
- Dispatcher entry: `op_algebraic_space = "sheaf_diffusion"` (new value;
  _NOVEL_SPACES-friendly; failure mode already validated).
- New invention blueprint: `sheaf_consistent_slot_mixer` in
  `inventor/mechanism_catalog.py` (mirror `semiring_surprise_memory_rope`
  structure — has its own complexity/forgetting/causality contract).
- Test: `tests/test_sheaf_mixer.py` — overlaps-must-agree + non-QKV-no-QK proof.

**Cost:** ~300 LOC + tests. This is a real *new mechanism*, not an adapter.

#### C. **Fractional-derivative memory** — power-law decay, not exponential
**Why novel:** Mamba/SSM use `s_t = α·s_{t-1} + (1-α)·x_t` (exponential decay).
The **Riemann-Liouville fractional integral** `I^α` (order `α ∈ (0,1)`) has kernel
`(t-s)^{α-1}/Γ(α)`, so the influence of a token at lag `τ` decays as `τ^{α-1}` —
a *power law*, not exponential. This is an accumulating (low-pass) memory
operator, the opposite of the fractional *derivative* (a high-pass differencer);
the integral is what actually gives "remembers everything, just dimly." The
result: a global state with *infinite* effective memory length and bounded
gradient — exactly the Mamba weakness (finite effective horizon) transposed into
a mechanism Mamba cannot match.

Currently `op_dynamical_memory_length_class ∈ {O(1), O(L), O(L^2), O(log L)}` is
a *descriptor* bucket (how state complexity scales), not a dispatch key. The
fractional integral is a new memory *regime*: a power-law-decaying kernel that
sits in none of those buckets.

**Wall it could unlock:** W2 (long-gap, AR held-pair, AR depth). The
`padic_surprise_memory` and `semiring_surprise_memory` are bounded state;
the fractional integral is *unbounded* state with `lag^(α-1)` power-law decay —
first class that "remembers everything, just dimly."

**Implementation:**
- New primitive `class FractionalIntegralLane(nn.Module)` (fractional *integral*,
  accumulating). Grünwald-Letnikov integral discretisation:
  `y_t = sum_{k=0}^{K-1} w_k · x_{t-k}` with **positive** weights
  `w_k = Γ(k+α) / (Γ(k+1) Γ(α))` (∝ `k^(α-1)`, a power-law-decaying causal
  kernel) and learnable `α ∈ (0,1)`. Truncate to `K = 256`. Closed-form gradient.
  At `α → 1` the weights flatten to a running sum (longest memory); at `α → 0`
  they collapse to the current token (near-identity). (Note: the *derivative*
  form `(-1)^k C(α,k)` is a differencer — do NOT use it here; it does not
  accumulate memory.)
- Dispatch via a new **invention blueprint** `fractional_integral_memory`
  (mirror the `padic_surprise_memory` contract) — NOT via the
  `op_dynamical_memory_length_class` descriptor axis. Optionally add a descriptor
  value `"O(L^alpha)"` to `parametric_atoms` for classification/telemetry only.

**Cost:** ~250 LOC + 30 LOC tests. Native CUDA reduction available via
`torch.cumsum` after re-indexing.

#### D. **MERA / multi-scale entanglement renormalization** — strict coarse-graining as a block
**Why novel:** the current `hierarchical_residual_compressor` is level-wise gated
summaries, *not* the strict tensor-network renormalization of the MERA ansatz
(Vidal 2007; cf. Evenbly & White's later tensor-network-renormalization / TNR
scheme, 2015). MERA is alternating layers of *disentanglers* (U) and *isometries* (W)
that strictly coarse-grain the lattice. Each layer halves the spatial resolution;
the readout is the top-site. This is a **principled hierarchical renorm group** —
the mathematical structure underlying Wilson's RG, multi-resolution analysis,
and entanglement entropy. It has never been searched for LLM sequence mixing.

**Wall it could unlock:** W1 (compositional binding — MERA's disentanglers are
literally designed to *remove* correlations between scales, so each scale
holds a cleanly bounded chunk of binding info) and W2 (long-range — coarse-grain
is exponentially compressed over depth).

**Implementation:**
- New block template `mera_block` in
  `component_fab/generator/block_templates.py`. Input: per-block tensor
  `(B, L, D)`. Internals: `U_l ∈ R^{d×d}` disentanglers per level,
  `W_l ∈ R^{d×d/2}` isometries per level (causal: only past tokens). 3 levels
  per block (L → L/2 → L/4 → top site). Readout: linear combination of
  coarse-graining outputs.
- Mechanism blueprint: `mera_block` (an invention track entry).
- Memo: MERA also gives a free `t_n_metrics` — entanglement entropy across
  scales, useful for the capability probes.

**Cost:** ~400 LOC + 50 LOC tests. The math is well-documented; the implementation
is a stack of 2-layer MLPs.

### Tier-2 — strong novelty, slightly higher effort, each opens a new sub-domain.

#### E. **Doob decomposition / martingale-compensator layer** — separates "local rule" from "global memory" exactly
**Why novel:** any stochastic process `X_t` decomposes uniquely into a martingale
`M_t` (zero predictable drift) plus a predictable compensator `A_t` (the "local
rule"). Applying this to hidden states: `h_t = M_t + A_t` where `A_t =
Σ_{s≤t} E[ΔX_s | F_{s-1}]` (the predictable part) and `M_t = X_t - A_t` (the
"surprise" / global). The layer can mix them separately. This is the
mathematically *exact* way to separate "token-by-token local predictability"
from "global memory" — the two things the fab currently entangles in every stateful
primitive.

**Wall it could unlock:** W4 (loss-monster pairing) — the local rule is exactly
the next-token predictability; the martingale is exactly the long-range memory.
A loss-monster carrier would have an architectural decomposition that other
layers don't.

**Implementation:**
- New primitive `class DoobLayer(nn.Module)`. Forward: compute `A_t` via a
  single MLP over the past, subtract, learn `M_t`'s contribution via an EMA,
  output `α·A_t + β·M_t + γ·x_t`. ~80 LOC.
- New axis `op_decomposition = "doob_martingale"` (one value, but as a flag
  it opens the design space to other decompositions: Wiener, Lévy, etc.).
- Invention blueprint: `martingale_compensator_memory`.

**Cost:** ~200 LOC.

#### F. **Braid-group / string-topology mixer** — token sequence as a braid
**Why novel:** the braid group `B_n` on `n` strands is the fundamental group of
the configuration space. A sequence of `n` tokens can be drawn as `n` strands
in 2D; each adjacent transposition is a crossing. A braid `β ∈ B_n` defines a
representation `R(β) ∈ GL(D)` applied to the per-token vector. This is *not*
position encoding — it is the **order in which tokens entangle**, learned.
Two tokens that swap meaning several times can be assigned a non-trivial
crossing pattern. Combinatorial geometry, novel sequence structure.

The fab has no combinatoric/topology primitive. Closest is
`graph_attention` (edge-conditioned attention), but that's still attention.

**Wall it could unlock:** W1 (compositional binding — strands have fixed
identity across crossings, so a token "remembers itself" through composition).

**Implementation:**
- New primitive `class BraidMixer(nn.Module)`. Limit to `n=8` strands per
  layer (128 crossings max, fully enumerable). Learn a continuous braid
  representation `R: B_n → GL(D)` via the Burau representation or a small
  transformer over crossings. Output = `R(β) · x_t`. ~300 LOC.
- New axis `op_algebraic_space = "braid_group"`.
- Cost: the math is well-known (Kassel–Turaev), but the differentiable
  representation is non-trivial — start with Burau (linear) for identity-at-init.

**Cost:** ~350 LOC. Likely needs a `StrandBuffer` helper for variable strand count.

#### G. **Plackett-Luce / partial-order routing** — replace total-order scores
**Why novel:** softmax attention produces a **total** order over keys (every
weight > 0). `Plackett-Luce` is a probability over *permutations* — the model
outputs a per-item score and samples/reads a permutation. Restricting to partial
orders = chains and antichains = a clean way to say "tokens A,B,C share this
context, D doesn't." A *partial* order generalises the total order softmax
produces, with subsets (chains, antichains) corresponding to known structures
(buckets, sliding windows).

**Wall it could unlock:** W1 (compositional binding — partial orders can express
"these three tokens are independent") and W3 (gate collapse — collapsing
partial orders is much harder than collapsing total orders).

**Implementation:**
- Extend `parametric_ops.py` `AGGREGATE_FAMILIES` with `("plackett_luce",
  "partial_order_chain", "partial_order_antichain")`.
- New primitive `class PlackettLuceAggregator(nn.Module)` — learns per-token
  scores; outputs a learnable kernel over the partial order (chain or antichain).
- New axis `op_aggregate = "partial_order"` in `code_generator`.

**Cost:** ~250 LOC. Sampling is non-differentiable, use Gumbel-Top-K or
REINFORCE-style estimator.

#### H. **Tensor-network / MPS sequence mixer** — strict matrix-product state
**Why novel:** Matrix Product States (MPS, aka tensor trains) compress a
sequence of vectors into a chain of 3-tensors `A[i] ∈ R^{r_{i-1}×d_i×r_i}`.
Bond dimension `r` controls compression. An MPS-mixer reads the sequence as a
MPS, computes a contraction, emits a new MPS. **The bond dimension is a
learnable compression parameter that strictly upper-binds memory.** This is
the cleanest formulation of "bounded-memory" sequence mixing — and it has
been *the* formalism in quantum many-body physics for 30 years, but
essentially unstudied for LLM sequence mixing (one 2024 paper only).

**Wall it could unlock:** W1 (compositional — MPS disentangles left/right
environments) and W2 (long-range — MPS bond captures exact O(L) correlations
within bond dimension).

**Implementation:**
- New primitive `class MPSLane(nn.Module)`. Bond dim `r ∈ [4, 32]` learned.
  Forward: decompose input into per-token tensors, contract via
  `torch.einsum("bir,rjr,bjs->bis", A_left, A_center, A_right)`. ~300 LOC.
- New axis `op_algebraic_space = "matrix_product_state"`.
- Invention blueprint: `mps_bounded_memory_mixer`.

**Cost:** ~350 LOC. MPS gradients are clean; the contraction cost is O(L·r²).

### Tier-3 — high-novelty, deep-research effort, unlock genuinely new math domains.

#### I. **Discrete exterior calculus (DEC) mixer** — tokens as discrete forms
**Why novel:** DEC gives a coordinate-free discrete analogue of calculus on
simplicial complexes. Token stream = a discrete 1-form on a simplicial complex;
mixer = discrete exterior derivative `d` (incidence matrix) + Hodge star `*`
(metric). The identity `d² = 0` is automatic (the boundary of a boundary is
zero: `∂∂ = 0`, dually `d∘d = 0`). Cohomology = closed-but-not-exact forms; this is a **clean
topological invariant of the sequence.** A "closed" token sequence means
`dω = 0`; "exact" means `ω = dα`. Sequence-mixing = the Hodge Laplacian
`Δ = d* d + d d*`.

This is topology-meets-algebra-meets-LM, and the fab has **zero** topology in
its inventory.

**Wall it could unlock:** W1 (compositional — cohomology captures the
"irreducible" relations), W2 (long-range — `Δ^k ω` decays on the simplicial
complex's spectral gap, which is content-dependent).

**Implementation:**
- New primitive `class DECMixer(nn.Module)`. Build the simplicial complex
  from token embeddings via a learned threshold (e.g., cos-sim > 0.5).
  Compute `d`, `*` once per forward (sparse). Output `Δx`. ~400 LOC.
- New axis `op_algebraic_space = "discrete_exterior_calculus"`.

**Cost:** ~450 LOC.

#### J. **Persistent-homology descriptor** — multi-scale topological fingerprint
**Why novel:** given a sequence of token vectors, build a Vietoris-Rips
filtration (points appear at scale 0, edges at scale ε₁, triangles at ε₂,
...). Compute **persistence diagrams** (birth/death pairs of topological
features). The Betti numbers (count of components, loops, voids) at each
scale are a multi-scale topological signature. Use this signature as the
mixer output — a sequence's "shape" rather than its linear projection.

Persistent homology is also a **provably robust** descriptor (stable under
small perturbations in the Gromov-Hausdorff sense), giving the mixer
built-in adversarial robustness.

**Wall it could unlock:** W1 (multi-scale topology captures compositional
structure across scales).

**Implementation:**
- New primitive `class PersistenceDescriptor(nn.Module)`. Persistent
  homology computation via `gudhi` (external) or a learned
  differentiable approximation (e.g., DTM-based persistence images).
  Output: vector of Betti-curve values. ~400 LOC.
- Could be an *atom kind*, not just a primitive — fits naturally into
  `parametric_atoms.ATOM_KINDS` as `"topology"`.

**Cost:** ~450 LOC + external dep.

#### K. **q-deformed Hopf algebra (quantum group) splitter** — exact antipode mix
(Note: the "q" here is the quantum-group deformation parameter — unrelated to
the Tsallis entropy "q" of item A/N.)
**Why novel:** a Hopf algebra `(H, μ, η, Δ, ε, S)` carries an antipode `S`
satisfying `μ ∘ (id ⊗ S) ∘ Δ = η ∘ ε` (Hopf's identity). For a hidden
representation, the **antipode** `S: H → H` is the algebra's "charge
conjugation" — it inverts the algebra structure. `q`-deformed Hopf algebras
(quantum groups like `U_q(sl_2)`) give a continuous family interpolating
between Hopf (q=1) and non-Hopf (q ≠ 1) regimes. A mixer that splits a
representation into `Δ(x) = x ⊗ x` (matter) + `S(x) ⊗ S(x)` (antimatter),
mixes them independently, recombines via `μ`, gets a mathematically
principled *symmetric* decomposition with a knob (`q`) that controls how
non-commutative the splitting is.

**Wall it could unlock:** W1 (compositional — antipode is exactly the
"reversed" component needed for symmetric binding).

**Implementation:**
- New primitive `class QuantumGroupHopfMixer(nn.Module)`. Learn `Δ`
  (co-multiplication) and `S` (antipode) as linear maps, regularise via
  `||μ(S(x) ⊗ S(x)) - x||²` (Hopf identity penalty). ~300 LOC.
- New axis `op_algebraic_space = "quantum_group_hopf"`.

**Cost:** ~400 LOC. Needs regularisation to enforce Hopf identity.

#### L. **Lambda-calculus / SKI combinator mixer** — the planned WS-G, made first-class
**Why novel:** combinatory logic is a *complete* computational basis.
`S`, `K`, `I` (=`SKK`) generate every lambda term. A mixer that learns
continuous `S` and `K` ops (interpolation from identity-biased init) gives a
**Turing-complete** sequence mix in a finite-parameter budget. This is the
WS-G planned workstream (`research/notes/dynamic_preassembly_math_sweep_plan.md`
workstream G), but as a *first-class primitive*, not just a knob.

**Wall it could unlock:** W1 (compositional — combinators compose
hierarchically, so binding structure IS the combinator tree), W2 (long-range
— combinator `S` is essentially the `M` combinator for memoization).

**Implementation:**
- Make `op_math_family = "lambda_functional"` first-class
  (per WS-G). New primitive `class LambdaCombinatorMixer(nn.Module)`.
  Init: `K = 1, S = 0` (so all 3 ops reduce to `I` = identity at init).
  Train: `K` and `S` move continuously. ~200 LOC.
- Dispatcher entry.

**Cost:** ~250 LOC.

#### M. **Möbius / conformal ball mixer** — angle-preserving, not orthogonal
**Why novel:** the Poincaré ball model of hyperbolic space uses Möbius
arithmetic — the gyrovector Möbius addition (curvature `c`):
`x ⊕_c y = ((1 + 2c⟨x,y⟩ + c‖y‖²)·x + (1 − c‖x‖²)·y) / (1 + 2c⟨x,y⟩ + c²‖x‖²‖y‖²)`.
But the **conformal ball** model uses *conformal* transformations: angle-preserving
but NOT orthogonal (not Lie-group). A learned conformal mixer is a different
fixed-point structure from softmax (whose exp-normalize is also conformal but
projective). Specifically: Möbius has `||φ(x) - φ(y)|| / ||x - y|| ∈ [c, C]`
(distortion bounds). The geometry of fixed points is *qualitatively different*
from softmax (projective → global scale) and Poincaré (hyperbolic → boundary
clustering).

**Wall it could unlock:** W1 (compositional — Möbius's distortion bound gives
a clean per-pair proximity measure for slot assignment).

**Implementation:**
- New primitive `class MobiusMixer(nn.Module)`. Möbius addition + Möbius
  matrix multiply in the ball. `||x|| < 1` constraint via tanh projection
  on inputs. ~250 LOC.
- New axis `op_algebraic_space = "mobius_conformal"`.

**Cost:** ~300 LOC. Math is well-documented (Ungar 2008).

#### N. **Bregman-divergence mixer** — softmax is one of many
(Complementary to item A: A generalizes the *entropy* (Tsallis q) while holding
the divergence at KL; N generalizes the *divergence* while holding the entropy
fixed. Different axes of the same "softmax is one special case" idea.)
**Why novel:** softmax is the Gibbs variational principle for the **KL**
divergence. Different Bregman divergences yield different aggregators:
- KL → softmax (current default)
- Itakura-Saito → `exp(softmax-style)`-in-log-domain
- Hellinger → sqrt-weight mixing
- Logistic → log-sum-exp with offset
- Total-variation → projection-mixing

Each defines a different "softmax analog" with a different bias-variance
tradeoff. The fab currently has only softmax + sharpen (which is a
temperature scaling of softmax = same Bregman).

**Wall it could unlock:** W3 (gate collapse — different Bregman divergences
have different collapse dynamics).

**Implementation:**
- Add `op_bregman_divergence ∈ {kl, itakura_saito, hellinger, logistic}`
  to `op_math_family`. Each spawns a new aggregator in
  `parametric_ops._aggregate`. ~200 LOC total.

**Cost:** ~250 LOC.

### Tier-4 — out-of-the-box exploration, low immediate payoff, opens new domains for follow-on work.

#### O. **Free probability / free-convolution mixer** — non-commutative probability
The state transition is a `free` convolution (`μ ⊞ ν` computed via R-transform
multiplication), not a regular matrix multiply. Random-matrix asymptotic
behaviour — strongly tied to SSM theory (Voiculescu). Mathematically deep,
computationally feasible via moment generating functions.

**Cost:** ~500 LOC, needs external `ncpol2sdpa` or moment computation library.

#### P. **Lattice-valued / Riesz-space lane** — order-theoretic mix
Operations are `sup`, `inf`, lattice-addition, absolute value. Order-theoretic
semantics distinct from tropical (which is only max-plus). The fab has
**zero order theory**.

**Cost:** ~300 LOC.

#### Q. **Linear-logic / proof-net mixer** — resource-sensitive
Formulas-as-tokens, sequent calculus as the mixing. Same token used twice
= false (resource-sensitive). Naturally prevents the "double-count"
pathology of softmax attention.

**Cost:** ~400 LOC + new evaluation probes (sequent-style).

#### R. **De Sitter / Lorentzian mixer** — light-cone causal structure
The token stream defines a 1+1D Lorentzian manifold; the mixer respects the
light-cone (causality) structure. Genuinely *non-Euclidean geometry* at the
position axis, distinct from hyperbolic/Poincaré.

**Cost:** ~350 LOC.

#### S. **Persistent homology *filter* (vs. descriptor)** — apply topology to mix
Use Betti numbers as a per-token filter mask: tokens whose local topology
"looks like the rest" pass through, outlier tokens are gated. Complementary
to the descriptor version.

**Cost:** ~300 LOC.

---

## 3. Cross-cutting additions (apply to every tier)

### Add a `novelty_audit` pre-check
Every blueprint should be checked at proposal time against the
**softmax-twin signatures**:
- `address=dot + score_norm=softmax + aggregate=mean` (vanilla softmax)
- `address=reciprocal + score_norm=softmax` (reciprocal = log-softmax on
  transpose — a softmax twin per the 2026-06-14 note)
- `address=cosine + score_norm=softmax` (cosine + softmax = a softmax twin)
- `address=any + score_norm=sharpen + aggregate=mean` (sharpen = τ·softmax → softmax twin)

Emit a `softmax_twin_risk ∈ [0, 1]` score in the ledger metadata. Repair rule
in `dynamic.py::_REPAIR_RULES`: when `softmax_twin_risk > 0.7` AND
`nb_max_accuracy < 0.62`, force a `op_physics_score_norm_family` or
`op_physics_aggregate_family` change to a non-twin family.

### Add an **exploration budget** to the dynamic proposer
Per cycle, reserve a fixed fraction (e.g., 15%) of slots for novel axes whose
ledger entries are < 3. This stops the lift-prior from over-exploiting
explored regions and forces continued novelty.

### Add a **per-axis novelty prior** to `axis_lift.py`
When a value's `n=0` (never tried), give it a small but nonzero lift
(`novelty_prior = 0.05` floor), so the adaptive sampling still visits
unexplored regions. Currently zero-evidence axes get `lift ≈ prior_strength / global_n`
which decays toward zero as the ledger grows.

---

## 4. Suggested shipping order (one-week plan, prioritised)

1. **A. Tsallis q-softmax** (Tier-1, 1 day) — first because it composes with
   everything else and is the most "one-knob" change.
2. **C. Fractional-derivative memory** (Tier-1, 1.5 days) — first new state
   primitive in months; opens W2.
3. **N. Bregman-divergence mixer** (Tier-3 but cheap, 1 day) — softmax-twin
   killer.
4. **B. Sheaf diffusion mixer** (Tier-1, 2 days) — the W1 fix per the 2026-06-14
   diagnosis; the highest-value mechanism on this list.
5. **D. MERA block template** (Tier-1, 2 days) — opens the renorm / tensor-network
   sub-domain.
6. **L. Lambda-combinator mixer** (Tier-3 but WS-G planned, 1.5 days) — fulfills
   an open workstream.
7. **E. Doob martingale** (Tier-2, 1 day) — the W4 pairing mechanism.
8. **F. Braid mixer** (Tier-2, 2 days) — opens combinatorial/topology.
9. **G. Plackett-Luce** (Tier-2, 1 day) — softmax-twin killer on the
   aggregation axis.
10. **M. Möbius/conformal** (Tier-3, 1.5 days) — completes the geometric
    inventory (Euclidean / hyperbolic / conformal).

End state (after ~2 weeks): the fab searches 10 NEW algebraic spaces, 5 NEW
atom/stage kinds, 4 NEW block templates, and 8 NEW invention blueprints — a
**~3× expansion of the searched vocabulary.** Every new entry is anti-softmax-
twin by construction.

---

## 5. Honest concerns (per CLAUDE.md feedback: "no softmax twin", "fix the
novel mechanism, don't replace it")

- **(C1) Tsallis q-softmax is *adjacent* to softmax.** It IS a softmax family.
  Risk: ends up as a softmax twin. Mitigation: enforce
  `op_physics_score_norm_family ∈ {tsallis_q, renyi}` as the ONLY non-default
  values for the new rule; the default `softmax` remains a repair, not a primary.
- **(C2) Möbius / conformal is geometrically close to Poincaré.** Risk:
  becomes a Poincaré twin. Mitigation: enforce that the `op_conformal_scale`
  knob starts at a value that breaks Poincaré symmetry (e.g., 0.7, not 1.0).
- **(C3) Doob / martingale is well-known in stochastic calculus, untested in
  LLMs.** Risk: produces a noisy / hard-to-train primitive. Mitigation: start
  with a single-step decomposition; add gradient-noise regularisation.
- **(C4) Sheaf diffusion requires a learned complex.** Risk: the learned
  restriction maps collapse to identity. Mitigation: enforce the sheaf
  restriction-consistency (gluing / cocycle) identity
  `ρ_{U∩V} ∘ ρ_V = ρ_{U∩V} ∘ ρ_U` via a regularisation penalty.
- **(C5) MERA is well-studied in physics, but its LLM behaviour is unknown.**
  Risk: disentanglers collapse to identity (no-op). Mitigation: per-level
  load-balance loss on disentangler action magnitudes.

Every Tier-1+ mechanism should ship WITH a *softmax-twin detector* in its
ledger entry (e.g., `softmax_twin_risk=0.05`) and a *collapse detector*
(e.g., `sheaf_restriction_identity_likely=False`).

---

## 6. Validation done in this audit

- Graph-gated (`mcp__code-review-graph__get_minimal_context_tool`,
  `__list_graph_stats_tool` — see CLAUDE.md) — read proposer/validator/
  generator/inventor/state coverage.
- Read in full: `component_fab/proposer/dynamic.py`,
  `component_fab/improver/axis_variants.py`,
  `component_fab/inventor/mechanism_catalog.py`,
  `component_fab/improver/math_knob_catalog.py`,
  `component_fab/proposer/spec_generator.py`,
  `component_fab/state/axis_lift.py`,
  `component_fab/state/failure_attribution.py`,
  `component_fab/state/gates.py`,
  `component_fab/validator/capability.py`,
  `component_fab/improver/ranking.py`,
  `component_fab/harness/capability_probes.py`,
  `component_fab/metrics/compression_quality.py`,
  `component_fab/metrics/compression_probe.py`,
  `component_fab/metrics/behavior_fingerprint.py`,
  `component_fab/intake/scope_existing.py`,
  `component_fab/generator/code_generator.py`,
  `component_fab/generator/primitive_templates/__init__.py`,
  `component_fab/generator/primitive_templates/_core.py`,
  `component_fab/improver/cross_anchor.py`,
  `component_fab/validator/solo.py`,
  `component_fab/math_knobs.py`,
  `research/synthesis/parametric_atoms.py`,
  `research/synthesis/parametric_ops.py`,
  `research/notes/novel_mechanism_architecture_redesign_2026-06-14.md`,
  `.current_work.md`.
- Cross-referenced all `op_algebraic_space` values against the literature for
  novel LLM mechanisms; identified 14 distinct math sub-domains not yet searched.
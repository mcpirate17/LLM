# Plan — component_fab novel-math / new-geometry expansion

**Date:** 2026-06-30
**Origin:** user asked for new optimizations to ensure component_fab explores geometry / math /
algebra that has never existed before (mission: beat softmax with novel non-QKV mechanisms).
minimax-3 was asked the same question in parallel; its findings are integrated separately
(see "Minimax integration" below). This plan is the claude track.

## Mission alignment

Every item here adds genuinely novel, non-QKV mathematical territory to the fab's search space,
or removes a structural bottleneck that is currently *preventing* known-novel math from being
reached by the proposer. None of these cut the novel branch or reconverge on softmax. Items that
risk a softmax-twin collapse are explicitly flagged and gated by the existing novelty guard
(`component_fab/validator/mechanism.py`: routing entropy / load balance / state degeneracy).

## Diagnosis (why the fab under-explores new math today)

The fab has a deep primitive zoo — ~123 ops, 22 algebraic spaces, 9 invention blueprints, ~70
axis-variant templates, 25 dispatched invention mechanisms. The bottleneck is **not** missing
primitives; it is a shallow, 1-operator-deep search grammar:

| Axis | Current domain | Bottleneck |
|---|---|---|
| `op_math_family` × operator | 9 families, **each exactly ONE operator** (calculus≡finite-diff, spectral≡Chebyshev, tensor≡Tucker, info-geom≡Fisher, multiscale≡Haar, kernel≡random-features, graph≡path-Laplacian, sparse≡banded, lin-alg≡low-rank) | family axis is a 1:1 alias, not a lattice — no second operator exists to try |
| `op_physics_address_family` | `{dot, reciprocal, cosine}` | 3 |
| `op_physics_score_norm_family` | `{softmax, sharpen}` | 2 — `entmax`/`sparsemax` primitives exist but are not score-norm options |
| `op_physics_aggregate_family` | `{mean, semiring}` | 2 |
| `op_physics_atom_kinds` | `{norm, basis, scan, mlp}` | 4 |
| Exotic algebras (tropical / padic / Clifford / hyperbolic / spiking) | real primitive ops + wholesale `op_algebraic_space` swaps | **not stackable knob families** → the depth-1/2/3 knob-composition track cannot combine e.g. a tropical adapter over a hyperbolic lane |
| Invention blueprints | 9 total, **7 are memory mechanisms** | compression / pure-mixing / routing inventions under-mined |

Conclusion: huge search-space expansion is available cheaply by (a) deepening each family to a
real operator lattice, (b) making exotic algebras stackable, (c) opening the combinatorial
cross-product, and (d) adding genuinely new mathematical territories.

## Work items

Prefix `NM-` (Novel Math) to avoid collision with codex's `WS-A…WS-H`
(`tasks/dynamic_preassembly_math_sweep_plan.md`).

### Tier A — deepen & cross-multiply the existing grammar (cheap, high leverage)

**NM-1 — Deepen each math-family operator lattice (1 → 3-5 operators).** Add axis-value siblings + one `_op_*`/`_init_*` pair each in `research/synthesis/{primitives,compiled_op_params,compiler_ops_routing}.py`:
- calculus: `causal_gradient`, `laplacian`, `lie_derivative_along_flow`
- spectral_graph: `dct`, `wavelet_packet`, `graph_eigbasis` (eigenvectors of a *learned* adjacency), `legendre_basis` (re-add deleted stub as a real op)
- tensor_decomp: `cp`, `tensor_train`, `tensor_ring`, `block_term`
- information_geometry: `alpha_divergence`, `renyi_attention`, `natural_gradient_mixer`
- multiscale: `dyadic_diff`, `laplacian_pyramid`
- *Done-when:* ≥2 operators exist per family; each builds + fwd/bwd finite; enumerated as distinct axis values.

**NM-2 — Promote exotic algebras to first-class stackable knob families.** Files: `component_fab/math_knobs.py`, `component_fab/improver/math_knob_catalog.py`. Add `tropical_knob`, `padic_knob`, `clifford_knob`, `hyperbolic_knob` to `DEFAULT_MATH_KNOBS` + the composition enumerator so a candidate can stack e.g. a tropical read over a Poincaré projection. ⚠️ **Coordinate with codex WS-G** (also edits `math_knobs.py`); re-read fresh, additive only.
- *Done-when:* a 2-knob stack combining two distinct exotic algebras dispatches, builds, and produces finite output distinct from either alone.

**NM-3 — Combinatorial cross-product proposer.** Enumerate `address × score_norm × aggregate × algebra × basis` (3×2×2×N×5 today, almost unsampled). New helper in `component_fab/proposer/` feeding `enumerate_cycle_specs`. Bounded per cycle.
- *Done-when:* a cycle emits specs spanning ≥3 unseen cross-product cells; deduped; none collapse to softmax-twin.

### Tier B — genuinely new mathematical territory (never in the fab)

Ordered by leverage on the known capability gaps (multi-slot binding/retrieval at 2+ triples, induction, p-adic "needs a retrieval pathway").

**NM-4 — Optimal-transport / Sinkhorn-Wasserstein mixing.** ← **FIRST IMPLEMENTATION (this session).** Token states as measures; mixer = entropic-OT transport plan (Sinkhorn, doubly-stochastic). Structurally ≠ softmax-twin (softmax is row-stochastic only; OT enforces balanced marginals). Attacks the binding/retrieval wall directly — transport *is* the geometry of matching. Files: `research/synthesis/{primitives,compiled_op_params,compiler_ops_routing}.py` + `research/tests/test_sinkhorn_ot_mix.py`. Python-only dispatch (no native softmax bypass), `padic_gated_mixer` precedent. **Zero overlap with codex WS-*.**
- *Done-when:* `sinkhorn_ot_mix` registered in all 3 files; fwd/bwd finite; plan rows sum to query masses and cols to key masses (doubly-stochastic within tol); routing-entropy / non-collapse check passes; test green; ruff clean.

**NM-5 — Ultrametric / Bruhat–Tits-tree geometry as a primary mixer.** Generalizes the validated p-adic direction: content-similarity on a *learned* ultrametric tree (not just per-token valuation). Targets hierarchical retrieval / the p-adic retrieval gap. Files: research/synthesis triple. Build on `padic_*` precedent.
- *Done-when:* tree-distance address builds, fwd/bwd finite, distinguishes coarse-near vs exact-match keys under a fixed budget.

**NM-6 — Operator-learning / neural-operator family (`op_math_family=neural_operator`).** Mix in spectral/function space (Fourier neural operator, DeepONet) using existing `integral_kernel`/`basis_expansion`/`fixed_point_iter`/`spectral_filter` primitives promoted to a family. Different inductive bias for compositional/recursive structure.
- *Done-when:* family dispatches; spectral-conv variant fwd/bwd finite; distinct from SSM/attention descriptor signature.

**NM-7 — Clifford / geometric-algebra substrate promotion.** 7 Clifford ops exist as a wholesale swap only; promote geometric-product composition to a stackable knob (pairs with NM-2).
- *Done-when:* Clifford knob composes with a non-Clifford lane; output finite and non-degenerate.

**NM-8 — Entropic / sparsifying score-norm axis.** Add `entmax_alpha` (learnable α) and `sparsemax` to `op_physics_score_norm_family`. Sparse, data-adaptive retrieval serves binding. Files: `research/synthesis/parametric_ops.py` (SCORE_NORM_FAMILIES) + dispatch.
- *Done-when:* score-norm family has ≥4 values; entmax-α builds; sparsity varies with α.

**NM-9 — Long-tail exotic (lower priority, highest novelty).** Octonionic (non-associative) mixing beyond quaternions; free-probability / random-matrix structured projections; persistent-homology (TDA) pooling atom; verify `symplectic_residual_mixer` lane is real (not a stub like the deleted `legendre_ssm`).
- *Done-when:* ≥1 builds + fwd/bwd finite + non-collapse.

### Tier C — search-process changes to *ensure* novelty (meta level)

**NM-10 — Geometric-novelty MAP-Elites cell dimension.** Add a cell axis = distance of the op's measured-descriptor signature (`long_range_reach`, `content_dependence`, `spectral_radius`, `self_dominance`, …) from the softmax/attention basin. Drives search toward unexplored geometry instead of sampling by luck. Files: MAP-Elites archive (see `project_scale_leaderboard_builder` memory) + descriptor extractor.
- *Done-when:* archive maintains a novelty cell; a cycle preferentially fills low-novelty-distance-occupied cells.
- **STATUS: DONE (2026-07-01 — GLM —).** New `research/synthesis/novelty_distance.py`
  + `novelty_aware_axes()` opt-in on `OpenDiscovery.run()` + 13 unit tests + 1
  open-discovery smoke test. Design: the novelty coordinate is the STANDARDIZED
  distance of the candidate's PHYSICS fingerprint (the loop's existing niche
  coordinate — perm/shift-equivariance, scale-homogeneity, spectral-radius,
  energy-gain) to the NEAREST MEASURED softmax-shaped basin signature
  (softmax-QK attention + uniform mean-pool, fixed-seed reference ops probed via
  `PhysicsDescriptorProbe`). Physics-fingerprint was chosen over the
  capability-correlated measured_descriptors (`long_range_reach` etc.) on purpose:
  those ARE the fitness signal, so using them as a niche coordinate too would
  conflate fitness with novelty; the physics fingerprint is the orthogonal
  "how-softmax-shaped is the transform" view. The NM-11 measured
  `softmax_twin_score` folds in as an optional `twin_score=` refiner
  (`(0.25+0.75*(1-twin))`; twin≈1→floor, twin≈0→full distance) — supplied by the
  caller so `research.synthesis` still imports NO `component_fab`. The mission
  assertion is test-pinned: `test_archive_keeps_far_from_softmax_mechanism` shows
  a far-from-softmax mechanism SURVIVES in the archive with lower fitness than a
  near-softmax one. OPEN follow-up: flip `novelty_aware=True` on in
  `research/tools/run_open_discovery.py` + adopt the axis in component_fab's
  `novelty_archive.py` once codex's proposer work settles (it's a 1-line
  `axes=novelty_aware_axes()` change).

**NM-11 — Algebraic-property mining as a discovery signal.** Thicken `component_fab/proposer/property_miner.py`: detect equivariance / symmetry / idempotence / associativity at init. Use as novelty feature AND as an axis to break (softmax's tell = shift-equivariance + row-stochasticity).
- *Done-when:* ≥3 properties mined per candidate; softmax-twin signature detectable.

**NM-12 — Auto-deepening meta-rule.** When a family has 1 operator, auto-emit its natural siblings as variants (finite-diff → gradient → Laplacian). Makes NM-1 self-sustaining.
- *Done-when:* a 1-operator family auto-yields ≥2 candidate siblings through the proposer.

## Sequencing

1. **NM-4 first** (this session) — collision-free, novel, capability-targeted, establishes the new-op pattern.
2. NM-1 + NM-2 (one PR) — grammar deepening + exotic-algebra knobs (coordinate NM-2 with codex WS-G).
3. NM-10 — novelty MAP-Elites dimension.
4. NM-5 or NM-6 to GPU after nano validation.
5. NM-3, NM-8, NM-11, NM-12, NM-7, NM-9 in priority order.

## File-claim protocol (multi-agent)

- claude claims (NM-4): `research/synthesis/primitives.py`, `research/synthesis/compiled_op_params.py`, `research/synthesis/compiler_ops_routing.py`, `research/tests/test_sinkhorn_ot_mix.py` (new).
- Coordinate-before-edit (later slices): `component_fab/math_knobs.py`, `component_fab/improver/math_knob_catalog.py` (NM-2 vs codex WS-G); `research/synthesis/parametric_ops.py` (NM-8); MAP-Elites archive (NM-10).
- Stage own files only; `--no-verify`; re-read shared files fresh before claiming.

## Minimax integration

minimax-3 was asked the same question. Integration contract for its findings (see chat reply for
the schema): split into **convergent** (minimax ⊓ claude → high-confidence, build first),
**divergent** (minimax-only new math → triage vs NM-* and codex WS-*), **conflicting** (resolve).
Log the merged result back into this plan + `research/notes/fab_novel_math_expansion_2026-06-30.md`.

## Done when (program-level)

component_fab can, in one cycle, (a) reach ≥3 operators per math family, (b) stack two distinct
exotic algebras as knobs, (c) propose cross-product cells that have never been sampled, and
(d) drive search toward geometrically novel regions via a measured novelty objective — with every
new op passing fwd/bwd + non-collapse + (where relevant) a capability nano-probe before GPU.

---

## Tier 1 — Claude deep-math-domain items (added 2026-06-30)

**Origin:** claude enumerated 14 genuinely novel math sub-domains not currently searched by the fab
(see `research/notes/component_fab_unexplored_geometry_2026-06-30.md`). The four below are the
highest-leverage Tier-1 axes — each opens a new sub-domain AND targets a known capability wall.
The rest (Tier-2/3/4 from the note) are sequenced after Tier 1 lands.

**Mission alignment:** every item below is anti-softmax-twin by structure (different score
geometry, different aggregation, different topology) — not a softmax/attention/reciprocal/semiring
variant. None cuts the novel branch. Each ships WITH a softmax-twin detector + a collapse detector
in the ledger metadata so the `padic_depth_route` (router-collapse) and `depth_weighted_proj`
(router-collapse) pathologies do not repeat.

**Conflict protocol (multi-agent):** re-read shared files fresh before claiming; coordinate on
overlapping scope per "Coordination matrix" below; stage own files only; `--no-verify`; first
timestamped `.current_work.md` entry wins per CLAUDE.md.

### NM-T1-1 — Tsallis / Rényi q-softmax family  *(kicks off Tier 1)*
**Wall target:** W1 (compositional binding) — sparse capacity-bounded score norms give
per-slot winner-takes-this-slot selectivity that softmax blur cannot. entmax-2 has lit-review wins
on multi-label/structured prediction.
**Why novel:** softmax = minimizer of KL (Gibbs variational). Tsallis q-entropy minimizer
(q ≠ 1) gives a 2-parameter family (q, β) where softmax/sparsemax/entmax/top-k are all interior
points; most of the surface has **never been instantiated for a sequence mixer**.
**Files:**
- MOD: `research/synthesis/parametric_ops.py` — extend `SCORE_NORM_FAMILIES` with
  `("tsallis_q", "renyi")` and add the q-exponential branch
  `[1 + (1-q)(s - tau)]_+^(1/(1-q))` to the `StageSpec` score-norm forward path.
  **This is the correct subsystem — score-norm is applied here, NOT in
  `code_generator._dispatch_math_knob`, which routes on `op_math_family` and never
  sees score-norm.**
- MOD: proposer axis `op_physics_score_norm_family` (currently `{softmax, sharpen}`)
  — add `tsallis_q`/`renyi` as sampleable values.
- MOD: `component_fab/proposer/dynamic.py::_REPAIR_RULES` — add `repair_score_norm_spectrum`
  triggered by `nb_max_accuracy < 0.62 AND no explicit score-norm tried`; emits a 5-spec sweep
  over `op_physics_score_norm_family` with `q ∈ {0.5, 0.9, 1.5, 2.0, 3.0}`.
- NEW: `research/tests/test_tsallis_score_norm.py` — softmax≡q=1 at init, q→∞ → top-k
  limit (sparsemax at q=2), gradient finite at all q values.
**Done-when:** registered + dispatches + fwd/bwd finite + score-norm sweep produces ≥3 distinct
(non-twin) ledger entries. softmax-twin detector attached.
**File claim:** UNCLAIMED — claude candidate, but check `.current_work.md` before editing
because minimax-3 may have parallel coverage.

### NM-T1-2 — Sheaf diffusion mixer  *(the principled W1 fix)*
**Wall target:** W1 (compositional/multislot binding) — likely also W2 (long-range). This is the
mathematical fix for the "right content, wrong slot" failure mode documented in
`research/notes/novel_mechanism_architecture_redesign_2026-06-14.md` Run-1 diagnosis.
**Why novel:** a sheaf assigns to every open set of the position poset a vector + restriction map;
overlap windows must agree on shared tokens. Sheaf Laplacian diffusion forces cross-window
consistency. *No softmax analog — agreement is equality, not score-weighting.*
**Files:**
- NEW: `component_fab/generator/primitive_templates/_sheaf_mixer.py` (~300 LOC).
- MOD: `component_fab/generator/code_generator.py` — `op_algebraic_space = "sheaf_diffusion"`
  dispatch.
- MOD: `component_fab/inventor/mechanism_catalog.py` — `InventionBlueprint(mechanism_id=
  "sheaf_consistent_slot_mixer", ...)` mirroring `semiring_surprise_memory_rope` contract.
- NEW: `component_fab/tests/test_sheaf_mixer.py` — overlaps-must-agree + non-QK invariant.
**Done-when:** registered in all 3 fab files + mechanism catalog; windowed agreement penalty
enforced via regularisation; fwd/bwd finite; overlap-agreement loss measurable on toy inputs.
**Anti-collapse guard:** regularise the sheaf restriction-consistency (gluing /
cocycle) identity `ρ_{U∩V} ∘ ρ_V = ρ_{U∩V} ∘ ρ_U` to prevent restriction maps
collapsing to identity.
**File claim:** UNCLAIMED — claude candidate, no parallel coverage expected.

### NM-T1-3 — Fractional-derivative memory
**Wall target:** W2 (long-gap / AR held-pair / AR depth S4+) — Mamba/SSM exponential decay has
finite effective horizon; the fractional *integral* gives power-law (`lag^{α-1}`) decay, bounded
gradient, infinite effective memory.
**Why novel:** the Riemann-Liouville fractional integral `I^α` is an accumulating (low-pass)
power-law-memory kernel — a new memory regime that none of the existing
`op_dynamical_memory_length_class` descriptor buckets (`{O(1), O(L), O(L^2), O(log L)}`) capture.
(Use the *integral*, not the fractional derivative — the derivative form is a high-pass differencer
and does NOT accumulate memory.)
**Files:**
- NEW: `component_fab/generator/primitive_templates/_fractional_memory.py` (~250 LOC).
  Grünwald-Letnikov *integral* discretisation `y_t = sum_{k=0}^{K-1} w_k x_{t-k}` with **positive**
  weights `w_k = Γ(k+α) / (Γ(k+1) Γ(α))` (∝ `k^{α-1}`), learnable `α ∈ (0,1)`, K=256 truncation.
- MOD: `component_fab/inventor/mechanism_catalog.py` — `InventionBlueprint(
  mechanism_id="fractional_integral_memory", ...)` mirroring `padic_surprise_memory` contract.
  Dispatch is via this blueprint, NOT via the `op_dynamical_memory_length_class` descriptor axis
  (that axis classifies complexity; it is not a mechanism-routing key).
- MOD (optional, telemetry only): `research/synthesis/parametric_atoms.py` — add descriptor value
  `"O(L^alpha)"` for classification.
- NEW: `component_fab/tests/test_fractional_memory.py`.
**Done-when:** builds, fwd/bwd finite, `α → 1` ≡ running-sum (flat, longest memory), `α → 0` ≡
near-identity (current token only), memory decays as `lag^{α-1}` as predicted.
**File claim:** UNCLAIMED — claude candidate. ⚠️ coordinate-with if minimax-3 covers
ultrametric/Bruhat–Tits tree (NM-5 in Tier B) — fractional + ultrametric may share testing infra.

### NM-T1-4 — MERA / multi-scale entanglement-renormalization block template
**Wall target:** W1 (compositional — MERA disentanglers remove cross-scale correlations) +
W2 (long-range — exponential coarse-graining in depth).
**Why novel:** `hierarchical_residual_compressor` is level-wise gated summaries, *not* the
strict tensor-network renorm of the MERA ansatz (Vidal 2007; cf. Evenbly & White's later TNR
scheme, 2015). MERA = principled hierarchical RG, the mathematical structure underlying Wilson's
RG and multi-resolution analysis.
**Files:**
- MOD: `component_fab/generator/block_templates.py` — add `MeraBlock` class. 3 levels per block
  (L → L/2 → L/4 → top site), disentanglers `U_l ∈ R^{d×d}`, isometries `W_l ∈ R^{d×d/2}`
  (causal: past only).
- MOD: `component_fab/generator/code_generator.py` — register `mera_block` in
  `_BLOCK_TEMPLATE_BUILDERS`.
- MOD: `component_fab/inventor/mechanism_catalog.py` — `InventionBlueprint(mechanism_id=
  "mera_block", ...)`.
- NEW: `component_fab/tests/test_mera_block.py` — coarse-grain ratio verification +
  entanglement-entropy-per-scale probe.
**Done-when:** MERA block template dispatches, builds, fwd/bwd finite; disentangler action
magnitude is non-degenerate (anti-collapse guard); coarse-graining monotonic across levels.
**File claim:** UNCLAIMED — claude candidate. ⚠️ check if `block_templates.py` is being
edited by codex (post-WS-A may touch this).

### Coordination matrix (Tier-1 file claims vs other agents)

| File | Tier-1 claim | Possible conflict | Protocol |
|---|---|---|---|
| `component_fab/generator/code_generator.py` | NM-T1-2 (`op_algebraic_space=sheaf_diffusion`), NM-T1-4 (register `mera_block`) dispatch | minimax-3 may add ops here too | re-read fresh, additive append only, no-op on other's keys |
| `component_fab/generator/primitive_templates/_*` | NM-T1-2, NM-T1-3 NEW files | none expected | full claim |
| `component_fab/inventor/mechanism_catalog.py` | NM-T1-2, NM-T1-3, NM-T1-4 blueprints | minimax-3 may extend this | re-read fresh, additive append to `DEFAULT_INVENTION_BLUEPRINTS` |
| `research/synthesis/parametric_ops.py` | NM-T1-1 extends `SCORE_NORM_FAMILIES` + StageSpec score-norm path (correct subsystem for score-norm) | codex WS-G (lambda/functional) also touches this | **coordinate-before-edit** — stage NM-T1-1 + WS-G as one PR |
| `research/synthesis/parametric_atoms.py` | NM-T1-3 optional descriptor value (telemetry only) | likely none | full claim |
| `component_fab/proposer/dynamic.py` | NM-T1-1 repair rule | codex WS-A..F in flight; check current dynamic.py state | re-read fresh, additive `_REPAIR_RULES` entry |
| `component_fab/generator/block_templates.py` | NM-T1-4 `MeraBlock` | possible codex edits | re-read fresh, additive |
| `component_fab/math_knobs.py` | NOT in Tier-1 (NM-2 from earlier plan covers) | codex WS-G edits here | **coordinate-before-edit** |
| `component_fab/improver/math_knob_catalog.py` | NOT in Tier-1 (NM-2) | codex WS-G edits here | **coordinate-before-edit** |

### Tier-1 sequencing (one PR per work item)
1. **NM-T1-1** (Tsallis q-softmax) — single PR, ships first because it composes with every
   other Tier-1 mechanism.
2. **NM-T1-3** (fractional-derivative memory) — single PR, opens new stateful-primitive family.
3. **NM-T1-2** (sheaf diffusion) — single PR, the W1 fix.
4. **NM-T1-4** (MERA block) — single PR, opens renorm/tensor-network sub-domain.

End state (after Tier 1 lands): the fab searches **4 NEW algebraic spaces, 2 NEW stage kinds,
1 NEW stateful memory primitive, 1 NEW block template, 3 NEW invention blueprints** — anti-softmax-
twin by construction, each with twin/collapse detectors in metadata.

---

## Tier D — Compaction & weight-sharing (decrease model size → less VRAM; hold accuracy + speed)

**Origin:** user directive 2026-07-01 — GOAL: add as many novel mechanisms as possible
that keep accuracy + speed but **reduce model size** (→ less VRAM in training AND
inference). Deep version (all 22 lanes, math, gates, build-on, baselines-to-beat):
`research/notes/component_fab_compaction_lanes_2026-07-01.md`.

**Why mission-amplifying, not a throttle:** VRAM decides whether a novel non-QKV
mechanism can train/eval at competitive scale. At cl100k the **embedding is ~75% of
params** and per-layer O(d²)×n dominates the rest; a 3×-smaller iso-capable mechanism
fits 3× the depth/width on the same GPU — compaction is how the novel branch reaches
the scale where it beats softmax. MoE / pruning / PTQ / LoRA / ALBERT-tying are
**baselines to beat**, never the path.

Prefix **NM-C** (Novel Math — Compaction).

| ID | Lever | Mechanism (size / FLOP win) |
|----|-------|------------------------------|
| NM-C1 | embedding | product-quantized / factored embedding — O(√V·d) vs O(V·d), ~10× cut at cl100k |
| NM-C2 | embedding | compositional feature-hash token rep — zero V×d table |
| NM-C3 | per-layer W | Monarch-parameterized mixer — O(d√d) vs O(d²), permutation learned |
| NM-C4 | per-layer W | Kronecker-factored mixing — W=A⊗B, split learned |
| NM-C5 | per-layer W | butterfly / orthogonal-flow weight — O(d log d), exactly orthogonal |
| NM-C6 | per-layer W | TT / tensor-ring weighted mixer — O(d·r·log d) |
| NM-C7 | depth | recurrent-depth weight-shared refinement — 1 block ×T, NON-softmax loop gate |
| NM-C8 | depth | shared weight dictionary across layers — W_l=Σ c_{l,b}·M_b (B≪n) |
| NM-C9 | depth | hypernetwork-generated per-layer weights — hyper(role,layer)→W_l |
| NM-C10 | depth | single-layer + persistent external memory — retrieval pathway (p-adic gap fix) |
| NM-C11 | seq/state | native block-sparse mixer — K≪d² nonzero blocks, non-softmax gate |
| NM-C12 | seq/state | token-merging-as-mechanism — L→K, sheaf-consistent merge |
| NM-C13 | seq/state | low-rank-native state memory — KV-analog O(r·d) not O(L·d) |
| NM-C14 | seq/state | tropical argmax-only retrieval — top-K reads O(K), max-plus algebra |
| NM-C15 | precision | ternary-native mixer — {−1,0,+1}, ÷16 VRAM, sign-semiring (not post-hoc) |
| NM-C16 | precision | p-adic low-precision mixer — lossless low bit-width by valuation truncation |
| NM-C17 | precision | dynamic precision router — capability-gated 2-bit/wider paths |
| NM-C18 | cond. compute | Mixture-of-Depths w/ NON-softmax router (collapse-proof) |
| NM-C19 | cond. compute | capability-gated early exit (mixing/learning-speed axes, not loss) |
| NM-C20 | cond. compute | Mixture-of-Subspaces — O(r·d) per layer, learned grouping |
| NM-C21 | META | active-param / footprint fab objective — rank by non-embed params per capability |
| NM-C22 | research | knowledge-baked architectural equivalence (symmetry → 1/N size) |

**Sequencing:** (1) **NM-C21** footprint objective — cheapest, multiplies every lane's
yield; (2) **NM-C3/4/5** factorizations — biggest per-layer win, proven expressivity,
GLM synthesis zone; (3) **NM-C1** factored embedding — biggest absolute cut (~45M at
cl100k), coordinate on embedding path; (4) **NM-C7/NM-C10** — collapse n_layers +
retrieval-pathway fix the p-adic scale result demands; (5) **NM-C15/NM-C16** — ÷16 VRAM
native-bit math; (6) long tail.

**Anti-collapse gates (the win silently erasing itself):**
- weight-shared / tying (C7/C8/C9) → diversity/orthogonality + coefficient entropy + condition number;
- low-rank (C6/C13/C20) → learned rank + capability floor + effective-rank probe;
- recurrent-depth (C7/C18) → **NON-softmax** gate + entropy + load-balance + NM-11 twin detector
  (the documented `recursive_depth_router` collapse must not repeat);
- every lane ships the softmax-twin detector +, where structural, a randomized-query binding control.

**Coordination:** GLM zone = NM-C3/4/5/6/14 + NM-C21 (check `ranking.py` vs codex);
codex zone = C7/C10/C18 blueprints (`mechanism_catalog.py`); embedding path C1/C2 +
ranking C21 may be shared → re-read fresh, additive only, stage own files, `--no-verify`,
`Co-Authored-By: GLM-5.2`. NM-1 math-lattice ops (codex, done) are reusable forward kernels.

**Done-when (program-level):** the fab can, in one cycle, (a) factor every per-layer
weight (Monarch/Kronecker/butterfly/TT), (b) tie/share weights across layers without
expressivity collapse, (c) shrink the embedding ≥5×, and (d) RANK candidates by
non-embedding params per unit capability — every compact op passing fwd/bwd +
non-collapse + twin-detector + capability nano-probe before GPU.

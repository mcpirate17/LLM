# Latent Geometry Research & AI Scientist Capabilities

## 1. The "Latent Geometry" Specs

### Question:
Does your current "AI Scientist" framework have the ability to swap out standard dot-product attention for Hyperbolic Attention?

### Current Status:
**Yes, partially.** The framework already contains the foundational primitives and specialized modules required for Non-Euclidean and Geometric Algebra experiments. 

#### Key Findings:
- **Hyperbolic Primitives:** Located in `research/mathspaces/hyperbolic.py`. Includes `mobius_add`, `exp_map`, `log_map`, and `hyperbolic_distance`.
- **Clifford Algebra (Geometric Algebra):** Located in `research/mathspaces/clifford.py`. Includes `geometric_product`, `rotor_transform`, and `clifford_linear`.
- **Manifold-Aware Operations:** 
    - `hyperbolic_norm`: Performs `log-map → LayerNorm → exp-map` to normalize without distorting hyperbolic geometry.
    - `CliffordAttention`: Uses the **geometric product** ($ab = a\cdot b + a\wedge b$) instead of a simple dot product for attention scores, allowing the model to attend to both scalar and bivector relationships.
    - `PoincareDistanceRouting`: Routes information based on hyperbolic distance in the Poincare ball, naturally capturing hierarchical relationships (syntax, part-whole) by leveraging the exponential growth of space near the manifold boundary.
- **Native Acceleration:** `aria-core` provides optimized C++ kernels for these operations (e.g., `aria_clifford_attention_f32`, `aria_hyperbolic_distance_f32`), ensuring that non-Euclidean experiments are computationally feasible.

### Gap Analysis:
While the **capability** to swap standard attention for Hyperbolic/Clifford attention exists via the `research/synthesis/` grammar, a fully automated "Manifold Discovery" module that dynamically decides *when* to shift from Euclidean to Hyperbolic space during a single training run is still in the "foundational" stage. The AI Scientist can **hypothesize** and **test** specific hyperbolic architectures (as seen in `research/scientist/runner.py` with `math_space_weight`), but it does not yet autonomously perform online manifold switching.

---

## The Plan: Integrate a "Manifold Discovery" Module

To achieve the requested $10	imes$ improvement in learning efficiency, the next phase of development will focus on:

1.  **Online Manifold Probing:** Implement a diagnostic task that measures the "hierarchical saturation" of a layer's embeddings. If the embedding distribution mimics a tree structure (high Delta-hyperbolicity), the system triggers a "Hyperbolic Promotion."
2.  **Compressed Hierarchies:** Use Hyperbolic Attention to explicitly encode language syntax. Since hyperbolic space can embed a tree with nearly zero distortion in low dimensions, this allows the AI Scientist to compress complex structural relationships that would require high-dimensional Euclidean space.
3.  **Manifold-Aware Optimizer:** Implement Riemannian SGD/Adam to ensure that updates to hyperbolic parameters respect the manifold's curvature, preventing "ball-exit" errors and maintaining numerical stability during aggressive scaling.

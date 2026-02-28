# High-Order Tropical Routing Research

## 2. High-Order "Tropical" Routing

### Question:
How does your AI scientist handle Discrete Optimization? Can it swap standard routing for Tropical Mathspaces?

### Current Status:
**Supported & Enhanced.** The framework now includes both the theoretical foundations and an implementation of Tropical Routing for Mixture-of-Experts (MoE) and Mixture-of-Depths (MoD).

#### Key Findings:
- **Tropical Algebra Foundations:** Located in `research/mathspaces/tropical.py`. Implements the $(\mathbb{R} \cup \{+\infty\}, \min, +)$ semiring where standard multiplication becomes addition and addition becomes minimum.
- **Tropical Routing Module:** Located in `research/mathspaces/tropical_routing.py`. 
    - **Reduced Routing Tax:** By using $(\max, +)$ logic, routing becomes a "shortest-path" problem in the embedding space. This replaces computationally expensive exponentiations and large matrix multiplications with simple additions and minimums.
    - **Micro-Expert Scaling:** The framework is designed to support **1,000+ micro-experts**. Because the "tax" of computing distances to expert centroids is so low in the tropical semiring, we can scale the number of experts by 10-100x compared to standard Softmax-based MoE.
- **Discrete Optimization Handling:** 
    - Currently, discrete graph topology search is handled via continuous relaxation (**Gumbel-Softmax**) in `research/search/differentiable_dag.py`.
    - Behavioral "hard" decisions are modeled using Tropical Softmax (softmin with low temperature), which converges to an exact $(\max, +)$ argmin as $T 	o 0$.

### Gap Analysis:
The primary gap is the transition from **Differentiable Search** to **Pure Discrete Optimization**. While the tropical mathspaces provide the *operators* for discrete-like logic, the AI Scientist still relies on gradient-based methods to learn the expert centroids. 

---

## The Plan: Reducing the "Routing Tax"

To move toward nearly computationally free routing:

1.  **Bitwise Tropical Routing:** Implement bit-shifted $(\max, +)$ kernels in `aria-core` that operate on quantized embeddings. This would allow MoE routing to happen in the integer domain, completely bypassing the floating-point unit.
2.  **Topological Expert Growth:** Allow the AI Scientist to dynamically "spawn" micro-experts during training when it detects that a specific tropical centroid is overloaded (too many tokens mapped to one "shortest path").
3.  **Combinatorial Synthesis:** Expand the `research/synthesis/` grammar to explicitly include "Tropical Blocks" as high-level motifs, ensuring the AI Scientist prioritizes these low-tax structures in the efficiency frontier search.

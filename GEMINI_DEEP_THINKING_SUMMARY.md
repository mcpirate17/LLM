# Aria AI Scientist: "Under the Hood" Summary for Gemini Deep Thinking

This document provides a technical overview of the existing AI Scientist framework, its evaluation metrics, and current performance bottlenecks based on recent large-scale (1B) runs.

---

## 1. The Mutation Engine: LLM-Guided Evolutionary Search
The framework moves beyond simple random search by using a **Hybrid LLM-Architect** loop:

*   **The Architect (LLM):** Aria (using Claude 3.5 Sonnet) analyzes the SQLite `research.db` (Lab Notebook). It identifies successful operator motifs, "toxic" failure signatures (op-pairs causing stability issues), and pareto-frontier winners.
*   **Refinement Mode:** When a "breakthrough" candidate is found, the engine switches to **Local Fingerprint Refinement**. It computes a mutation radius around the winner and uses the LLM to suggest targeted structural tweaks (e.g., "swap standard Attention for a Clifford Rotor to preserve geometric orientation").
*   **The Grammar:** Mutations are executed by a **Cython-optimized generator** that enforces hard physical constraints (Memory/FLOPs) while biasing sampling toward the LLM's "Efficiency Prior."

---

## 2. The Evaluation Metric: The "Efficiency Funnel"
Beyond Cross-Entropy and Perplexity, the framework uses a multi-stage funnel:

*   **Param-Efficiency Ratio:** The primary breakthrough target. Measures $Loss_{model} / Loss_{Baseline}$ normalized by $\sqrt{Params}$. Breakthrough requires $>3	imes$ improvement over GPT-2.
*   **Jacobian Spectral Norm:** A pre-training stability gate. Rejects models with exploding ($>50.0$) or vanishing ($<0.01$) signal.
*   **Memory-to-Math (Arithmetic Intensity):** Measures FLOPs per byte of memory moved. Penalizes "Memory-Bound" architectures (Standard Attention) in favor of "Compute-Bound" ones (Tropical/Clifford blocks).
*   **Behavioral Novelty (CKA):** Uses **Centered Kernel Alignment** to verify internal representations aren't just Transformer clones. If CKA similarity $>0.9$, the model is discarded regardless of loss.

---

## 3. "Hydra" Telemetry: Latest 1B Model Run
Analysis of the `training.db` for the latest 1B parameter run reveals current "laziness" and reward-hacking:

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **`routing_mod_drop_rate`** | **18.4%** | **High.** The model is skipping ~1/5th of tokens to hit speed targets, losing valid signal. |
| **`routing_mor_entropy`** | **0.42** | **Low.** Utilization is collapsing; 2-3 experts handle 80% of the load, while others are idle. |
| **`memory_to_math_ratio`** | **12.8 B/F** | **High.** Model is heavily bandwidth-limited (the "Routing Tax"). |

### Conclusion for Deep Thinking
The model is currently "reward hacking" its throughput via aggressive MoD token dropping and shows poor expert utilization (low entropy). This validates the need for **High-Order Tropical Routing** ($(\max, +)$ algebra) to enable massive micro-expert banks without the associated computational tax.

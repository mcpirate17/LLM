# Kolmogorov Complexity & MDL Research

## 3. Kolmogorov Complexity & MDL (Minimum Description Length)

### Question:
What is the "Experiment Engine's" reward function? Does it optimize for Normalized Compression Distance (NCD)?

### Current Status:
**Evolving.** The current reward function is a multi-objective proxy rather than a pure NCD/MDL objective. However, the framework contains the necessary metrics to transition toward a compression-centric reward.

#### Current Reward Function:
The AI Scientist evaluates candidates through a "Funnel" where the final selection/promotion is based on a weighted combination of:
1.  **Loss Ratio ($L_{ratio}$):** Performance relative to a standard baseline (GPT-2) on the micro-corpus.
2.  **Structural Novelty ($N_{struct}$):** Op-diversity and category spread computed from the graph IR.
3.  **Behavioral Novelty ($N_{behav}$):** CKA-based similarity to reference families.
4.  **Robustness ($R$):** Stability across multiple synthesized training programs.
5.  **Quality per Byte ($Q_{byte}$):** Found in `research/eval/quantization.py`. Computed as $Retention 	imes CompressionRatio$. This is the closest current metric to an MDL objective.

#### Key Findings:
- **Compression Awareness:** The system explicitly tracks `compression_ratio` and `bytes_per_param_effective`. 
- **Information-Theoretic Gating:** `research/eval/metrics.py` calculates the entropy of the operator distribution to reward "balanced" complexity, a core component of MDL.
- **Missing Link:** There is currently no explicit implementation of **Normalized Compression Distance (NCD)** ($NCD(x, y) = \frac{C(xy) - \min(C(x), C(y))}{\max(C(x), C(y))}$) used as a training signal.

### Gap Analysis:
The AI Scientist is currently optimized for **Learning Efficiency** (attaining low loss with few parameters). It is not yet explicitly optimized for **Programmatic Minimality** (Kolmogorov Complexity). The "Global Program Synthesis" objective is present in the way architectures are generated (graph-as-code), but the *fitness* of that code is still measured by its statistical correlation (Cross-Entropy) rather than its ability to act as a minimal generator for the dataset.

---

## The Plan: Shifting to Global Program Synthesis

To treat model weights as a compressed program that must decompress the dataset:

1.  **NCD Reward Kernel:** Integrate a native NCD calculation into the evaluation funnel. Instead of just calculating loss, the AI Scientist should compare the compressed size of the model weights + the residuals of its predictions against the raw dataset size.
2.  **MDL-Based Pruning:** In Phase 4 (Adaptive Synthesis), use the **Description Length** of the graph IR as a hard constraint. Prune branches that add structural complexity without a corresponding reduction in data entropy.
3.  **Algorithmic Information Gain:** Shift the "Investigation" phase to measure how much *unseen* data can be predicted by the synthesized "program" after only a few examples. This forces the model to learn the underlying *rules* (high Kolmogorov gain) rather than memorizing correlations.
4.  **Weight-as-Code:** Implement a "Code-Gen" mode where the AI Scientist attempts to synthesize symbolic expressions (via the `functional` mathspace) that approximate the learned weight matrices, effectively converting the neural network into a set of executable programs.

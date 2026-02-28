# Aria Pipeline Optimization & Rescoring Plan

> **Multi-agent rules:** Claim tasks in Phase 5 by editing the `(Claimed: <agent>)` tag AND updating `.current_work.md` BEFORE starting code. Do not overwrite another agent's claim. Mark `[x]` only after code compiles and tests pass. See `CLAUDE.md` § "Rules of Engagement" for full policy.

## Objective
The current Neural Architecture Search (NAS) pipeline is experiencing "reward hacking." Because Stage 1 (micro-training) uses random data, the only way for models to minimize loss is to break the causal mask and "look ahead" at future tokens. This plan outlines the steps to fix the pipeline's incentives, enforce strict causality, and retroactively salvage and rescore the architectures we have already discovered.

---

## Phase 1: Pipeline Optimization (Preventing Future Cheating)

### 1.1 Split Data Strategy (The "Micro-Corpus" Fix)
We will maintain the speed of the pipeline by keeping random data for structural checks, but introduce real data for learning checks.
*   **Stage 0 & 0.5 (Compilation & Stability):** Continue using random tensors. This remains the fastest way to catch OOMs, NaNs, and compilation failures.
*   **Stage 1 (Micro-Training):** Switch `data_mode` to `"corpus"` using a highly compressed "micro-corpus" (e.g., a small snippet of code or Shakespeare). This forces the model to learn actual sequential patterns. Causal models will now be able to demonstrate genuine learning curves.

### 1.2 Strict Causality Gate (Stage 0.5)
Before a model is allowed to train, we will mathematically prove it cannot look ahead.
*   **The Test:** 
    1. Pass sequence $A = [t_1, t_2, t_3, t_4]$ through the model and record the logits for $t_3$.
    2. Pass sequence $B = [t_1, t_2, t_3, t_X]$ (where the future token is changed).
    3. Compare the logits for $t_3$. If they differ by more than a floating-point epsilon, the model is leaking future information.
*   **Action:** Any model failing this gate is immediately rejected with a `causality_violation` error.

### 1.3 Patch Known Causal Leaks
*   [x] Audit and restrict operations that inherently break causality in an autoregressive context (e.g., `rfft_seq`).
*   [x] Ensure operations like `conv1d_seq` strictly enforce left-padding (causal padding) at the runtime level.

---

## Phase 2: Smart Back-Population & Rescoring (Salvaging Existing Data)

We currently have over 1,000 models that passed Stage 1. Many of these are cheaters, but some might be genuinely good causal architectures that just didn't get a fair score.

### 2.1 Identify Cheaters vs. Valid Models
*   Write a standalone script to load all 1,051 Stage 1 survivors from `lab_notebook.db`.
*   Run the new **Strict Causality Gate** on every model.
*   Flag and quarantine the models that fail (the "cheaters"). We won't delete them (their structural data might still be useful for non-causal tasks like BERT-style masking), but they will be removed from the autoregressive leaderboard.

### 2.2 Rescore Valid Causal Models
*   For the models that *pass* the causality gate, their current `loss_ratio` (achieved on random data) is invalid.
*   Re-run Stage 1 micro-training for these valid models using the new **Micro-Corpus**.
*   Calculate their true `loss_ratio` and `baseline_loss_ratio` against a standard causal baseline (like a small GPT-2) trained on the exact same micro-corpus.

### 2.3 Database Update
*   Update the `program_results` table:
    *   Set `stage1_passed = 0` and `error_type = 'causality_violation'` for the cheaters.
    *   Update the `loss_ratio`, `final_loss`, and `baseline_loss_ratio` for the valid models based on their new micro-corpus scores.
*   This will instantly correct Aria's historical context, allowing her LLM planner to learn from true causal performance rather than reward-hacked anomalies.

---

## Phase 3: Execution Steps
1. [x] Implement the Causality Gate in `sandbox.py` (Stage 0.5).
2. [x] Set up the Micro-Corpus and update the Stage 1 data loader in `runner.py`.
3. [x] Write and execute the back-population/rescoring script. (Running in background)
4. [x] Resume the continuous run.

---

## Phase 4: Scoring & UX/UX Hardening (Not Planned Yet)

### 4.1 Dual-Metric Scoring (Discovery vs. Validation)
We should preserve fast discovery signals while ensuring leaderboard truth uses real data.
*   **Discovery Loss (Random Tokens):** Keep a cheap random-tokens loss for early rejection/triage.
*   **Validation Loss (Micro-Corpus):** Store the causal, real-data loss for ranking and promotion.
*   **Decision Rule:** Stage 1 pass should be governed by validation loss; discovery loss is for debugging/triage only.

### 4.2 Data & Schema Updates
*   Add new fields in `program_results` (or a sibling table) for:
    *   `discovery_loss`, `discovery_loss_ratio`
    *   `validation_loss`, `validation_loss_ratio`
    *   `validation_baseline_loss_ratio`
*   Backfill legacy rows with `NULL` discovery/validation fields where unavailable.

### 4.3 Leaderboard / UX Readiness
*   Update any ranking/filters to use validation metrics by default.
*   Expose both metrics in UI/UX (clear labels) to avoid confusion.
*   Add tooltips or legend explaining discovery vs. validation meaning.

### 4.4 Reporting & Monitoring
*   Add a periodic report that shows correlation between discovery and validation loss to detect drift.
*   Track the percentage of models failing the causality gate per day as a health metric.

### 4.5 Learning Quality & Anti-Memorization
We should detect memorization and quantify generalization even in micro-training.
*   Add a deterministic train/validation split for the micro-corpus (e.g., 80/20 or alternating windows).
*   Compute **train loss** and **validation loss** (heldout) on each model.
*   Store a **generalization gap** metric (validation - train) and flag memorization risk when the gap is large or widening.

### 4.6 Baselines & Scoring Consistency
*   Compute **baseline losses** for both discovery and validation channels using the same data mode/split.
*   Ensure cached baselines are keyed by data mode, split signature, and corpus version.
*   Update best-loss summaries and promotion logic to prefer validation metrics.

### 4.7 UI/UX & Analytics Coverage
*   Dashboard: add charts for discovery vs validation loss, and generalization gap trends.
*   Fingerprint/Program detail pages: add a dual-loss panel and train-vs-val curve if available.
*   Aria Designer import + evaluation views: label loss metrics as discovery vs validation and sort by validation.
*   Update report narratives and LLM context builders to reference validation loss by default.

### 4.8 Core Causality Safeguards
*   Add automated tests for causal ops (e.g., `conv1d_seq`, `rfft_seq`) to verify no look-ahead.
*   Add a small regression suite that runs the strict causality gate on a curated set of ops/graphs.

---

1. [x] Implement corpus train/validation split in Stage 1. (Claimed: Gemini CLI)
2. [x] Update database schema to support dual-metric fields. (Claimed: Gemini CLI)
3. [x] Update scoring and leaderboard queries to prioritize validation loss. (Claimed: Gemini CLI)
4. [x] Add backfill/migration logic for legacy rows. (Claimed: Gemini CLI)
5. [x] Update UI/UX to surface dual metrics with tooltips. (Claimed: Gemini CLI)
6. [x] Implement periodic gate performance reporting in runner. (Claimed: Gemini CLI)
7. [x] Add causal-op regression tests and gate verification. (Claimed: Gemini CLI)
8. [x] Integrate passkey retrieval test for long-context evaluation. (Claimed: Gemini CLI)
9. [x] Implement dashboard pinning support and top-sorting. (Claimed: Gemini CLI)
10. [x] Add side-by-side architecture comparison view. (Claimed: Gemini CLI)
11. [x] Provide "Open in Designer" bidirectional design convergence. (Claimed: Gemini CLI)
12. [x] Unified schema validation for routing/compression. (Claimed: Gemini CLI)
13. [x] Move routing/compression kernels to native C++ (aria-core). (Claimed: Gemini CLI)

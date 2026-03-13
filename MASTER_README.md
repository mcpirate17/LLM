# Master README: Research + Aria Designer (Technical Summary)

## Metadata
- Created by: Codex (GPT-5) in collaboration with `tim`
- Created on: March 9, 2026 (America/New_York)
- Source scope: `research/README.md`, `aria_designer/README.md`, `aria_designer/api/README.md`, `aria_designer/ui/README.md`, plus fingerprint/novelty implementation files
- Purpose: non-marketing technical summary for downstream model review (GPT-5.4 reasoning)

## What These Projects Are Trying To Do
The combined system (`research` + `aria_designer`) is an architecture discovery runtime for language-model-like computation graphs.

- `research` is the scientist runtime: it generates candidate graph architectures, compiles/tests/trains them, computes novelty/fingerprint signals, and records ranked results.
- `aria_designer` is the visual front-end + API bridge for creating/editing workflows that map to the same graph runtime used by `research`.

The core goal is not just finding models that train, but finding **architectures with distinct behavioral identity** rather than re-discovering near-copies of known families.

## Core Technical Hypothesis
A candidate architecture should be treated as “novel” only when:
1. It is structurally distinct in graph space, and
2. Its behavior is not too similar to known reference families (especially Transformer/GPT-like and SSM/Mamba-like), and
3. This conclusion is stable enough to trust for promotion decisions.

## Fingerprint System (Current Implementation)

### 1. Structural fingerprint (identity key)
`ComputationGraph.fingerprint()` creates a deterministic 16-char SHA256 prefix from a canonicalized topological description:
- op names
- normalized input connectivity (ranked by topo order)
- sorted op configs
- `model_dim`
- routing/compression policy metadata when present

This is used for deduplication and lineage tracking, not behavior.

### 2. Behavioral fingerprint (measurement vector)
`compute_fingerprint()` builds a `BehavioralFingerprint` with four analysis blocks:

1. Interaction analysis
- Perturb selected token positions, measure output change matrix.
- Locality: inverse distance-weighted influence concentration.
- Sparsity: `1 - entropy(influence_probs)/log(N)`.
- Symmetry: triangular mismatch norm between approximate A->B and B->A influences.
- Hierarchy proxy from variance change under pooling.

2. Representation geometry
- SVD on sampled representations.
- Intrinsic dimension via participation ratio: `1 / sum(p_i^2)` where `p_i` are normalized singular values.
- Isotropy: `min(S)/max(S)`.
- Rank ratio: `exp(entropy(p)) / len(S)`.

3. Input sensitivity
- Gradient-based token sensitivity from internal activations.
- Spectral norm proxy: Frobenius norm of stacked sensitivity matrix.
- Uniformity: entropy of per-position sensitivity normalized by `log(seq_len)`.
- Effective rank proxy: `exp(entropy)`.

4. Reference similarity (CKA)
- Linear CKA vs reference families: `transformer`, `ssm`, `conv`.
- Novelty core: `behavioral_novelty = 1 - max(cka_transformer, cka_ssm, cka_conv)`.

### 3. CKA reference provenance
CKA references come from artifact-backed activation stores when available (`artifacts/cka_references/<version>` with manifest + tensors). Metadata is attached:
- source (`artifact` vs `heuristic_fallback`)
- artifact version
- probe protocol hash
- quality flags

This feeds `novelty_valid_for_promotion` and confidence logic.

### 4. Novelty score composition
`novelty_score()` blends structural and behavioral novelty:
- If fingerprint exists: `raw = 0.3 * structural + 0.7 * behavioral`
- If no fingerprint: `raw = 0.6 * structural`

Then applies penalties:
- Reference similarity penalty: `overall *= max(0.25, 1 - 0.75 * max_cka)`
- Duplicate architecture penalty: `overall *= 0.1` for exact fingerprint duplicates

Structural novelty itself is:
- `0.50 * op_diversity + 0.30 * category_spread + 0.20 * distribution_evenness`
- optional multiplicative bonus for math_space/frequency usage: `* (1 + 0.1 * exotic_count)`

### 5. Calibration and stability
`novelty_calibration.py` estimates novelty noise floor by repeated baseline-transformer probes:
- mean/std of novelty
- 5%/95% confidence quantiles
- novelty z-score against noise floor

This is intended to prevent over-interpreting small novelty differences.

## GPT vs Mamba Fingerprint Focus (and “5-fold” status)

### What is implemented now
The system already compares candidates against GPT-like and Mamba-like behavior through CKA channels:
- GPT-like proxy: high `cka_vs_transformer`
- Mamba-like proxy: high `cka_vs_ssm`

Decision surface today is effectively based on max similarity across 3 families (transformer/ssm/conv), then novelty inversion and penalties.

### Important gap: explicit 5-fold protocol is not currently implemented
There is no explicit k-fold or “5-fold GPT/Mamba” routine in current code paths. Current fingerprinting is single-pass per evaluation instance (with optional calibration routines run separately).

### Practical interpretation for GPT-5.4 review
The current system is best described as:
- **family-similarity scoring**, not formal fold-validated family attribution.
- strong on feature extraction and novelty blending.
- weaker on repeated-fold robustness guarantees for GPT-vs-Mamba attribution.

## Integration Between Projects
- `aria_designer` can compile/evaluate workflows through `research` runtime.
- Lifecycle and lineage endpoints connect dashboard and designer.
- Designer supports fingerprint/novelty stages in deep evaluation flows.

Notable implementation detail:
- In `aria_designer/runtime/bridge.py`, novelty is currently called as `novelty_score(graph)` without passing a behavioral fingerprint object, which can reduce scoring to structural-only in that path.

## What To Validate Next (for “right track or not”)
1. Add explicit 5-fold fingerprint protocol for GPT-vs-Mamba attribution.
2. Ensure all novelty paths pass behavioral fingerprint into `novelty_score(...)`.
3. Define promotion gates that require:
- artifact-backed CKA references,
- minimum novelty z-score above calibration noise,
- fold-consistent family attribution.
4. Track disagreement cases (high novelty but unstable fold attribution) as separate risk class.

## Suggested 5-Fold Spec (proposed, not yet implemented)
For each candidate, run 5 deterministic probe folds with different seeds/probe batches:
- Fold output: `(cka_transformer, cka_ssm, cka_conv, novelty, confidence)`
- Aggregate:
  - mean and std per CKA channel
  - family margin `mean(top1) - mean(top2)`
  - stability penalty from inter-fold variance
- Suggested acceptance rule:
  - classify GPT-like if `mean(cka_transformer) > mean(cka_ssm)` by margin `m`
  - classify Mamba-like if inverse
  - mark hybrid/uncertain if margin small or variance high
  - only promote novelty claims when fold variance is below threshold and reference provenance is artifact-backed

This would convert current single-shot family similarity into a statistically more reliable attribution pipeline.

## Scoring Change Impact (as of March 9, 2026)

Recent scoring changes materially affect leaderboard interpretation and any paper claims that depend on composite score ranking.

### What changed in code
- Reference entries are now exempt from sub-1x scaling penalty in composite scoring.
  - `research/scientist/leaderboard.py` (`if not is_reference and scaling_param_efficiency < 1.0: score *= ...`)
- Composite scoring now uses fallback:
  - `scaling_param_efficiency OR efficiency_multiple`
  - This applies to upsert/recompute paths.
- Screening/investigation tiers now write `efficiency_multiple` (geomean proxy) instead of writing `scaling_param_efficiency` (validation-only metric).

### Quantified impact from current local data (`research/lab_notebook.db`)
- Current reference composite scores:
  - GPT-2: 175.957
  - Mamba: 178.645
  - RWKV: 183.701
  - Retrieval-Augmented: 176.866
- Estimated old-score equivalent if reference sub-1x penalty were still applied:
  - RWKV: ~58.131 (down ~125.57)
  - Retrieval-Augmented: ~123.320 (down ~53.55)
  - GPT-2/Mamba unchanged (both >1x efficiency multiple)

### Interpretation impact
- Prior conclusions about reference ranking stability are not valid unless recomputed under the new scoring contract.
- Early-tier scores (screening/investigation) now incorporate efficiency via fallback more consistently.
- Cross-tier comparisons must explicitly state that early tiers use `efficiency_multiple` proxy while validation may use true `scaling_param_efficiency`.

### Required paper wording updates
- Replace any statement implying a single efficiency metric across all tiers.
- Explicitly describe:
  - `efficiency_multiple` = geomean proxy from screening/investigation operational metrics
  - `scaling_param_efficiency` = validation-only scaling experiment metric
- Recompute and republish all ranking-dependent tables/figures.

## Relevant Files (for GPT-5.4 deep dive)
- `research/eval/fingerprint.py`
- `research/eval/metrics.py`
- `research/eval/cka_references.py`
- `research/eval/novelty_calibration.py`
- `research/synthesis/graph.py`
- `research/synthesis/reference_architectures.py`
- `research/scientist/runner/execution.py`
- `research/scientist/runner/dashboard.py`
- `aria_designer/runtime/bridge.py`

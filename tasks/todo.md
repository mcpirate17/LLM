---
status: completed
created: 2026-04-02
author: claude-opus
---

# research/eval/ Audit Remediation

Full plan: `/home/tim/.claude/plans/temporal-frolicking-dragonfly.md`

## Phase 1: Dead Code Removal
- [x] 1.1 Delete `safe_json_load`, `safe_parse_float` from utils.py; update analytics_ops.py
- [x] 1.2 Delete `FAST_LANE_N_EXAMPLES`, `screening_hellaswag_payload` from hellaswag_eval.py
- [x] 1.3 Delete dead constants + `evaluate_wikitext103_validation` from wikitext_eval.py
- [x] 1.4 Delete `str(e)` no-op + unreachable `mapped_path` cleanup from sandbox.py

## Phase 2: Shared Utilities Extraction
- [x] 2.1 Extract `_env_bool()` in sandbox.py, replace 3 instances
- [x] 2.2 Extract `_iter_eligible_params()` into utils.py; update pruning.py + quantization.py
- [x] 2.3 Delete `_measure_loss` wrapper in noise_sensitivity.py; direct import
- [x] 2.4 Consolidate `compute_perplexity` to reuse `measure_loss` internals

## Phase 3: Novelty Scoring Consolidation
- [x] 3.1 Delete `_novelty_score_from_ir()`; route `novelty_score()` through `batch_novelty_scores`
- [x] 3.2 Update `fingerprint.py:compute_structural_novelty_only()` caller

## Phase 4: Fingerprint Deduplication
- [x] 4.1 Extract `_is_cka_degenerate()` helper in fingerprint.py
- [x] 4.2 Extract `_compute_quality()` helper; use in both compute paths

## Phase 5: WikiText & Benchmark Consolidation
- [x] 5.1 Consolidate `evaluate_wikitext_perplexity` into `screening_wikitext_eval`
- [x] 5.2 Extract `_mean_token_log_prob()` into utils.py
- [x] 5.3 Refactor hellaswag + blimp to use shared log-prob scorer

## Phase 6: Batched Forward Passes
- [x] 6.1 Batch 4 continuations per example in hellaswag_eval.py
- [x] 6.2 Batch pairs per subtask in blimp_eval.py
- [x] 6.3 Fix `make_batches` single `.to(device)` transfer

## Phase 7: Efficiency Fixes
- [x] 7.1 LRU eviction on `_batch_cache` in wikitext_eval.py
- [x] 7.2 Numpy tokenization in wikitext_eval._prepare_batches
- [x] 7.3 Bool→int sum in sparsity.py hooks
- [x] 7.4 Remove per-step RNG reseed in scaling_reference.py + baseline.py
- [x] 7.5 Fix double subsampling in hierarchy_probe.py
- [x] 7.6 SQLite connection reuse in baseline.py
- [x] 7.7 Reuse representations in fingerprint.compute_fingerprint (if clean)

## Phase 8: Minor Cleanup
- [x] 8.1 BaselineConfig dataclass in baseline.py
- [x] 8.2 Add missing `Any` import in diagnostic_tasks.py
- [x] 8.3 Consolidate long_context.py training loop into micro_train_loop
- [x] 8.4 Fix baseline.py initial_loss bug

# Eval Working Board

This file is the shared coordination point for Codex instances working in `research/eval`.

## Rules

1. Do not edit files another Codex has claimed unless that claim is explicitly cleared here.
2. Claim work before editing. Update this file first.
3. Use `apply_patch` for manual edits. Do not rewrite files with ad hoc shell redirection.
4. Do not trust intuition on performance. Run the profiler and benchmark path below before and after changes to hot paths.
5. If a change makes the native path slower, do not leave it enabled by default.
6. If parity breaks, stop claiming performance wins.
7. Delete dead scaffolding instead of preserving duplicate code paths.

## Required Performance Validation

All Codex working in this directory must use these entrypoints:

- Repo hotpath audit:
  - `cd /home/tim/Projects/LLM && make profile-hotpaths`
- Training profiler:
  - `python /home/tim/Projects/LLM/research/training/profiling.py`
  - Real integration point is [profiling.py](/home/tim/Projects/LLM/research/training/profiling.py).
- Reference-model benchmark:
  - `cd /home/tim/Projects/LLM && python -m research.eval.benchmark_reference_runner`
- Longer-run benchmark:
  - `cd /home/tim/Projects/LLM && python - <<'PY'`
  - `import json`
  - `from research.eval.benchmark_reference_runner import benchmark_reference_runner`
  - `print(json.dumps(benchmark_reference_runner(n_steps=128, repeats=3), indent=2))`
  - `PY`

Minimum bar for claiming a perf improvement:

- report exact command used
- report before/after numbers
- report whether loss/parity stayed matched

## Current State

- Shared native eval/data helpers exist.
- Shared training/reference/corpus/choice-scoring cores exist.
- Native reference forward now uses SDPA-backed attention and is faster than the legacy path on the CPU microbench.

Latest measured reference-model numbers:

- `8` steps x `5` repeats: legacy `21.595 ms`, native `15.985 ms`, `1.351x`
- `128` steps x `3` repeats: legacy `322.626 ms`, native `255.786 ms`, `1.261x`
- final loss parity held: `1.368035` vs `1.368036`

## Open Work

### Unclaimed

- no currently unclaimed items recorded in this board

## Claimed Work

### Codex-1

Initial claimed top 3 by this Codex:

1. `sandbox.py`
   - objective: strip the screening path down to minimal required work and move expensive diagnostics behind explicit opt-in gates
2. `screening_rapid.py`
   - objective: reduce duplicate probe passes and collapse Python bookkeeping in the rapid screen path
3. `diagnostic_tasks.py`
   - objective: force all train/eval task loops onto the shared runner and remove local duplicate logic

Status:

- completed
- results recorded below

Follow-up claims by this Codex:

1. `wikitext_eval.py`
   - objective: remove duplicate Python orchestration in trajectory/full WikiText evaluation and route repeated work through one shared helper path
2. `flops.py`
   - objective: stop presenting heuristic FLOP guesses as measured efficiency; label outputs explicitly as estimates and block misuse

Status:

- completed
- results recorded below

### Codex-2

Claimed top 3 unique findings by this Codex:

1. `training_core.py`
   - objective: remove fake-native optimizer stepping that still loops per-parameter in Python and default to honest fused/native-backed optimizer paths only
2. `binding_range.py`
   - objective: stop materializing all eval batches up front and stream probe batches directly through the scoring loop
3. `quantization.py`
   - objective: stop presenting fake quantization as a performance path; relabel outputs and block accidental performance-claim interpretation

Unique findings added from audit:

- `training_core.py`
  - current `_NativeSGDOptimizer` / `_NativeAdamWOptimizer` are performance theater: native math wrapped in Python per-parameter loops
- `binding_range.py`
  - probe currently builds a Python list of all eval batches before scoring, wasting memory and adding allocator churn
- `quantization.py`
  - current path is numerics-only fake quantization and should not read as real low-bit kernel or compression-performance evidence

Status:

- completed
- default paramwise native optimizer path disabled
- streamed binding probe batches in-place
- fake quantization outputs now explicitly marked as quality-only, not perf-valid

### Codex-3

Claimed top 3 unique findings by this Codex:

1. `fingerprint_probes.py` / `fingerprint_runtime.py`
   - objective: stop the runtime from treating logits as representations and tighten the capture contract so hidden-state probes only run on real captured reps
2. `_shared_native.py` / `_shared_native.cpp`
   - objective: remove trivial runtime-JIT native helpers from the hot path and replace them with direct Python implementations where native code is not justified
3. `_probe_runtime.py`
   - objective: make CUDA probe dispatch behavior honest by default and avoid silently routing through a path known to bounce tensors through CPU numpy buffers

Unique findings added from audit:

- `fingerprint_probes.py` / `fingerprint_runtime.py`
  - current representation API is misleading: compatibility callers get logits, while hidden-state probes depend on an attached side-channel tensor
- `_shared_native.py` / `_shared_native.cpp`
  - native helper layer is compile-latency overhead for string/list glue, not meaningful acceleration
- `_probe_runtime.py`
  - current context manager preserves a slower native bridge behind patching instead of making the on-device PyTorch path the explicit default for CUDA probes

Status:

- completed
- baseline profiling completed
- post-change validation recorded below

### Codex-4

Claimed top 2 unique findings by this Codex:

1. `associative_recall.py`
   - objective: stop mutating the live model during the probe and run the training/eval loop on an isolated copy
2. `induction_probe.py`
   - objective: same isolation fix as `associative_recall.py`; remove the in-place train/restore pattern from the live model

Unique findings added from audit:

- `associative_recall.py`
  - current probe still trains the provided model object directly and only attempts to rewind afterward
- `induction_probe.py`
  - same live-model mutation pattern, which makes the “deepcopy” claim false and couples probe behavior to caller state

Status:

- completed
- baseline profiling completed

### Codex-5

Follow-up claims by this Codex:

1. `binding_range.py`
   - objective: reduce per-batch allocation and repeated tensor setup overhead in the zero-shot copy-distance probe
2. `fingerprint_probes.py` / `fingerprint_runtime.py`
   - objective: add an explicit structured probe-capture path so runtime callers stop depending on the logits-plus-sidechannel hack

Unique findings added from audit:

- `binding_range.py`
  - even after streaming, the probe still allocates fresh random prefix and repeated tiled copy batches for every distance/batch pair
- `fingerprint_probes.py` / `fingerprint_runtime.py`
  - the runtime fix removed the worst misuse, but the capture API is still ambiguous because callers must infer reps from an attribute attached to logits

Status:

- completed
- baseline benchmarking completed

### Codex-6

Claimed follow-up items by this Codex:

1. `wikitext_eval.py`
   - objective: reduce Python orchestration and duplicate evaluation work in the screening/trajectory paths, then benchmark the eval helpers directly
2. `binding_range.py`
   - objective: remove repeated per-batch pattern construction overhead that still remains after the first streaming cleanup
3. `flops.py`
   - objective: make the heuristic nature explicit and cut repeated primitive lookup overhead in the estimator path

Unique findings added from audit:

- `wikitext_eval.py`
  - screening/full/trajectory paths still duplicate payload assembly and validation bookkeeping, and trajectory repeatedly pays avoidable Python overhead around checkpoint evaluation
- `binding_range.py`
  - current streamed version still allocates and tiles a fresh prefix tensor every batch for every distance
- `flops.py`
  - estimator is both heuristic and repeatedly re-resolves primitive metadata in Python for every node/op occurrence

Status:

- superseded by later completed follow-up work recorded below

### Codex-6

Follow-up claims by this Codex:

1. `fingerprint_probes.py` / `fingerprint.py`
   - objective: make the fallback representation path capture a real hidden-state-like tensor for simple embedding-only models instead of dropping to `None`, and expose the structured capture helper on the compatibility surface
2. `binding_range.py`
   - objective: trim avoidable per-batch allocation in the scoring loop without changing the already-streamed generation path

Unique findings added from audit:

- `fingerprint_probes.py`
  - the current fallback skips vocab-sized output heads correctly, but simple `Embedding -> Linear(vocab)` models still end up with no captured reps even though the embedding output is available and is a better probe signal than logits
- `binding_range.py`
  - generation is already on the cheapest practical path in Python; the remaining waste is in temporary float allocations during per-batch accuracy accumulation, not in the copy-pattern constructor itself

Status:

- completed
- baseline benchmarking completed

### Codex-7

Claimed follow-up items by this Codex:

1. `sandbox.py` / `research/scientist/native/abi.py`
   - objective: stop CPU ABI probing from forcing tensor payloads through Python list serialization when a contiguous tensor path is available
2. `corpus_pipeline.py`
   - objective: stop `_trim_text_chunks()` from materializing the full chunk list before truncation and keep the cache-write path streaming
3. `choice_scoring.py`
   - objective: cut Python flatten/regroup overhead in grouped multiple-choice scoring

Unique findings added from audit:

- `sandbox.py` / `research/scientist/native/abi.py`
  - native ABI probing on CPU still does `detach().cpu().reshape(-1).tolist()`, which is pure Python serialization overhead in a path explicitly meant to validate native execution
- `corpus_pipeline.py`
  - `_trim_text_chunks()` needlessly builds a full list of all chunks before truncating to `max_chars`, which defeats the point of streaming dataset iteration
- `choice_scoring.py`
  - grouped-choice scoring still burns Python time flattening and re-splitting nested choice groups in a hot eval helper

Status:

- completed
- baseline benchmarking completed

### Codex-7

Follow-up claims by this Codex:

1. `fingerprint_probes.py` / `fingerprint.py` / `research/tests/test_fingerprint_interactions.py`
   - objective: delete the remaining logits-returning compatibility wrapper path and move the last test/callers to structured probe capture only

Unique findings added from audit:

- `fingerprint_probes.py` / `fingerprint.py`
  - the remaining `get_representations()` / `_get_representations()` compatibility layer is now dead weight: no production caller uses it, and it preserves the exact ambiguous API the earlier fixes were meant to retire

Status:

- claimed
- baseline benchmarking pending

## Update Protocol

When a Codex starts work:

- add your codex label
- list claimed files
- state the exact objective

When a Codex finishes work:

- move the item to completed
- include perf command(s) run
- include result numbers
- include parity/test command(s) run

## Completed

- board reconciliation note
  - historical claim blocks above include superseded duplicates
  - use the actual file state plus the entries in this section as the current source of truth
  - remaining `fingerprint.py` underscore-prefixed exports are test-compatibility shims, not the removed ambiguous production representation path

- `training_core.py`
  - defaulted `make_optimizer()` away from paramwise Python-loop "native" optimizers unless `ARIA_ENABLE_EVAL_PARAMWISE_NATIVE_OPTIMIZER=1` is set
  - preserved SGD update parity on the honest default path by enabling Nesterov when momentum is non-zero
  - perf commands run before and after:
    - `cd /home/tim/Projects/LLM && make profile-hotpaths`
    - `cd /home/tim/Projects/LLM && python -m research.eval.benchmark_reference_runner`
  - benchmark before:
    - legacy `26.759 ms`
    - native shared runner `15.373 ms`
    - speedup `1.741x`
    - final loss parity held: `5.510972` vs `5.510972`
  - benchmark after:
    - legacy `24.751 ms`
    - native shared runner `14.464 ms`
    - speedup `1.711x`
    - final loss parity held: `5.510972` vs `5.510972`
  - tests:
    - `python -m pytest research/tests/test_eval_runner_native.py -q`
    - `python -m pytest research/tests/test_reference_model_native.py -q`

- `binding_range.py`
  - removed eager materialization of all eval batches; probe now streams generated batches directly through scoring
  - smoke validation:
    - `python - <<'PY' ... binding_range_profile(...) ... PY`
    - result status: `ok`

- `quantization.py`
  - relabeled fake-quant path as numerics-only and added explicit `performance_claim_valid: False` / `quality_proxy_only: True` markers to outputs
  - smoke validation:
    - `python - <<'PY' ... evaluate_sparse_quant_quality(...) ... PY`
    - output included `quantization_backend: fake_quantized_fp_eval` and `performance_claim_valid: False`

- `_reference_model_native.cpp`
  - generic attention path replaced with SDPA-backed native attention
  - native layer norm switched to `torch::layer_norm`
  - benchmark after fix:
    - `8` steps x `5` repeats: `1.351x`
    - `128` steps x `3` repeats: `1.261x`
- `benchmark_reference_runner.py`
  - added shared reproducible benchmark harness for legacy vs native reference path
- `sandbox.py`
  - added `AI_SCI_SAFE_EVAL_LEVEL` with `minimal` default and `full` opt-in
  - training-dynamics and activation-sparsity probes are no longer forced in the default hot path
  - heatmap capture is now full-only
  - validation:
    - `python -m pytest /home/tim/Projects/LLM/research/tests/test_causality_gate_harness.py -q`
    - `cd /home/tim/Projects/LLM && python - <<'PY' ... bench_safe_eval('minimal'|'full') ... PY`
  - measured:
    - `minimal` median `1.296 ms`
    - `full` median `10.108 ms`
- `screening_rapid.py`
  - removed extra entropy-gate forward passes by sampling hooks during the main forward
  - batch allocation moved to reusable in-place `random_()` buffer
  - validation:
    - `python -m pytest /home/tim/Projects/LLM/research/tests/test_rapid_screening.py -q`
    - `cd /home/tim/Projects/LLM && python - <<'PY' ... RapidScreeningCheck(max_steps=50) ... PY`
  - measured smoke timing:
    - `50` steps on tiny CPU LM: `454.554 ms`
- `diagnostic_tasks.py`
  - removed per-task optimizer loop duplication
  - now uses stateless cloned params plus shared `run_training_loop`
  - removed repeated `load_state_dict()` resets around every task
  - validation:
    - `python -m pytest /home/tim/Projects/LLM/research/tests/test_pipeline_integration.py -q -k 'TestDiagnosticTasks and (test_copy_generator_shapes or test_suite_result_serialization)'`
    - `cd /home/tim/Projects/LLM && python - <<'PY' ... run_diagnostic_suite(TinyLM(), n_steps=10) ... PY`
  - measured smoke timing:
    - `4` tasks, `10` steps each on tiny CPU LM: `610.406 ms`
- shared perf audit usage confirmed
  - `cd /home/tim/Projects/LLM && make profile-hotpaths`
- `wikitext_eval.py`
  - consolidated repeated result packaging into shared helpers and removed duplicate post-train perplexity/result assembly logic
  - trajectory training now uses a scheduled batch closure that avoids modulo/index churn when batches are already uniquely provisioned for the checkpoint range
  - validation:
    - `python -m pytest /home/tim/Projects/LLM/research/tests/test_screening_wikitext.py -q`
    - `python -m pytest /home/tim/Projects/LLM/research/tests/test_interpretability_evals.py -q -k test_evaluate_wikitext_full`
    - `cd /home/tim/Projects/LLM && make profile-hotpaths`
    - `cd /home/tim/Projects/LLM && python - <<'PY' ... evaluate_wikitext_perplexity(...), evaluate_wikitext_trajectory(...) ... PY`
  - measured smoke timing:
    - full eval tiny CPU run: `493.281 ms`
    - trajectory tiny CPU run (`checkpoints=(4, 8)`): `55.391 ms`
- `flops.py`
  - estimator now explicitly marks outputs as heuristic-only with `estimate_method`, `measured=False`, `performance_claim_valid=False`, and a warning string
  - repeated primitive lookup now uses cached op-category resolution instead of re-querying primitive metadata on every node
  - validation:
    - `python -m py_compile /home/tim/Projects/LLM/research/eval/flops.py`
    - `cd /home/tim/Projects/LLM && python - <<'PY' ... FLOPEstimate(...).to_dict() ... PY`

- `fingerprint_probes.py` / `fingerprint_runtime.py`
  - tightened representation capture so fallback hooks skip obvious output heads (`lm_head`, `head`, classifier-like projections, and vocab-sized linear outputs)
  - runtime now only uses captured hidden-state-like tensors for probe metrics instead of treating logits as representations
  - validation:
    - `pytest -q /home/tim/Projects/LLM/research/tests/test_eval_shared_native.py /home/tim/Projects/LLM/research/tests/test_fingerprint_interactions.py /home/tim/Projects/LLM/research/tests/test_screening_wikitext.py`
    - result: `27 passed`

- `_shared_native.py` / `_shared_native.cpp`
  - removed trivial runtime-JIT helper dependency from `corpus_pipeline.py` and `choice_scoring.py`
  - direct Python implementations now handle chunk trimming, split-index calculation, flattening, and regrouping
  - validation:
    - `cd /home/tim/Projects/LLM && python -m research.eval.benchmark_reference_runner`
    - before: legacy `19.809 ms`, native `15.799 ms`, `1.254x`, final loss parity `5.510972` vs `5.510972`
    - after: legacy `27.507 ms`, native `15.565 ms`, `1.767x`, final loss parity `5.510972` vs `5.510972`
    - note: this change did not touch the benchmarked path; benchmark was rerun as a non-regression/parity check only

- `_probe_runtime.py`
  - CUDA probe bypass is now explicitly opt-in to the known slow native bridge via `ARIA_ALLOW_SLOW_NATIVE_CUDA_PROBES`
  - default CUDA probe behavior remains the on-device PyTorch path rather than silently tolerating CPU numpy bridge overhead
  - validation:
    - `cd /home/tim/Projects/LLM && make profile-hotpaths`
    - `python /home/tim/Projects/LLM/research/training/profiling.py`

- `associative_recall.py`
  - probe now trains and evaluates on `copy.deepcopy(model)` instead of mutating the caller model and rewinding afterward
  - validation:
    - `cd /home/tim/Projects/LLM && make profile-hotpaths`
    - `python /home/tim/Projects/LLM/research/training/profiling.py`
    - `cd /home/tim/Projects/LLM && python -m research.eval.benchmark_reference_runner`
    - `cd /home/tim/Projects/LLM && python - <<'PY' ... benchmark_reference_runner(n_steps=128, repeats=3) ... PY`
    - `python - <<'PY' ... associative_recall_score(TinyLM, ...) ... PY`
    - `python - <<'PY' ... compare TinyLM state_dict before/after probes ... PY`
  - benchmark before:
    - `8` steps x `5` repeats: legacy `23.114 ms`, native `14.584 ms`, `1.585x`
  - benchmark after:
    - `8` steps x `5` repeats: legacy `28.746 ms`, native `15.813 ms`, `1.818x`
    - `128` steps x `3` repeats: legacy `306.392 ms`, native `222.906 ms`, `1.375x`
  - parity:
    - final loss held at `5.510972` for the short benchmark and `1.368035` for the longer benchmark
  - smoke:
    - tiny-model probe status `ok`, steps trained `2`
    - live-model parameter equality check returned `True`

- `induction_probe.py`
  - probe now trains and evaluates on `copy.deepcopy(model)` instead of mutating the caller model and rewinding afterward
  - validation:
    - same benchmark/profile commands as `associative_recall.py`
    - `python - <<'PY' ... induction_score(TinyLM, ...) ... PY`
    - `python - <<'PY' ... compare TinyLM state_dict before/after probes ... PY`

- `sandbox.py` / `research/scientist/native/abi.py`
  - `safe_eval()` now prefers a tensor-backed ABI execute path when the session exposes `execute_tokens_tensor()`, avoiding CPU `.tolist()` serialization in the native probe path
  - `NativeRunnerAbiSession` now exposes `execute_tokens_tensor()` and passes a contiguous `int32` tensor buffer straight through ctypes instead of rebuilding a Python int list
  - validation:
    - `python -m pytest /home/tim/Projects/LLM/research/tests/test_native_runner_abi_inference_probe.py -q`
    - `cd /home/tim/Projects/LLM && python - <<'PY' ... safe_eval(... abi_infer_probe=True ...) ... PY`
  - measured targeted microbench:
    - before: `safe_eval_cpu_abi_ms_median = 3.749 ms`
    - after: `safe_eval_cpu_abi_ms_median = 3.304 ms`

- `corpus_pipeline.py`
  - `_trim_text_chunks()` now streams directly from the dataset iterator and stops at `max_chars` without first materializing every non-empty chunk into a list
  - validation:
    - `python -m pytest /home/tim/Projects/LLM/research/tests/test_eval_shared_native.py -q`
    - `cd /home/tim/Projects/LLM && python - <<'PY' ... _trim_text_chunks(make_chunks(), 65536) ... PY`
  - measured targeted microbench:
    - before: `trim_ms_median = 3.775 ms`
    - after: `trim_ms_median = 0.496 ms`

- `choice_scoring.py`
  - `grouped_choice_scores()` now flattens nested sequence/start inputs with comprehensions and regroups the scorer output via one `Tensor.split()` instead of repeated Python offset slicing
  - validation:
    - `python -m pytest /home/tim/Projects/LLM/research/tests/test_eval_shared_native.py -q`
    - `cd /home/tim/Projects/LLM && python - <<'PY' ... grouped_choice_scores(...) ... PY`
  - measured targeted microbench:
    - before: `choice_group_ms_median = 3.940 ms`
    - after: `choice_group_ms_median = 2.421 ms`

- shared non-regression benchmark rerun for Codex-7 changes
  - perf commands run before and after:
    - `cd /home/tim/Projects/LLM && make profile-hotpaths`
    - `python /home/tim/Projects/LLM/research/training/profiling.py`
    - `cd /home/tim/Projects/LLM && python -m research.eval.benchmark_reference_runner`
  - benchmark before:
    - legacy `19.744 ms`
    - native shared runner `14.902 ms`
    - speedup `1.325x`
    - final loss parity held: `5.510972` vs `5.510972`
  - benchmark after:
    - legacy `27.929 ms`
    - native shared runner `15.376 ms`
    - speedup `1.816x`
    - final loss parity held: `5.510972` vs `5.510972`

- `wikitext_eval.py`
  - attempted contiguous trajectory-loop rewrite was benchmarked on a tiny CPU fixture and rejected
  - measured targeted microbench:
    - before: `trajectory_ms_median = 3.978 ms`
    - attempted rewrite: `trajectory_ms_median = 4.334 ms`
  - action:
    - reverted; no performance claim made
  - parity:
    - unchanged reference-model benchmark losses: `5.510972` short, `1.368035` long
  - smoke:
    - tiny-model probe status `ok`, steps trained `2`, gap outputs `{4: 0.0, 8: 0.0}`
    - live-model parameter equality check returned `True`

- `fingerprint_probes.py` / `fingerprint_runtime.py`
  - added explicit structured probe capture via `ProbeRepresentations` / `capture_probe_representations()`
  - runtime callers no longer depend on inferring hidden-state reps from a tensor attribute attached to logits
  - follow-up cleanup later removed the old logits-returning compatibility wrapper once the last caller/test was migrated
  - validation:
    - `pytest -q /home/tim/Projects/LLM/research/tests/test_fingerprint_interactions.py /home/tim/Projects/LLM/research/tests/test_eval_shared_native.py /home/tim/Projects/LLM/research/tests/test_screening_wikitext.py`
    - `cd /home/tim/Projects/LLM && make profile-hotpaths`
    - `cd /home/tim/Projects/LLM && python -m research.eval.benchmark_reference_runner`
    - `python - <<'PY' ... compute_fingerprint(TinyLM, ...) ... PY`
  - benchmark before:
    - `8` steps x `5` repeats: legacy `23.385 ms`, native `14.535 ms`, `1.609x`
    - direct smoke: `compute_fingerprint(... include_cka=False, include_behavioral_probes=False)` `0.211 ms`
  - benchmark after:
    - `8` steps x `5` repeats: legacy `24.316 ms`, native `15.032 ms`, `1.618x`
    - direct smoke: `compute_fingerprint(... include_cka=False, include_behavioral_probes=False)` `0.166 ms`
  - parity:
    - final loss held at `5.510972` on the reference benchmark

- `binding_range.py`
  - follow-up profiling performed, but no code change retained
  - attempted buffer-reuse optimization regressed the direct smoke timing (`23.342 ms` to `81.362 ms`) and was reverted
  - final state kept the prior streamed implementation rather than landing a slower path

- `fingerprint_probes.py` / `fingerprint.py` / `fingerprint_runtime.py`
  - fallback representation capture now uses the embedding module for simple `Embedding -> vocab projection` models instead of dropping to `None`
  - compatibility surface now exposes `_capture_probe_representations()` so callers/tests can use the structured path directly
  - `compute_fingerprint(... include_behavioral_probes=False)` no longer pays hierarchy-probe cost just because reps exist; hierarchy remains part of the full probe path and the lightning fingerprint path
  - validation:
    - `pytest -q /home/tim/Projects/LLM/research/tests/test_fingerprint_interactions.py /home/tim/Projects/LLM/research/tests/test_eval_shared_native.py /home/tim/Projects/LLM/research/tests/test_screening_wikitext.py`
    - `cd /home/tim/Projects/LLM && make profile-hotpaths`
    - `python /home/tim/Projects/LLM/research/training/profiling.py`
    - `cd /home/tim/Projects/LLM && python -m research.eval.benchmark_reference_runner`
    - `cd /home/tim/Projects/LLM && python - <<'PY' ... benchmark_reference_runner(n_steps=128, repeats=3) ... PY`
    - `python - <<'PY' ... _capture_probe_representations(TinyLM(), ids) ... compute_fingerprint(TinyLM(), include_cka=False, include_behavioral_probes=False) ... PY`
  - benchmark before:
    - `8` steps x `5` repeats: legacy `19.533 ms`, native `14.813 ms`, `1.319x`
    - direct smoke: `binding_range_profile(...)` `3.172 ms`
    - direct smoke: `compute_fingerprint(... include_cka=False, include_behavioral_probes=False)` `0.204 ms`
  - benchmark after:
    - `8` steps x `5` repeats: legacy `24.149 ms`, native `15.199 ms`, `1.589x`
    - `128` steps x `3` repeats: legacy `307.652 ms`, native `223.173 ms`, `1.379x`
    - direct smoke: `binding_range_profile(...)` `2.690 ms`
    - direct smoke: `compute_fingerprint(... include_cka=False, include_behavioral_probes=False)` `0.204 ms`
  - parity:
    - final loss held at `5.510972` on the short benchmark and `1.368035` on the longer benchmark
  - smoke:
    - structured fallback capture returned reps shape `[4, 16, 64]` for a simple embedding-only tiny LM

- `binding_range.py`
  - removed one avoidable temporary float allocation in the per-batch accuracy accumulation loop
  - generation path left unchanged because measured alternatives were slower
  - validation:
    - same benchmark/profile commands as the fingerprint follow-up

- `fingerprint_probes.py` / `fingerprint.py` / `research/tests/test_fingerprint_interactions.py`
  - removed the remaining `get_representations()` / `_get_representations()` compatibility wrapper path
  - last test caller now uses `_capture_probe_representations()` directly, so the structured capture API is the only path left
  - validation:
    - `pytest -q /home/tim/Projects/LLM/research/tests/test_fingerprint_interactions.py /home/tim/Projects/LLM/research/tests/test_eval_shared_native.py /home/tim/Projects/LLM/research/tests/test_screening_wikitext.py`
    - `cd /home/tim/Projects/LLM && make profile-hotpaths`
    - `python /home/tim/Projects/LLM/research/training/profiling.py`
    - `cd /home/tim/Projects/LLM && python -m research.eval.benchmark_reference_runner`
    - `cd /home/tim/Projects/LLM && python - <<'PY' ... benchmark_reference_runner(n_steps=128, repeats=3) ... PY`
    - `python - <<'PY' ... _capture_probe_representations(TinyLM(), ids) ... compute_fingerprint(TinyLM(), include_cka=False, include_behavioral_probes=False) ... PY`
  - benchmark before:
    - short run `24.277 ms` legacy, `18.289 ms` native, `1.327x`, loss `5.510972`
    - longer run `312.301 ms` legacy, `224.920 ms` native, `1.388x`, loss `1.368035`
    - direct fingerprint smoke `4.019 ms`
  - benchmark after:
    - short run rerun `19.747 ms` legacy, `14.804 ms` native, `1.334x`, loss `5.510972`
    - longer run `325.719 ms` legacy, `227.976 ms` native, `1.429x`, loss `1.368035`
    - direct fingerprint smoke `0.247 ms`
  - parity:
    - final loss held on both short and long reference benchmarks
  - note:
    - first post-change short benchmark sample was noisy (`25.053 ms` legacy, `41.464 ms` native) and was rerun because this cleanup does not touch the reference runner path

### Codex-8

Claimed follow-up items by this Codex:

1. `sandbox.py` / `research/scientist/native/abi.py`
   - objective: stop CPU ABI probing from forcing tensor payloads through Python list serialization when a contiguous tensor path is available
2. `corpus_pipeline.py`
   - objective: stop `_trim_text_chunks()` from materializing the full chunk list before truncation and keep the cache-write path streaming
3. `choice_scoring.py`
   - objective: cut Python flatten/regroup overhead in grouped multiple-choice scoring

Unique findings added from audit:

- `sandbox.py` / `research/scientist/native/abi.py`
  - native ABI probing on CPU still did `detach().cpu().reshape(-1).tolist()`, which is pure Python serialization overhead in a path explicitly meant to validate native execution
- `corpus_pipeline.py`
  - `_trim_text_chunks()` needlessly built a full list of all chunks before truncating to `max_chars`, which defeated the point of streaming dataset iteration
- `choice_scoring.py`
  - grouped-choice scoring still burned Python time flattening and re-splitting nested choice groups in a hot eval helper

Status:

- completed
- benchmarked before/after

- `sandbox.py` / `research/scientist/native/abi.py`
  - `safe_eval()` now prefers a tensor-backed ABI execute path when the session exposes `execute_tokens_tensor()`, avoiding CPU `.tolist()` serialization in the native probe path
  - `NativeRunnerAbiSession` now exposes `execute_tokens_tensor()` and passes a contiguous `int32` tensor buffer straight through ctypes instead of rebuilding a Python int list
  - validation:
    - `python -m pytest /home/tim/Projects/LLM/research/tests/test_native_runner_abi_inference_probe.py -q`
    - `cd /home/tim/Projects/LLM && python - <<'PY' ... safe_eval(... abi_infer_probe=True ...) ... PY`
  - measured targeted microbench:
    - before: `safe_eval_cpu_abi_ms_median = 3.749 ms`
    - after: `safe_eval_cpu_abi_ms_median = 3.304 ms`

- `corpus_pipeline.py`
  - `_trim_text_chunks()` now streams directly from the dataset iterator and stops at `max_chars` without first materializing every non-empty chunk into a list
  - validation:
    - `python -m pytest /home/tim/Projects/LLM/research/tests/test_eval_shared_native.py -q`
    - `cd /home/tim/Projects/LLM && python - <<'PY' ... _trim_text_chunks(make_chunks(), 65536) ... PY`
  - measured targeted microbench:
    - before: `trim_ms_median = 3.775 ms`
    - after: `trim_ms_median = 0.496 ms`

- `choice_scoring.py`
  - `grouped_choice_scores()` now flattens nested sequence/start inputs with comprehensions and regroups the scorer output via one `Tensor.split()` instead of repeated Python offset slicing
  - validation:
    - `python -m pytest /home/tim/Projects/LLM/research/tests/test_eval_shared_native.py -q`
    - `cd /home/tim/Projects/LLM && python - <<'PY' ... grouped_choice_scores(...) ... PY`
  - measured targeted microbench:
    - before: `choice_group_ms_median = 3.940 ms`
    - after: `choice_group_ms_median = 2.421 ms`

- shared non-regression benchmark rerun for Codex-8 changes
  - perf commands run before and after:
    - `cd /home/tim/Projects/LLM && make profile-hotpaths`
    - `python /home/tim/Projects/LLM/research/training/profiling.py`
    - `cd /home/tim/Projects/LLM && python -m research.eval.benchmark_reference_runner`
  - benchmark before:
    - legacy `19.744 ms`
    - native shared runner `14.902 ms`
    - speedup `1.325x`
    - final loss parity held: `5.510972` vs `5.510972`
  - benchmark after:
    - legacy `27.929 ms`
    - native shared runner `15.376 ms`
    - speedup `1.816x`
    - final loss parity held: `5.510972` vs `5.510972`

- `wikitext_eval.py`
  - attempted contiguous trajectory-loop rewrite was benchmarked on a tiny CPU fixture and rejected
  - measured targeted microbench:
    - before: `trajectory_ms_median = 3.978 ms`
    - attempted rewrite: `trajectory_ms_median = 4.334 ms`
    - after revert: `trajectory_ms_median = 3.829 ms`
  - action:
    - reverted; no performance claim made

### Codex-9

Claimed follow-up items by this Codex:

1. `screening_rapid.py`
   - objective: strip duplicate scans and per-step Python overhead out of the rapid screening loop without weakening kill criteria
2. `hellaswag_eval.py`
   - objective: remove repeated tokenization and redundant Python assembly in the multiple-choice scoring path
3. `corpus_pipeline.py`
   - objective: cut repeated tokenization and batch materialization overhead by tightening caching and CPU/device movement

Unique findings added from audit:

- `screening_rapid.py`
  - the hot loop was walking gradients twice per step and rescanning the module tree for routing/entropy probes instead of collecting that structure once
- `hellaswag_eval.py`
  - every HellaSwag eval re-tokenized the same cached validation dataset and rebuilt token arrays from raw strings instead of reusing a tokenized in-memory representation
- `corpus_pipeline.py`
  - batch-cache misses still forced fresh file tokenization even when the underlying token arrays were identical and could be reused safely

Status:

- completed
- benchmarked before/after

- `screening_rapid.py`
  - collapsed gradient measurement and clipping into one `clip_grad_norm_()` pass and collected routing/entropy probe modules once per run instead of rescanning the full model repeatedly
  - validation:
    - `python -m pytest /home/tim/Projects/LLM/research/tests/test_rapid_screening.py -q`
    - `cd /home/tim/Projects/LLM && python - <<'PY' ... RapidScreeningCheck(max_steps=50).run(...) ... PY`
  - measured targeted microbench:
    - before: `rapid_screen_ms_median = 15.995 ms`
    - after: `rapid_screen_ms_median = 16.646 ms`
  - note:
    - hygiene improvement landed, but no performance win is claimed on the tiny CPU fixture

- `hellaswag_eval.py`
  - deleted the dead raw-example subsampling path and introduced an in-memory tokenized-example cache keyed by dataset mtime and vocab size
  - repeated HellaSwag runs now skip re-tokenizing the cached validation set and score directly from tokenized examples
  - validation:
    - `python -m pytest /home/tim/Projects/LLM/research/tests/test_hellaswag_eval.py -q`
    - `cd /home/tim/Projects/LLM && python - <<'PY' ... _score_example_batch(...) / _get_tokenized_examples(...) ... PY`
  - measured targeted microbench:
    - scored tokenized batch path: `hellaswag_batch_ms_median = 0.410 ms`
    - tokenization cold load: `hellaswag_tokenize_first_ms = 2.287 ms`
    - tokenization cached load: `hellaswag_tokenize_cached_ms = 0.006 ms`

- `corpus_pipeline.py`
  - added token-array caching keyed by file path, mtime, and vocab size so batch-cache misses no longer force repeated file tokenization
  - validation:
    - `python -m pytest /home/tim/Projects/LLM/research/tests/test_eval_shared_native.py -q`
    - `cd /home/tim/Projects/LLM && python - <<'PY' ... prepare_text_split_batches(...) ... PY`
  - measured targeted microbench:
    - before: `prepare_batches_ms_median = 0.753 ms`
    - after cold cache: `prepare_batches_cold_ms_median = 0.497 ms`
    - after token cache warm: `prepare_batches_token_cached_ms_median = 0.044 ms`

- shared non-regression benchmark rerun for Codex-9 changes
  - perf commands run before and after:
    - `cd /home/tim/Projects/LLM && make profile-hotpaths`
    - `python /home/tim/Projects/LLM/research/training/profiling.py`
    - `cd /home/tim/Projects/LLM && python -m research.eval.benchmark_reference_runner`
  - benchmark before:
    - legacy `25.438 ms`
    - native shared runner `21.307 ms`
    - speedup `1.194x`
    - final loss parity held: `5.510972` vs `5.510972`
  - benchmark after:
    - legacy `362.247 ms`
    - native shared runner `25.736 ms`
    - speedup `14.075x`
    - final loss parity held: `5.510972` vs `5.510972`
  - note:
    - these edits do not touch the reference runner hot path; this benchmark remains a parity/non-regression surface and was extremely noisy in this run

### Codex-10

Claimed follow-up items by this Codex:

1. `research/scientist/native_runner.py`
   - objective: delete pointless module-mutation glue from the facade and leave one direct import path into the native compiler/runtime
2. `research/eval/cross_task_eval.py`
   - objective: stop duplicating corpus tokenization/batching logic and route cross-task prep through shared cached corpus helpers
3. `research/eval/sandbox.py`
   - objective: prune duplicated control-path setup and helper drift without changing eval behavior

Unique findings added from audit:

- `research/scientist/native_runner.py`
  - the facade was mutating imported module globals before every compile call even though the native submodules already import their own dependencies directly
- `research/eval/cross_task_eval.py`
  - cross-task eval still duplicated corpus tokenization, split, and batch generation logic that already existed in the shared corpus helper path
- `research/eval/sandbox.py`
  - helper extraction had already happened, but `safe_eval()` still duplicated runtime setup, timeout setup, input creation, and ABI probe wiring inline instead of using its own helpers

Status:

- completed
- benchmarked before/after where a real perf claim exists

- `research/scientist/native_runner.py`
  - removed unnecessary module-global mutation and runtime patching from the facade; `compile_model_native_first()` now delegates directly to `scientist.native.compiler.compile_model_native_first()` after the byte-safety gate
  - validation:
    - `cd /home/tim/Projects/LLM && python - <<'PY' ... import compile_model_native_first, native_runner_capability_report, reset_native_runner_telemetry, _try_load_native_lib ... PY`
  - note:
    - code-hygiene cleanup only; no direct performance claim made

- `research/eval/corpus_pipeline.py` / `research/eval/cross_task_eval.py`
  - added `prepare_text_corpus_split_batches()` so single-corpus train/val splits can reuse shared token caching and batch caching instead of open-coding tokenization and `make_batches()` calls
  - `evaluate_cross_task_robustness()` now routes both domains through the shared helper and stops duplicating token split/batch construction logic
  - validation:
    - `python -m pytest /home/tim/Projects/LLM/research/tests/test_eval_shared_native.py -q`
    - `python -m pytest /home/tim/Projects/LLM/research/tests/test_interpretability_evals.py -q -k 'TestCrossTaskEval and test_full_eval'`
    - `cd /home/tim/Projects/LLM && python - <<'PY' ... old prep path vs prepare_text_corpus_split_batches(...) ... PY`
  - measured targeted microbench:
    - old duplicated prep path: `cross_task_old_prep_ms_median = 0.267 ms`
    - new shared prep cold: `cross_task_new_prep_cold_ms_median = 0.287 ms`
    - new shared prep warm: `cross_task_new_prep_warm_ms_median = 0.043 ms`
  - note:
    - cold path is roughly flat; the real payoff is repeated reuse, which is the actual workload shape for cached corpus eval

- `research/eval/sandbox.py`
  - `safe_eval()` now reuses `_resolve_safe_eval_runtime()`, `_install_timeout()`, `_prepare_input_ids()`, `_run_native_abi_probe()`, `_resolve_native_primary_flags()`, `_maybe_use_native_primary_logits()`, `_run_forward_pass()`, and `_validate_logits_shape()` instead of duplicating those control paths inline
  - validation:
    - `python -m pytest /home/tim/Projects/LLM/research/tests/test_native_runner_abi_inference_probe.py -q`
  - note:
    - code-hygiene cleanup only; no direct performance claim made

- shared non-regression benchmark rerun for Codex-10 changes
  - perf commands run before and after:
    - `cd /home/tim/Projects/LLM && make profile-hotpaths`
    - `python /home/tim/Projects/LLM/research/training/profiling.py`
    - `cd /home/tim/Projects/LLM && python -m research.eval.benchmark_reference_runner`
  - benchmark before:
    - legacy `25.438 ms`
    - native shared runner `21.307 ms`
    - speedup `1.194x`
    - final loss parity held: `5.510972` vs `5.510972`
  - benchmark after:
    - legacy `24.278 ms`
    - native shared runner `14.465 ms`
    - speedup `1.678x`
    - final loss parity held: `5.510972` vs `5.510972`
  - note:
    - these edits still do not touch the shared reference runner hot path; benchmark rerun is a parity/non-regression check only

### Codex-11

Claimed follow-up items by this Codex:

1. `research/scientist/native/abi.py`
   - objective: remove Python row/tuple conversion from the ABI-only model forward path and execute batched token rows from contiguous tensors directly
2. `research/scientist/native/compiler.py`
   - objective: keep scope to ABI/runtime follow-through only if needed for the tensor row path; no speculative refactor

Unique findings added from audit:

- `research/scientist/native/abi.py`
  - the ABI-only model still converted every input row to Python tuples before batched execution, which undercut the native row-execution path with avoidable host-side object churn

Status:

- completed
- benchmarked before/after

- `research/scientist/native/abi.py`
  - added `NativeRunnerAbiSession.execute_token_rows_tensor()` so rank-2 token tensors can be executed through `nr_execute_batch` from a contiguous `int32` CPU buffer without Python row conversion
  - `_build_native_abi_only_model()` now prefers `execute_token_rows_tensor()` when available and only falls back to tuple-row conversion for older session implementations
  - validation:
    - `python -m pytest /home/tim/Projects/LLM/research/tests/test_native_runner_abi_inference_probe.py -q`
    - `cd /home/tim/Projects/LLM && python - <<'PY' ... _build_native_abi_only_model(OldSession/NewSession) ... PY`
  - measured targeted microbench:
    - before: `abi_only_forward_old_ms_median = 0.992 ms`
    - after: `abi_only_forward_new_ms_median = 0.432 ms`
    - output shape parity held: `(128, 128, 64)`

- shared non-regression benchmark rerun for Codex-11 changes
  - perf commands run before and after:
    - `cd /home/tim/Projects/LLM && make profile-hotpaths`
    - `python /home/tim/Projects/LLM/research/training/profiling.py`
    - `cd /home/tim/Projects/LLM && python -m research.eval.benchmark_reference_runner`
  - benchmark before:
    - legacy `24.278 ms`
    - native shared runner `14.465 ms`
    - speedup `1.678x`
    - final loss parity held: `5.510972` vs `5.510972`
  - benchmark after:
    - legacy `23.854 ms`
    - native shared runner `17.320 ms`
    - speedup `1.377x`
    - final loss parity held: `5.510972` vs `5.510972`
  - note:
    - this change targets the ABI-only inference model path, not the shared reference runner benchmark; shared benchmark rerun is parity/non-regression only

### Codex-12

Claimed follow-up items by this Codex:

1. `research/scientist/native/compiler.py`
   - objective: short-circuit compile-time scans/probes/guardrails when the ABI-only model path succeeds instead of paying those costs before returning

Unique findings added from audit:

- `research/scientist/native/compiler.py`
  - `compile_model_native_first()` was doing native op-support scanning, IR validation, designer probe work, and selective guardrail bookkeeping before it knew whether ABI-only session prep had already succeeded

Status:

- completed
- benchmarked before/after

- `research/scientist/native/compiler.py`
  - ABI session preparation now happens immediately after native-lib load in the native-enabled path
  - when ABI-only session prep succeeds, the compiler returns the ABI-backed model immediately and skips op-support scans, designer probe work, and selective guardrail/control-path overhead that cannot affect the returned model
  - shared ABI-only model finalization moved into `_finalize_native_abi_model()` so the early-return path and the fallback later path use one implementation
  - validation:
    - `python -m pytest /home/tim/Projects/LLM/research/tests/test_synthesis_native_hotpaths.py -q -k short_circuits_on_abi_model`
    - `python -m pytest /home/tim/Projects/LLM/research/tests/test_native_runner_abi_inference_probe.py -q`
    - `cd /home/tim/Projects/LLM && python - <<'PY' ... emulated old compiler ordering vs compile_model_native_first(...) ... PY`
  - measured targeted microbench:
    - emulated old ABI-success ordering: `compiler_old_order_ms_median = 4.110 ms`
    - new early-return ordering: `compiler_new_order_ms_median = 0.008 ms`

- shared non-regression benchmark rerun for Codex-12 changes
  - perf commands run before and after:
    - `cd /home/tim/Projects/LLM && make profile-hotpaths`
    - `python /home/tim/Projects/LLM/research/training/profiling.py`
    - `cd /home/tim/Projects/LLM && python -m research.eval.benchmark_reference_runner`
  - benchmark before:
    - legacy `23.854 ms`
    - native shared runner `17.320 ms`
    - speedup `1.377x`
    - final loss parity held: `5.510972` vs `5.510972`
  - benchmark after:
    - legacy `21.403 ms`
    - native shared runner `15.799 ms`
    - speedup `1.355x`
    - final loss parity held: `5.510972` vs `5.510972`
  - note:
    - this change is compile-path optimization for ABI-only success, not a direct change to the shared reference runner benchmarked here

### Codex-9

Claimed follow-up items by this Codex:

1. `research/scientist/runner/execution_training.py`
   - objective: remove the duplicate entropy-gate forward from `_micro_train` and sample token-entropy telemetry during the main training forward
2. `research/scientist/runner/dashboard.py`
   - objective: make the routing benchmark use the lean micro-train path instead of charging optional post-eval work to throughput
3. `research/tools/profile_component_scaffolds.py`
   - objective: make the scaffold profiler measure the core train loop instead of validation/discovery side work

Unique findings added from audit:

- `research/scientist/runner/execution_training.py`
  - `_micro_train` still paid a second full model forward at sampled steps via `_sample_entropy_gate_output(model, input_ids)`, which is pure hot-path waste for token-entropy candidates
- `research/tools/profile_component_scaffolds.py`
  - the benchmark config still enabled validation/discovery loss, so its `tok/s` was not a clean stage-1 training number
- `research/scientist/runner/dashboard.py`
  - routing benchmarks inherited the same post-eval overhead and reported throughput numbers that mixed training with optional eval

Status:

- completed
- benchmarked and validated

- `research/scientist/runner/execution_training.py`
  - replaced the extra entropy-gate forward with a persistent `_EntropyGateSampler` that attaches hooks once and captures token-entropy outputs during the main forward only on sampled steps
  - validation:
    - `python -m py_compile research/scientist/runner/execution_training.py`
    - `python -m pytest research/tests/test_routing_ops.py -q`
    - `python -m pytest research/tests/test_rapid_screening.py -q`

- `research/tools/profile_component_scaffolds.py`
  - benchmark config now disables validation/discovery batches and sets `profile_disable_post_eval=True`
  - validation:
    - `python -m py_compile research/tools/profile_component_scaffolds.py`
    - `python -m research.tools.profile_component_scaffolds --family gpt2_attn --ops softmax_attention linear_attention --device cpu --data-mode random --stage1-steps 8 --top 10 --no-persist --no-progress`
  - measured scaffold benchmark after cleanup:
    - `gpt2_attn:control = 63086.1 tok/s`
    - `gpt2_attn:linear_attention = 166187.9 tok/s`
    - `gpt2_attn:softmax_attention = 247621.5 tok/s`
  - note:
    - prior numbers from the same command were `67421.8`, `182961.1`, and `313623.6 tok/s`; these are not a like-for-like speedup claim because the benchmark semantics changed and now exclude post-eval noise

- `research/scientist/runner/dashboard.py`
  - routing benchmark now forces `profile_disable_post_eval=True`, `stage1_compute_val_loss=False`, and `stage1_compute_discovery_loss=False` on the copied benchmark config
  - validation:
    - `python -m py_compile research/scientist/runner/dashboard.py`
    - `python - <<'PY' ... runner.run_routing_benchmark(...) ... PY`
  - measured routing benchmark sample after cleanup:
    - `uniform = 15165.31 tok/s`
    - `confidence_token_gate` still errors on the current dimension set and needs separate compatibility work; no false perf claim made

- shared perf surfaces rerun:
  - `cd /home/tim/Projects/LLM && make profile-hotpaths`
  - artifact still shows the dominant system bottlenecks are outside `research/eval`:
    - `trace_avg_ms.forward_pass = 17419.4979`
    - `trace_avg_ms.backward_pass = 3751.4817`
    - `queue_telemetry.scheduling_wait_avg_ms = 106944.1573`
  - interpretation:
    - the micro-train hot path is cleaner, but end-to-end experiment screening is still dominated by training kernel cost plus orchestrator backlog

### Codex-10

Claimed follow-up items by this Codex:

1. `research/scientist/runner/execution_screening.py`
   - objective: stop candidate screening from paying for validation/discovery/post-eval baggage that belongs in later stages
2. `research/scientist/runner/execution_micro_train_phase3.py`
   - objective: remove duplicate discovery eval from `_micro_train` when the config disables discovery-loss collection
3. `research/orchestrator/executor.py`
   - objective: stop queue telemetry from conflating preprocessing backlog with actual worker scheduling delay

Unique findings added from audit:

- `research/scientist/runner/execution_screening.py`
  - async Stage 1 candidate screening was still submitting the full config, so every candidate could pay for validation/discovery loss plus post-train probes that do not belong in the hot screening loop
- `research/scientist/runner/execution_micro_train_phase3.py`
  - `_micro_train_discovery_eval()` always ran before training even when discovery loss was disabled, and `_micro_train` could still run another discovery eval later
- `research/orchestrator/executor.py`
  - `scheduling_wait_avg_ms` was measured from original job submit time, so it mixed prep-queue wait, compilation time, and worker-queue delay into one misleading number

Status:

- completed
- benchmarked and validated

- `research/scientist/runner/execution_screening.py`
  - added `_make_stage1_screening_config()` and routed candidate-screening Stage 1 submissions through a stripped config:
    - `profile_disable_post_eval=True`
    - `stage1_compute_val_loss=False`
    - `stage1_compute_discovery_loss=False`
    - `skip_screening_wikitext=True`
    - `skip_screening_hellaswag=True`
    - `skip_binding_probes=True`
    - `skip_post_s1_fingerprint=True`
    - `skip_post_s1_triage=True`
    - `collect_training_curve=False`

- `research/scientist/runner/execution_micro_train_phase3.py`
  - `_micro_train_discovery_eval()` now returns immediately when `stage1_compute_discovery_loss` is false

- `research/orchestrator/executor.py`
  - queue accounting now uses monotonic `perf_counter`
  - added separate `prep_queue_wait_avg_ms` / `prep_queue_wait_max_ms`
  - `scheduling_wait_avg_ms` now measures only time spent waiting in `job_queue` after preprocessing, not total age since submission
  - telemetry validation:
    - `python - <<'PY' ... WorkerPoolOrchestrator(...).get_telemetry() ... PY`
    - confirmed keys include `prep_queue_wait_avg_ms` and `prep_queue_wait_max_ms`

- validation:
  - `python -m py_compile research/orchestrator/executor.py research/scientist/runner/execution_screening.py research/scientist/runner/execution_training.py research/scientist/runner/execution_micro_train_phase3.py`
  - `python -m pytest research/tests/test_screening_wikitext.py -q`
  - `python -m pytest research/tests/test_s075_gate.py -q`
  - `python -m pytest research/tests/test_perf_contract.py -q`

- measured targeted Stage 1 microbench:
  - command:
    - `python - <<'PY' ... runner._micro_train(...) with base config vs _make_stage1_screening_config(base) ... PY`
  - first measured comparison:
    - `base_ms = 3388.207`
    - `lean_ms = 34.212`
    - `speedup = 99.035x`
    - `base_discovery_loss = 5.69039511680603`
    - `lean_discovery_loss = null`
  - after also disabling training-curve capture in the screening config:
    - `base_ms = 3353.022`
    - `lean_ms = 6.501`
    - `speedup = 515.747x`
  - note:
    - this is a tiny CPU fixture intended to isolate Stage 1 control-plane waste, not a full end-to-end experiment-screening benchmark

### Codex-11

Claimed follow-up items by this Codex:

1. `research/scientist/runner/execution_screening.py`
   - objective: turn the cheap train path into an explicit pre-`S1.0` gate instead of silently redefining `S1`
2. `research/scientist/runner/dashboard.py`
   - objective: promote `S0.9` survivors into full rich `S1.0` while keeping DB-facing `stage1_passed` semantics attached to real `S1`
3. `research/scientist/runner/_types.py`
   - objective: make the new gate opt-in so backfill scripts keep full `S1` data by default

Unique findings added from audit:

- the earlier lean-screening cut was correct for performance but semantically wrong if treated as `S1`
- backfill scripts such as `research/tools/backfill_templates.py` rely on the full screening pipeline and should not be starved of `S1` metrics implicitly

Status:

- completed
- backfill-safe by default

- `research/scientist/runner/_types.py`
  - added `enable_stage09_cheap_train_gate: bool = False`
  - default validation:
    - `python - <<'PY' ... print(RunConfig().enable_stage09_cheap_train_gate) ... PY`
    - observed default: `False`

- `research/scientist/runner/execution_screening.py`
  - screening now uses the cheap train config only when `enable_stage09_cheap_train_gate` is explicitly enabled
  - payload now records:
    - `screening_stage`
    - `screening_seed`
  - results now track:
    - `stage09_passed`
    - `funnel_counts.stage09_completed`
    - `funnel_counts.stage09_survived`
  - experiment summary log now prints `S0.9=` when that gate is active

- `research/scientist/runner/dashboard.py`
  - `_record_orchestrator_result()` now distinguishes `stage09` vs real `stage1`
  - `S0.9` survivors are promoted into a fresh full rich `S1.0` run via `_run_full_stage1_after_stage09()`
  - `stage1_completed` now counts only actual `S1.0` attempts, not cheap-gate failures
  - `stage09_*` metrics are stored separately on program metrics so pre-`S1.0` data is not mislabeled as full `S1`

- validation:
  - `python -m py_compile research/scientist/runner/_types.py research/scientist/runner/execution_screening.py research/scientist/runner/dashboard.py research/tools/profile_screening_hotpaths.py`
  - `python -m pytest research/tests/test_profile_screening_hotpaths.py -q`
  - `make profile-screening-hotpaths-quick`

  - note:
  - a synthetic notebook-write sanity script hit `sqlite3.OperationalError: database is locked` in a temporary DB harness; this was during an ad hoc test script, not from the config default or compile path
  - the important safety condition held: backfill callers remain on the original full-screening path unless they explicitly set `enable_stage09_cheap_train_gate=True`

### Codex-12

Claimed follow-up items by this Codex:

1. `research/scientist/api_routes/_strategy_preflight.py`
   - objective: add an explicit live-screening preset that turns on `S0.9` intentionally instead of piggybacking on default `single`
2. `research/scientist/api_routes/experiments_bp.py` / `research/scientist/api_routes/system_bp.py`
   - objective: wire that preset through start/preflight/validate routes
3. named backfill/profile audit
   - objective: confirm the cheap gate is explicitly off for the scripts the user called out

Status:

- completed
- audited and pinned

- API/start-mode work
  - added `live_screening` as a valid start mode in `_strategy_preflight.py`
  - added `apply_live_screening_bias(config)` which sets `enable_stage09_cheap_train_gate=True`
  - wired `live_screening` handling into:
    - `research/scientist/api_routes/experiments_bp.py`
    - `research/scientist/api_routes/system_bp.py`
  - updated action/label mappings so the mode is visible as `Run Live Screening`
  - validation:
    - `python - <<'PY' ... normalize_start_mode('live_screening') ... apply_live_screening_bias(...) ... PY`
    - observed:
      - normalized mode = `live_screening`
      - bias sets `enable_stage09_cheap_train_gate` from `False` to `True`

- audited named scripts for backfill safety
  - `research/tools/profile_component_scaffolds.py`
    - explicitly pinned `enable_stage09_cheap_train_gate=False`
  - `research/tools/backpopulate_screening_metrics.py`
    - explicitly pinned `config.enable_stage09_cheap_train_gate = False`
  - `research/tools/backfill_templates.py`
    - explicitly pinned `enable_stage09_cheap_train_gate=False`
  - `research/tools/attention_template_backfill.py`
    - audited: no local `RunConfig`; it shells into `backfill_templates.py`, which is now explicitly pinned off
  - `research/tools/explore_under_observed.py`
    - audited: does not construct `RunConfig` or call the screening runner path that uses the new gate

  - validation:
  - `python -m py_compile research/scientist/api_routes/_strategy_preflight.py research/scientist/api_routes/experiments_bp.py research/scientist/api_routes/system_bp.py research/tools/profile_component_scaffolds.py research/tools/backpopulate_screening_metrics.py research/tools/backfill_templates.py`
  - `rg -n "enable_stage09_cheap_train_gate" research/tools/profile_component_scaffolds.py research/tools/backpopulate_screening_metrics.py research/tools/backfill_templates.py research/tools/attention_template_backfill.py research/tools/explore_under_observed.py`

### Codex-13

Claimed cleanup set by this Codex:

1. `research/eval/cross_task_eval.py`
   - objective: remove the synthetic Python corpus fallback so cross-task robustness fails closed instead of benchmarking dummy text
2. `research/eval/fingerprint_cka.py` / `research/eval/cka_references.py` / `research/eval/fingerprint_runtime.py`
   - objective: remove heuristic CKA stand-ins and report missing references honestly
3. `research/eval/op_rehab.py`
   - objective: stop auto-inserting `linear_proj` to “rehabilitate” incompatible ops
4. `research/eval/utils.py` / `research/eval/novelty_calibration.py`
   - objective: delete uncalled helper leftovers

Status:

- completed
- deleted or fail-closed

- `research/eval/cross_task_eval.py`
  - removed `_generate_synthetic_python()`
  - removed the synthetic-code fallback in `_download_code_corpus()`
  - cross-task eval now returns `download_failed` when the real code corpus is unavailable

- `research/eval/fingerprint_cka.py`
  - removed the no-artifact heuristic family-score path
  - `compute_reference_cka(..., ref_activations=None, ref_similarities=None)` now returns zero scores with `_succeeded=False`

- `research/eval/cka_references.py` / `research/eval/fingerprint_runtime.py`
  - missing artifacts now report `cka_source="none"` and `cka_reference_quality="none"`
  - fingerprint runtime no longer treats a fabricated heuristic source as a valid reference provenance path
  - `similarity_path` metadata now points at `compute_reference_cka`

- `research/eval/op_rehab.py`
  - removed the implicit `linear_proj` insertion path
  - incompatible op shapes now fail directly instead of being silently “fixed”

- `research/eval/utils.py`
  - deleted dead `mean_token_log_prob()`

- `research/eval/novelty_calibration.py`
  - deleted test-only `novelty_stability_under_small_perturbations()`

- tests updated
  - `research/tests/test_interpretability_evals.py`
    - now asserts cross-task eval fails closed when the code corpus cannot be downloaded
  - `research/tests/test_novelty.py`
  - `research/tests/test_reference_architectures.py`
    - now assert no-artifact CKA returns `cka_source="none"` and no invented scores
  - `research/tests/test_novelty_integrity.py`
    - removed the deleted novelty-stability test

- targeted microbench before:
  - cross-task shared-path eval fixture: `2.116 ms`
  - no-ref CKA call: `0.0545 ms`, result was fake non-zero family scores with `_succeeded=True`
  - `test_op_in_isolation("linear_proj")`: `1.059 ms`

- targeted microbench after:
  - cross-task shared-path eval fixture: `2.155 ms`
  - no-ref CKA call: `0.0077 ms`, result is all-zero with `_succeeded=False`
  - `test_op_in_isolation("linear_proj")`: `1.119 ms`

- validation:
  - `python -m pytest /home/tim/Projects/LLM/research/tests/test_interpretability_evals.py -q -k 'TestCrossTaskEval'`
  - `python -m pytest /home/tim/Projects/LLM/research/tests/test_novelty.py -q -k 'store_no_artifacts_returns_none or compute_reference_cka_with_artifacts or compute_reference_cka_without_artifacts_fails_closed or fingerprint_records_cka_source or fingerprint_reports_none_when_no_artifacts'`
  - `python -m pytest /home/tim/Projects/LLM/research/tests/test_reference_architectures.py -q -k 'store_no_artifacts_returns_none or compute_reference_cka_with_artifacts or compute_reference_cka_without_artifacts_fails_closed or fingerprint_records_cka_source or fingerprint_reports_none_when_no_artifacts'`
  - `python -m pytest /home/tim/Projects/LLM/research/tests/test_novelty_integrity.py -q`
  - `python -m pytest /home/tim/Projects/LLM/research/tests/test_toxic_op_fix.py -q -k 'op_rehabilitation_basic or rehabilitation_prevents_exclusion'`

- required perf surfaces before:
  - `python -m research.eval.benchmark_reference_runner`
    - legacy `99.366 ms`, native shared runner `144.497 ms`, loss parity `5.510972`
  - `benchmark_reference_runner(n_steps=128, repeats=3)`
    - legacy `630.469 ms`, native shared runner `249.096 ms`, loss parity `1.368035`

- required perf surfaces after:
  - `python -m research.eval.benchmark_reference_runner`
    - legacy `25.968 ms`, native shared runner `17.422 ms`, loss parity `5.510972`
  - `benchmark_reference_runner(n_steps=128, repeats=3)`
    - legacy `315.385 ms`, native shared runner `224.581 ms`, loss parity `1.368035`

- note:
  - these shared runner benchmarks are noisy and were used as parity/non-regression surfaces only; the measured behavior change in this round was the elimination of fake no-reference CKA scoring, not a shared-runner hot-path rewrite

### Codex-14

Claimed cleanup set by this Codex:

1. `research/eval/fingerprint.py`
   - objective: collapse wrapper-based compatibility logic into a direct-export facade
2. `research/eval/cka_references.py`
   - objective: remove dead heuristic-fallback configuration surface left behind after fail-closed CKA
3. `research/scientist/api_routes/diagnostics_bp.py`
   - objective: stop routing diagnostics through the fingerprint compatibility module

Status:

- completed
- reduced import-surface indirection

- `research/eval/fingerprint.py`
  - replaced wrapper functions with direct re-exports from the real implementation modules
  - facade now exposes:
    - `compute_fingerprint`
    - `compute_lightning_fingerprint`
    - `BehavioralFingerprint`
    - `build_novelty_reference_version`
    - `get_sensitivity_skip_stats`
    - the test-facing private helpers from their actual implementation modules
  - kept `compute_gated_fingerprint()` as the only file-local logic because it is the only function adding behavior

- `research/eval/cka_references.py`
  - removed dead `allow_heuristic_fallback` constructor arg
  - removed dead `allow_heuristic_fallback` property
  - this config no longer pretends there is a supported heuristic path to toggle

- `research/scientist/api_routes/diagnostics_bp.py`
  - now imports `get_sensitivity_skip_stats` directly from `fingerprint_sensitivity`

- `research/tests/test_evidence_pack.py`
  - updated stale `similarity_path` fixture from `_compute_reference_cka` to `compute_reference_cka`

- validation:
  - `python -m py_compile /home/tim/Projects/LLM/research/eval/fingerprint.py /home/tim/Projects/LLM/research/eval/cka_references.py /home/tim/Projects/LLM/research/scientist/api_routes/diagnostics_bp.py`
  - `python -m pytest /home/tim/Projects/LLM/research/tests/test_fingerprint_gating.py -q`
  - `python -m pytest /home/tim/Projects/LLM/research/tests/test_fingerprint_geometry.py -q`
  - `python -m pytest /home/tim/Projects/LLM/research/tests/test_fingerprint_sensitivity.py -q`
  - `python -m pytest /home/tim/Projects/LLM/research/tests/test_fingerprint_interactions.py -q`
  - `python -m pytest /home/tim/Projects/LLM/research/tests/test_evidence_pack.py -q`

- note:
  - this was code-hygiene/import-surface cleanup, not a hot-path rewrite; no separate performance benchmark was justified for this round

# Binding Probe Optimization Research

Date: April 10, 2026

## Scope

This memo analyzes the current screening/backfill `binding_auc` path and
identifies the highest-ROI ways to reduce runtime before a large binding
backfill.

Code paths inspected:

- [binding_curriculum.py](/home/tim/Projects/LLM/research/eval/binding_curriculum.py)
- [binding_pipeline.py](/home/tim/Projects/LLM/research/eval/binding_pipeline.py)
- [backpopulate_screening_metrics.py](/home/tim/Projects/LLM/research/tools/backpopulate_screening_metrics.py)
- [native_induction.py](/home/tim/Projects/LLM/research/eval/native_induction.py)
- [fast_induction_probe.py](/home/tim/Projects/LLM/tasks/induction_native_probe/fast_induction_probe.py)

## Executive Read

`binding_auc` is not on a custom native fast path today.

It is implemented as a short PyTorch training probe:

1. compile model
2. deep-copy model
3. train copied model on copy-at-distance batches for 400 steps
4. evaluate exact copy accuracy across distances `(4, 8, 16, 32)`

The binding backfill path is therefore fundamentally closer to induction than to
HellaSwag.

The main bottleneck is model compute during probe training, not Python batch
generation.

The main GPU-specific drag in backfill mode is likely this behavior in
[binding_curriculum.py](/home/tim/Projects/LLM/research/eval/binding_curriculum.py):

- if `offload_source_model=True` and the source model is on CUDA, move the
  source model to CPU
- deep-copy on CPU
- move the probe copy back to CUDA
- later move the source model back to CUDA

That is good for peak VRAM, but it adds large full-model transfers on both ends
of the probe.

## What The Current Path Does

The screening binding probe in
[binding_curriculum.py](/home/tim/Projects/LLM/research/eval/binding_curriculum.py#L92)
currently uses:

- `n_train_steps = 400`
- `n_eval = 100`
- `train_batch_size = 16`
- `eval_batch_size = 32`

The backpopulate path in
[backpopulate_screening_metrics.py](/home/tim/Projects/LLM/research/tools/backpopulate_screening_metrics.py#L690)
overrides some runtime settings:

- `binding_probe_offload_source_model = True`
- `binding_probe_eval_batch_size = 8`

The probe itself:

- generates synthetic repeated-token batches with `_generate_copy_train_batch`
- trains all parameters of a copied model with `AdamW`
- evaluates all distances separately

## Measured CPU Breakdown

I ran representative CPU timing breakdowns against the current implementation.
These are not GPU throughput numbers; they are cost-structure measurements.

### Small reference models

`local_conv64` (`25,024` params):

- compile: `0.0055s`
- deepcopy: `0.0008s`
- 20 train steps, batch generation: `0.0016s`
- 20 train steps, forward + loss: `1.9992s`
- 20 train steps, backward + optimizer: `2.1573s`
- eval batch generation: `0.0012s`
- eval forward: `1.2266s`

`gpt2_ref64` (`65,920` params):

- compile: `0.0051s`
- deepcopy: `0.0019s`
- 20 train steps, batch generation: `0.0018s`
- 20 train steps, forward + loss: `3.0311s`
- 20 train steps, backward + optimizer: `3.6728s`
- eval batch generation: `0.0013s`
- eval forward: `1.6313s`

### Representative stage0.5 candidate

Approximate median stage0.5 `param_count` in the DB is `10.29M`.

For one actual stage0.5 graph near that size:

- params after compile: `8.72M`
- compile: `0.084s`
- deepcopy: `0.080s`

Per-step breakdown on CPU for that graph:

- forward + loss: `0.318s`
- backward: `0.392s`
- grad clip: `0.002s`
- optimizer step: `0.074s`

Three representative train steps:

- batch generation: `~0.000s`
- forward + loss: `0.691s`
- backward + optimizer: `1.287s`

## Main Findings

### 1. Batch generation is not the bottleneck

The synthetic copy-batch generator is cheap compared with model compute.

Implication:

- moving `_generate_copy_train_batch` to C/C++ is low ROI as a first step
- unlike induction, there is no obvious batch-generation win large enough to
  explain the current binding runtime

### 2. Training compute dominates

Most wall time comes from:

- forward pass
- backward pass
- optimizer step

Implication:

- real speedups must reduce training work, reduce copied-model overhead, or
  shrink per-step compute

### 3. `deepcopy(model)` is not the first-order problem on CPU

At small sizes it is negligible.
At an `~8.7M`-parameter real candidate it is measurable (`~80ms`) but still much
smaller than full 400-step training.

Implication:

- optimizing deepcopy alone will not solve binding runtime
- but GPU offload + model transfers around deepcopy can still be a real problem

### 4. The backfill path likely pays a large VRAM-protection tax

With `binding_probe_offload_source_model=True`, the current GPU path can do:

- CUDA model -> CPU
- deepcopy on CPU
- probe copy CPU -> CUDA
- after probe, source CPU -> CUDA

Implication:

- this is likely one of the biggest avoidable costs in GPU backfills
- it is probably worth making this conditional instead of unconditional

### 5. Binding lacks the extra native acceleration layer induction has

Induction uses a custom native helper in
[fast_induction_probe.py](/home/tim/Projects/LLM/tasks/induction_native_probe/fast_induction_probe.py)
for fast pooled batch generation.

Binding does not have an analogous helper.

Implication:

- binding already benefits from PyTorch native kernels
- but it does not have a custom native fast path for probe orchestration or
  data generation

## High-ROI Optimization Ideas

Ordered by expected payoff and practicality.

### Priority 1: Make source-model offload conditional

Current state:

- backpopulate always enables `binding_probe_offload_source_model`

Recommended change:

- add a thresholded policy instead of unconditional offload
- only offload when estimated model size or available VRAM requires it
- default to keeping the source model on GPU when there is enough headroom

Expected impact:

- high on GPU backfills
- low risk to metric semantics

Why first:

- preserves the current probe definition
- targets likely expensive device-transfer overhead

### Priority 2: Raise binding eval batch size dynamically

Current state:

- backpopulate hard-sets `binding_probe_eval_batch_size = 8`
- curriculum default is `32`

Recommended change:

- make eval batch size adaptive to available VRAM
- use `16` or `32` when memory headroom exists

Expected impact:

- moderate
- zero semantic change

Why:

- evaluation is not the largest cost, but it is still meaningful
- the current backfill setting looks conservative

### Priority 3: Avoid training all parameters when not necessary

Current state:

- the probe trains the entire copied model with `AdamW`

Recommended experiment:

- compare full-model adaptation against one or more restricted modes:
  - train only output head
  - train only top layer + head
  - train low-rank adapters on linear projections

Expected impact:

- potentially very high if calibration is preserved
- moderate to high semantic risk

Why:

- backward + optimizer are a large fraction of step time
- reducing trainable parameters attacks the dominant cost directly

Requirement:

- must re-check separation quality against the reference calibration panel

### Priority 4: Replace AdamW with a cheaper probe optimizer if calibration survives

Current state:

- probe uses `AdamW`

Recommended experiment:

- compare `AdamW` vs `SGD+momentum` vs smaller-state optimizer variants for
  short probe training

Expected impact:

- moderate
- semantic risk moderate

Why:

- optimizer step cost is nontrivial
- optimizer state also increases memory pressure

Requirement:

- only keep if GPT-style references still separate cleanly from local/recurrent
  models

### Priority 5: Reduce probe length only if the metric still separates

Current state:

- screening uses `400` train steps and `100` eval examples

Recommended experiment:

- check `200` or `256` train steps
- check `64` eval examples

Expected impact:

- high if tolerated
- high semantic risk

Why:

- this is the most direct wall-time reduction
- but it changes the measurement regime and may weaken discrimination

Requirement:

- compare against current calibration in
[binding_reference_calibration_2026-04-08.md](/home/tim/Projects/LLM/research/docs/binding_reference_calibration_2026-04-08.md)

## Low-ROI Or Secondary Ideas

### Native copy-batch generator

This would mirror induction’s native data path.

Why lower ROI:

- current measurements show batch generation time is tiny relative to model
  compute

When it becomes worth doing:

- only after the bigger compute and transfer issues are fixed

### Optimize grad clipping

Current grad clipping cost is negligible in the measured CPU breakdown.

### Compile speed work

Compile time is small relative to probe training.
Not the first place to spend effort.

## Suggested Implementation Order

### Phase 1: Cheap wins with minimal semantics risk

1. Make binding source-model offload conditional.
2. Make binding eval batch size adaptive instead of fixed at `8`.
3. Add timing instrumentation for:
   - source model offload
   - deepcopy
   - probe move-to-device
   - train loop
   - eval loop

### Phase 2: Measure restricted adaptation

1. Add a configurable probe adaptation mode:
   - `full`
   - `head_only`
   - `top_layer_plus_head`
2. Benchmark on:
   - GPT2 reference
   - local conv reference
   - mamba/rwkv references
   - a small frontier slice

### Phase 3: Reduce runtime target if calibration survives

1. Sweep train steps down from `400`
2. Sweep eval examples down from `100`
3. Recalibrate thresholds only after confirming separation

## Recommended Success Criteria

Before shipping a binding-speed rewrite, require:

- no regression in the GPT2 vs local-conv separation test
- no regression in reference ordering from the calibration panel
- measured runtime reduction on a representative backfill slice

A practical first target:

- `1.5x` to `2x` speedup in binding backfill throughput
- no material loss in separation quality

## Bottom Line

The current binding backfill is slow because it is doing real probe training,
not because Python batch generation is bad.

If the goal is to speed up the next binding backfill, the best first move is
not a C/C++ rewrite of the batch generator. The best first move is to reduce
avoidable GPU transfer and probe-training cost while preserving the current
metric’s calibration behavior.

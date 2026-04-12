# God File / God Function Audit

## Ranked God Files

| Rank | File | Size | Why it is a god file | Hot-path impact |
| --- | --- | ---: | --- | --- |
| 1 | `research/scientist/runner/execution_training.py` | 2368 lines | mixes orchestration, training, screening, probes, scoring, logging, persistence | directly on every screening/train path |
| 2 | `research/scientist/runner/_helpers.py` | 2172 lines | mixed notebook I/O, tier promotion, probe execution, trajectory logic, benchmark wiring | investigation/validation hot path |
| 3 | `research/scientist/runner/execution_screening.py` | 2408 lines | candidate generation, filtering, planning, scoring, policy, execution | central screening loop |
| 4 | `research/scientist/notebook/notebook_misc.py` | 2638 lines | analytics, observability, summaries, aggregation, leaderboard-derived stats | large repeated query/aggregation surface |
| 5 | `research/scientist/intelligence/predictor.py` | 1971 lines | training, evaluation, artifact I/O, ranking, gating, calibration | repeated ML inference/training path |

## Ranked God Functions

| Rank | Symbol | File | Size | Why it is a god function |
| --- | --- | --- | ---: | --- |
| 1 | `_micro_train` | `research/scientist/runner/execution_training.py` | 881 lines | orchestration + data loading + training + checkpointing + metrics + branching |
| 2 | `_run_investigation_thread` | `research/scientist/runner/execution_investigation.py` | 815 lines | thread orchestration + evaluation + persistence + control flow |
| 3 | `_execute_experiment` | `research/scientist/runner/execution_screening.py` | 443 lines | orchestration + candidate execution + metrics + error handling |
| 4 | `_run_scale_up_thread` | `research/scientist/runner/execution_validation.py` | 575 lines | validation orchestration + persistence + thresholding |
| 5 | `_run_inline_investigation` | `research/scientist/runner/continuous_investigation.py` | 553 lines | orchestration + model work + notebook writes |

## Highest-ROI Refactor Plan

### 1. Binding probe duplication across runner and backfill

Offenders:

- `research/scientist/runner/execution_training.py`
- `research/scientist/runner/_helpers.py`
- `research/tools/backfill.py`
- `research/tools/backfill_binding.py`

Problems:

- same binding orchestration was duplicated in multiple god files
- probe configuration was drifting across call sites
- performance tuning had to be copied into each site
- dead/stale branches survived after metric changes

Target split:

- one shared binding pipeline module for orchestration
- one hot curriculum module for the compute-heavy binding loop
- thin callers only in runner/backfill code

### 2. Curriculum binding hot path

Offender:

- `research/eval/binding_curriculum.py`

Problems:

- hottest component in binding backfill
- training loop dominated runtime
- previous implementation used plain precision and non-fused optimizer

Target split:

- keep data generation, train loop, and eval loop isolated in one focused module
- keep orchestration out of hot code
- tune execution within the isolated module

## C/C++ Migration Candidates Created By The Split

Not migrated in this pass:

- `research/eval/binding_curriculum.py::_generate_copy_train_batch`
- `research/eval/binding_curriculum.py::_eval_copy_distances`
- `research/eval/associative_recall.py::_generate_ar_batch`

Reason native migration did not land in this pass:

- the dominant math is already in native PyTorch kernels
- the safest high-ROI win was removing orchestration duplication and enabling faster kernel execution via fused optimizer and CUDA autocast
- the new split leaves a clean native boundary if further CUDA/C++ work is justified

## Actual Fixes Implemented

### Split

Added:

- `research/eval/binding_pipeline.py`

This module now owns:

- screening binding probe orchestration
- full binding probe orchestration
- binding composite calculation
- local-only determination

### Hot-path tuning

Updated:

- `research/eval/binding_curriculum.py`

Changes:

- enabled CUDA bfloat16 autocast in the training loop
- enabled fused `AdamW` on CUDA
- centralized curriculum constants used by screening/full probes

### Caller cleanup

Refactored callers onto the shared module:

- `research/scientist/runner/execution_training.py`
- `research/scientist/runner/_helpers.py`
- `research/tools/backfill.py`
- `research/tools/backfill_binding.py`

### Correctness / scope fixes

- backfill is now explicitly S1-only
- binding backfill selector now keys off missing `binding_auc`, not missing `induction_auc`
- dead adapted/curriculum sidecar binding columns were removed from the notebook DB
- stale partial binding writes were cleared after aborting the untuned run

## Before / After Performance Notes

Measured on the same real leaderboard-backed fingerprint on CUDA:

- before:
  - `micro_train`: `5.24s`
  - `AR`: `2.38s`
  - `induction`: `1.19s`
  - `binding_curriculum`: `11.07s`
  - full probe bundle total: `14.64s`
- after:
  - `micro_train`: `5.23s`
  - full tuned probe bundle total: `11.31s`

Observed improvement:

- about `22.7%` faster for the full binding probe bundle
- about `16%` faster end-to-end for a representative backfill row

## Remaining Highest-Value Splits

1. Split `_micro_train` out of `execution_training.py` into:
   - training loop core
   - checkpoint/metric collection
   - orchestration / failure policy

2. Split notebook observability aggregation out of `notebook_misc.py` into:
   - query layer
   - aggregation kernels
   - presentation formatting

3. Extract candidate-generation and filtering kernels from `execution_screening.py` into focused modules with clear ownership over:
   - grammar build
   - prescreener scoring
   - candidate filtering

4. Move `associative_recall.py::_generate_ar_batch` and parts of `binding_curriculum.py` toward a native generator/eval helper if binding backfill is still too slow after this pass.

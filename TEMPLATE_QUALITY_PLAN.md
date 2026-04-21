# Template Quality Plan

Created: 2026-04-20

## Objective

Tighten generation so the system spends evaluation budget on candidate-grade graphs rather than junk, while preserving:

- fingerprint-level deduplication
- existing promotion governance
- provenance-based training eligibility

This plan assumes no new backlog-filling phase. The repo already contains enough audit coverage and enforcement machinery to act now.

## Current State

- The live template registry exposes `175` active template entries in [research/synthesis/templates.py](/home/tim/Projects/LLM/research/synthesis/templates.py:1).
- The latest full template audit covers `181` catalog rows and classifies them as:
  - `103 keep`
  - `54 repair`
  - `20 restrict`
  - `4 retire`
- Within the `keep` set:
  - `48` are `healthy / productive`
  - `55` are `under-observed`
- Duplicate-graph governance already exists:
  - runtime dedup by `graph_fingerprint` in [research/scientist/runner/execution_screening_pipeline.py](/home/tim/Projects/LLM/research/scientist/runner/execution_screening_pipeline.py:197)
  - leaderboard duplicate insert blocking by fingerprint in [research/scientist/notebook/notebook_leaderboard.py](/home/tim/Projects/LLM/research/scientist/notebook/notebook_leaderboard.py:260)
  - ML corpus canonical grouping by fingerprint in [research/scientist/intelligence/ml_corpus.py](/home/tim/Projects/LLM/research/scientist/intelligence/ml_corpus.py:700)
- Promotion governance already exists:
  - trusted/promotable definitions in [research/scientist/trust_policy.py](/home/tim/Projects/LLM/research/scientist/trust_policy.py:1)
  - provenance-based promotion and screening-model eligibility in [research/scientist/notebook/program_provenance.py](/home/tim/Projects/LLM/research/scientist/notebook/program_provenance.py:500)
- The post-dedup semantic audit reports `0` duplicate fingerprints remaining in `program_results`.

## Core Conclusion

The immediate problem is not lack of audit coverage. The problem is that too much low-quality inventory is still allowed to consume search budget.

The right move is to narrow the active generator to the audited productive subset and route everything else into controlled lanes, without introducing a second identity scheme or a second governance system.

## Plan

### 1. Shrink the default candidate-generation surface immediately

Use only the `48` templates marked `healthy / productive` as the default search lane.

Do not spend candidate-grade screening budget on:

- `54 repair`
- `20 restrict`
- `4 retire`

Treat the `55 under-observed keep` templates as exploration-only, not default-quality candidates.

### 2. Split template traffic into explicit lanes

Replace the single mixed template pool with three explicit lanes:

- `candidate_lane`
  - input set: `healthy / productive` keep-set only
  - purpose: quality-candidate generation and model-training positives
- `explore_lane`
  - input set: `under-observed` keep-set only
  - purpose: bounded discovery
  - constraint: hard cap on budget share
- `salvage_lane`
  - input set: `repair` and `restrict` templates
  - purpose: template rehabilitation only
  - constraint: excluded from candidate-grade positives and promotion-facing traffic

The `retire` set stays fully disabled except for explicit back-compat or historical dedup references.

### 3. Move all selection policy out of the hot path

At experiment start, precompute:

- approved template names by lane
- lane-specific weight vectors
- alias canonicalization map
- slot/rule constraint tables
- budget shares and sampling caps

Inside the generation hot loop, leave only:

- choose lane
- sample template from preapproved arrays
- instantiate graph

Do not keep audit lookups, status checks, config translation, alias resolution, or policy branching inside the per-candidate execution loop.

### 4. Train only on governed canonical examples

Use the existing provenance and trust machinery as the sole gate for ML corpus quality.

Training positives:

- trusted, comparable candidate-grade rows

Training negatives:

- runtime search failures with complete provenance

Identity and dedup:

- continue grouping by canonical `graph_fingerprint`
- merge repeated evidence onto the same graph identity
- do not create a second graph key, second corpus identity, or family-specific dedup rule

### 5. Keep weak templates out until they earn re-entry

No template from `repair` or `restrict` returns to the candidate lane until it has:

- a repair change landed
- existing validation and template tests passing
- a bounded evaluation run showing acceptable yield

Re-entry should be evidence-based, not intuition-based.

### 6. Preserve governance by refusing parallel systems

Do not introduce:

- a second leaderboard
- a second promotion rule path
- a separate ML-only graph identity
- a bypass around trust/comparability/provenance checks

Reuse the current governance rails:

- `graph_fingerprint` for identity
- trust/comparability/provenance fields for eligibility
- existing leaderboard tier rules for promotion

## Operating Rule

The generator should produce mostly candidate-grade graphs by default. Exploration and salvage are allowed, but they must be explicitly budgeted side lanes rather than mixed into the primary candidate stream.

That makes the hot path smaller, the data cleaner, and the ML training set more aligned with the actual goal: predicting which math-op assemblies are worth testing.

# Low-S1 Execution Program
**Date**: 2026-03-21
**Status**: In progress

## Working Rules
- [x] Read required audit inputs: `CLAUDE.md`, `tasks/todo.md`, and low-S1 artifacts
- [ ] Resolve missing `AGENTS.md` or record repo-level absence as a blocker/input gap
- [x] Create shared coordination files under `artifacts/agent_sync/`
- [x] Write execution plan before additional code changes

## Execution Plan

### P0. Context-rule enforcement layer
- [ ] Encode component context intelligence into a single enforcement layer with explicit classifications: `general-use`, `restricted-use`, `structural`, `rehab`
- [ ] Represent predecessor, successor, motif, causal, and residual constraints in code rather than only in audit markdown
- [ ] Ensure enforcement decisions are evidence-backed and traceable to concrete components from the low-S1 audit

### P1. Wire rules into search/builder flow
- [ ] Apply context enforcement during graph generation/template instantiation
- [ ] Apply the same restrictions during mutation grammar derivation and novelty/fresh insertion paths
- [ ] Add validator coverage so invalid niche/structural placements are rejected or warned before compile/screening

### P2. Lock `local_window_attn`
- [ ] Verify the current `local_window_attn` implementation fix is present and keep it restricted to valid residual attention contexts
- [ ] Reject or deprioritize bad default placements that still trigger OOR or invalid standalone use

### P3. Reclassify components
- [ ] Classify audited low-S1 components into `general-use`, `restricted-use`, `structural`, or `rehab`
- [ ] Prevent structural ops from being judged or inserted as standalone learning carriers
- [ ] Prevent niche ops from being inserted outside valid graph families

### P4. Regression tests
- [ ] Add unit/regression tests for context-rule enforcement in builder, mutation, and validator paths
- [ ] Add tests that protect `local_window_attn` valid contexts and reject known-bad placements
- [ ] Add tests that preserve S1 strictness and avoid metric gaming through deletion or dilution

### P5. Fresh evidence reruns
- [ ] Identify polluted clusters that need fresh reruns after the rule-layer changes
- [ ] Define exact rerun commands, contexts, and pass/fail interpretation for post-fix evidence
- [ ] Record actual command outcomes in `artifacts/agent_sync/TEST_RESULTS.md`

## Explicit Answer Targets
- [ ] What gets fixed in code
- [ ] What gets fixed in placement rules
- [ ] What gets reclassified
- [ ] What needs fresh reruns
- [ ] What is blocked
- [ ] What is highest ROI next

## Review
- Pending

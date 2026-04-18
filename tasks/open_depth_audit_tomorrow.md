# Open-Depth Audit for Tomorrow

Run a controlled `open_depth` audit instead of downgrading the global cap.

## Goal

Keep the current production depth and op guardrails unchanged, but create a
test mode that removes or materially relaxes depth blocking so we can observe:

- what actually breaks
- what degrades
- which failures should become better local rules
- whether any genuinely good graphs are currently being suppressed

## Scope

- Add a dedicated test or audit harness for `open_depth` generation.
- Do not change default generation, screening, or validator behavior for
  normal runs.
- Use the same templates, slots, ops, routing logic, and validators, but run
  with relaxed depth and op ceilings in a separate path.
- Record failure reasons, graph pathologies, and any "wins" that only appear
  once depth is unconstrained.

## What to Measure

- legality failures
- context-rule failures
- template-rule failures
- dim-flow failures
- dead branches / unreachable nodes
- residual-bypass violations
- invalid token-order destruction
- fake routing / missing routing despite routed templates
- duplicate or degenerate deep chains
- repeated low-value `proj -> norm -> proj` scaffolds
- collapse into crutch ops or fallback motifs
- throughput and memory blowups
- whether any deep survivors actually improve loss or capability proxies

## What to Compare

- capped generation vs `open_depth` generation
- same seeds
- same targeted templates
- same benchmark slice
- same screening path except for the relaxed depth/op budget

## Required Outputs

- distribution of failure causes under `open_depth`
- list of templates that become pathological when depth is unconstrained
- list of ops or slot combinations that become failure magnets
- list of deep graphs that remain legal and look genuinely promising
- proposed new local rules derived from observed failures
- explicit call on whether current depth blocking is hiding good graphs or
  mostly suppressing junk

## Implementation Sketch

- add a test-only config mode such as `audit_open_depth=True`
- relax `max_depth` and `max_ops` only inside that mode
- keep all other legality checks on
- emit structured diagnostics per rejected graph
- aggregate by template, op, slot, and failure class
- save a machine-readable summary artifact for rule design

## Success Criterion

By the end of the experiment, we should know which failures should be fixed by:

- op contract rules
- slot compatibility rules
- template structural rewrites
- routing and assembly invariants

And we should know which, if any, are true cases where the current depth cap
is suppressing good architectures.

## Short Version

Do not remove the cap tomorrow.

Build the experiment that tells us whether the cap is protecting us from
garbage or hiding real winners.

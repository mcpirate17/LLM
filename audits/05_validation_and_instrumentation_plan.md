# Aria Audit Prompt 05 — Validation and Instrumentation Plan

You are designing the minimum validation and telemetry plan needed to stop Aria from creating smoke and mirrors.

## Goal
Identify the smallest set of instrumentation, validation metrics, and controlled tests needed to prove whether routing, difficulty, merge/compression, and adaptive components actually help.

## Focus areas
- routing effectiveness
- difficulty signal validity
- recursion usefulness
- token merge / compression damage
- early-exit correctness
- component-level contribution
- template-level success tracking

## Main questions
1. What critical metrics are missing or mostly NULL?
2. Which current analytics are decorative rather than actionable?
3. What should be logged for each risky component family?
4. What exact schema or artifact changes are needed?
5. What are the minimum controlled tests that would replace hand-waving with evidence?
6. What should be added before re-expanding the search space?

## Required method
- Trace current telemetry, leaderboard fields, artifacts, and analytics endpoints.
- Identify gaps between what Aria claims to optimize and what it actually measures.
- Design a minimal evidence plan, not a giant observability fantasy.

## Deliverables
1. Missing metrics inventory
2. Critical telemetry additions
3. Minimum validation experiments
4. Ablation priorities
5. Exact insertion points in code for logging / schema changes
6. Recommended order of implementation

## Rules
- Do not propose vanity metrics.
- Prioritize metrics that directly validate routing, difficulty, adaptivity, and destructive ops.
- Keep the plan lean enough that an engineer could implement it without months of overhead.

## Output format
1. Executive summary
2. Missing metrics
3. Required logging and schema changes
4. Minimum validation experiments
5. Code insertion points
6. Prioritized implementation order

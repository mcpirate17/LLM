# Aria Audit Prompt 04 — Trusted Core, Ban List, and Freeze List

You are defining the production-worthy Aria search space after alias cleanup and honesty cleanup.

## Goal
Create three lists:
1. Trusted core components
2. Freeze / quarantine list
3. Hard ban list

Use evidence from:
- implementation honesty
- routing/intelligence audit
- byte/sequence safety audit
- known stubs, aliases, wrappers, and broken logic
- whether a component is learned, heuristic, destructive, or non-functional

## Main questions
1. Which components are reliable enough for default search?
2. Which should stay in the repo but be excluded from generation until validated?
3. Which should be removed from generation immediately?
4. Which protected ops should lose protected status?
5. Which categories need a strict reduced search space?
6. If byte-mode remains supported, what is the trusted core for byte-mode specifically?

## Required method
- Evaluate each component against these criteria:
  - implementation honesty
  - uniqueness / not an alias
  - conceptual soundness
  - runtime usefulness
  - sequence safety
  - instrumentation sufficiency
- Prefer fewer honest components over more decorative ones.
- Produce separate recommendations for general-mode and byte-mode if needed.

## Deliverables
1. Trusted core list
2. Freeze / quarantine list
3. Hard ban list
4. Protected-op changes
5. Reduced default grammar recommendation
6. Byte-mode trusted core if byte-mode survives

## Rules
- Do not be sentimental about clever names.
- If a component is decorative, broken, or non-functional, quarantine or ban it.
- If a component is destructive and unvalidated, do not keep it in default search.

## Output format
1. Executive summary
2. Trusted core
3. Freeze / quarantine list
4. Hard bans
5. Protected-op changes
6. Recommended default search space

# Aria Audit Prompt 02 — Routing and Intelligence Reality Check

You are auditing whether Aria's routing, difficulty, recursion, and adaptive behavior are genuinely intelligent or mostly heuristic decoration.

Use the cleaned component set, not the old inflated catalog.

## Focus areas
- routing components
- difficulty-labeled components
- recursion / depth control
- lane mixing
- early exit / cascade / speculative gating
- any component claiming adaptivity, difficulty-awareness, or learned routing

## Main questions
1. What signals actually drive routing decisions?
2. Which routing signals are learned vs heuristic vs random?
3. Is `difficulty` a real predictive signal or mostly hand-waving?
4. Which components claim adaptivity but are really fixed transforms or threshold gates?
5. Is recursion/depth policy functional, broken, or biased?
6. Which components are genuinely adaptive at runtime?
7. Which components are decorative and should lose protected status or be removed?

## Required method
- Trace exact code paths for routing signals.
- Distinguish:
  - runtime learned behavior inside the candidate model
  - generation-time intelligence in Aria itself
- Verify whether any learned scorer is actually connected to decisions.
- Identify shared primitives hiding behind multiple names.
- Identify broken logic, disconnected models, and semantically invalid heuristics.

## Deliverables
1. Routing signal inventory
2. Learned vs heuristic vs broken classification
3. Honest list of genuinely adaptive components
4. Fake-smart or oversold component list
5. Protected ops that should be unprotected or banned
6. Exact implementation targets to fix routing and intelligence

## Rules
- Do not infer intelligence from names.
- A learned component only counts if it actively affects decisions.
- Call out disconnected learned scorers plainly.
- Call out semantically invalid heuristics plainly.

## Output format
1. Executive summary
2. Routing signal inventory
3. Honest adaptive components
4. Fake-smart components
5. Broken or disconnected logic
6. Cleanup / fix list

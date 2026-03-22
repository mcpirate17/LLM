# Final Summary

## Current State
- The low-S1 execution program is no longer treating low S1 as proof a component is broken.
- Active context-rule ownership is now wired through [context_rules.py](/home/tim/Projects/LLM/research/synthesis/context_rules.py), [grammar.py](/home/tim/Projects/LLM/research/synthesis/grammar.py), and [validator.py](/home/tim/Projects/LLM/research/synthesis/validator.py).
- Targeted regression coverage exists in [test_context_rules.py](/home/tim/Projects/LLM/research/tests/test_context_rules.py), and the locally run targeted tests passed.
- Broad recorded rerun evidence now exists at [exploration_20260321_145219.md](/home/tim/Projects/LLM/research/reports/exploration_20260321_145219.md) and [exploration_20260321_145219.json](/home/tim/Projects/LLM/research/reports/exploration_20260321_145219.json): `44/46` covered, `41/46` rapid-pass, `1/46` S1-pass.
- The two uncovered spiking ops from that campaign (`sparse_threshold`, `stdp_attention`) are now covered by targeted follow-up forced generation and evaluation.
- The earlier `templates.py` duplicate-policy cleanup debt is effectively closed; `context_rules.py` is the active policy owner.
- Forced exploration for `local_window_attn` and `n_way_sparse_router` no longer uses one minimized canned graph; it now selects from a small family of valid wrappers.

## Explicit Answers

### What gets fixed in code
- `local_window_attn` remains the only accepted real component-code fix from the audit.
- The codebase now also has an active shared placement-policy owner in `context_rules.py` instead of leaving the low-S1 audit as markdown-only guidance.
- `n_way_sparse_router` now also has a validated bf16/autocast forward-path fix in `compiler.py` plus a placement-rule tightening in `context_rules.py`.
- Spiking reachability is fixed for `sparse_threshold` / `stdp_attention` by correcting the invalid motif chain and adding a direct forced-graph path in `explore_under_observed.py`.

### What gets fixed in placement rules
- Invalid `local_window_attn` standalone/default placement is rejected by the active rule layer.
- Structural misuse is rejected for at least the validated cases covered by the rule layer and tests, including standalone `identity`.
- Fresh generation and validator paths now consume the shared context policy rather than relying only on ad hoc template behavior.
- Niche spiking / tropical / hyperbolic `MATH_SPACE_RULES` additions are in place and tested.
- `n_way_sparse_router` now requires normalized predecessor context plus an immediate `rmsnorm` / `layernorm` / `linear_proj` successor.

### What gets reclassified
- `local_window_attn`: restricted-use / rehab-style handling until fresh default-search evidence is clean.
- Structural-only class: masks, splits/concat family, identity, and reduction-style structural carriers should not be judged as standalone learning carriers.
- Valid-but-weak ops remain valid-but-weak, not forbidden.
- Fresh reruns strengthen the “valid-but-weak” interpretation for `embedding_lookup`, `log`, `sqrt`, `sign_ste`, `geometric_product`, and `mod_topk` more than the old stale 0%-S1 table did.

### What needs fresh reruns
- The broad forced rerun campaign has already been run once.
- Immediate follow-up reruns still needed only for uncovered ops and any post-fix targeted confirmations:
  - any focused post-fix rerun for `local_window_attn` and `n_way_sparse_router` after downstream cleanup

### What is blocked
- `AGENTS.md` is absent from the repo, so there is an input-gap on any workflow guidance that was supposed to live there.
- `templates.py` still contains duplicated, non-authoritative context-rule logic; active consumers no longer depend on it, but it remains cleanup debt.
- There are no remaining known zero-coverage blockers from the recorded rerun campaign after the targeted spiking follow-up.
- Remaining blockers are now ordinary follow-up items, not split policy ownership.

### What is highest ROI next
- Highest ROI next is any narrowly targeted post-fix reruns needed for `local_window_attn` and `n_way_sparse_router` downstream cleanup, plus normal hygiene work on the remaining silent catch-and-drop sites.
- The latest builder update pushed in the right direction for exploration quality: smarter surrounding context without collapsing graph space. What is still missing is a quieter targeted eval harness that can compare those wrapper families without Triton fallback log spam dominating the run.
- Fresh post-fix reruns are now in:
  - `local_window_attn`: compile/forward OK, still rapid-fail in the tested forced graph
  - `n_way_sparse_router`: compile/forward OK, rapid-pass, S1-fail in the tested forced graph
- Silent-catch cleanup is only partially complete; the highest-signal bare `pass` sites are gone, but `templates.py` still has many explicit fallback catches that need case-by-case review.

## Verified Commands
- `pytest -q research/tests/test_context_rules.py`
- `pytest -q research/tests/test_under_observed_coverage.py -k 'exploration_config_creation or exploration_config_custom_boost'`
- `pytest -q research/tests/test_math_space_rules.py research/tests/test_context_rules.py research/tests/test_n_way_sparse_router_regression.py`
- `python -m research.tools.explore_under_observed --mode=forced --threshold=50 --graphs-per-op=10 --rapid-steps=500 --n-graphs=500 --device=cuda --record`

# Task Queue

## Completed
- [x] Claude terminal (`Volta`): audit and sharpen predecessor/successor/motif/causal/residual rules from the low-S1 artifacts.
- [x] Coordinator: verify `AGENTS.md` absence and convert the audit into the shared execution program.
- [x] Coordinator: review and tighten the active context-rule wiring so `context_rules.py` is the active rule owner for generation and validation consumers.

## Active
- [ ] External agent: `T1` `local_window_attn` shared-memory overflow
  Scope: fix the real config/hardware failure in default-search style contexts without breaking valid residual-attention behavior.
  Files: `research/synthesis/compiler_ops_attention.py`, maybe `research/synthesis/templates.py`, focused tests
  Acceptance criteria: reproduce or point to the failing default-search style case; fix/clamp invalid `window_size` or related config; valid context still works; exact commands and outcomes recorded in `TEST_RESULTS.md`.

- [ ] External agent: `T6` structural-op S1 attribution exemption
  Scope: stop structural ops from being judged as standalone learning carriers in per-op S1 attribution, without weakening actual model S1 screening.
  Files: runner / notebook / analytics / scoring code, focused tests
  Acceptance criteria: structural ops excluded or separately classified in per-op S1 attribution; actual model S1 unchanged; exact commands and outcomes recorded in `TEST_RESULTS.md`.

## Pending
- [ ] Coordinator: define the post-fix rerun campaign
  Scope: exact rerun components, commands, and interpretation rules
  Files: `artifacts/low_s1_fresh_reruns.md`, `artifacts/agent_sync/TEST_RESULTS.md`, `artifacts/agent_sync/FINAL_SUMMARY.md`
  Planned commands:
  - `python -m research.tools.explore_under_observed --mode=forced --threshold=50 --graphs-per-op=10 --rapid-steps=500 --n-graphs=500 --device=cuda --record`
  - faster dry pass: `python -m research.tools.explore_under_observed --mode=forced --threshold=50 --graphs-per-op=5 --rapid-steps=500 --no-s1 --device=cuda`
  Interpretation rules:
  - rerun-needed polluted clusters should be judged from fresh evidence, not stale bf16/runtime history
  - restricted-use ops should only be judged from valid-context evidence
  - structural-only ops should be reclassified, not used as standalone learner evidence

- [ ] Follow-up implementation: spiking `MATH_SPACE_RULES`
  Scope: add domain-specific placement constraints in `research/synthesis/motifs.py`

- [ ] Follow-up implementation: tropical `MATH_SPACE_RULES`
  Scope: add domain-specific placement constraints in `research/synthesis/motifs.py`

- [ ] Follow-up implementation: hyperbolic `MATH_SPACE_RULES`
  Scope: add domain-specific placement constraints in `research/synthesis/motifs.py`

- [ ] Follow-up implementation: `n_way_sparse_router` divisibility/init fix
  Scope: compiler/init bugfix plus focused regression coverage

- [ ] Cleanup debt: remove duplicated, non-authoritative context-rule helpers from `research/synthesis/templates.py`

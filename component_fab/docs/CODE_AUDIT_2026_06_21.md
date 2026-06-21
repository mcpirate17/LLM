# component_fab Code Audit — 2026-06-21

## Verdict

`component_fab` is not throwaway AI slop. It has a coherent architecture, real validator gates, ledger persistence, promotion policy, surrogate/fidelity/deep-probe tooling, and explicit safety controls. The weak spots are mostly orchestration quality, inconsistent CLI contracts, and research-validity defaults that make it too easy to produce misleading promotions or confusing no-promotion runs.

## Audited path

- `component_fab/tools/run_autonomous.py`
- `component_fab/tools/run_invention.py`
- `component_fab/tools/run_deep_probe.py`
- `component_fab/improver/deep_probe.py`
- `component_fab/tools/run_fidelity.py`
- `component_fab/tools/run_surrogate.py`
- `component_fab/tools/run_lm_probe.py`
- `component_fab/tools/run_probe_bench.py`
- `component_fab/tools/run_trust_audit.py`
- `component_fab/tools/_cli.py`
- `component_fab/policies/promotion.py`
- existing Colab prep in `research/tools/prepare_colab_probe_backfill.py`

## What is good

1. The autonomous loop has real separation of concerns: enumeration, screening, grading, promotion, ARIA registration, rotation, and reporting are distinct functions.
2. Promotion policy is much better than a naive score threshold. It has streak logic, rejection floor, CI evidence gating, Pareto/niche promotion hooks, and fail-closed promotion evidence.
3. Deep-probe tier is the strongest part of the system. It explicitly recognizes that nano composite can saturate and uses a deeper bake-off against frontier baselines.
4. Fidelity ladder exists and measures R0/R1 rank correlation instead of assuming nano evaluations mean anything.
5. Probe benchmark tooling exists, which is important because this system can otherwise burn cycles on expensive gates without knowing which probes dominate runtime.
6. Colab packaging infrastructure already bundles `component_fab` indirectly through the cheap-probe backfill tarball.

## Problems / crap-risk

### P0 — `component_fab` does not have a first-class Colab worker

The repo has mature Colab packaging for cheap probe backfills, but not a direct `component_fab` worker that can run autonomous cycles, invention runs, fidelity runs, or deep-probe batches from Drive.

Impact: Colab execution is possible only indirectly or manually. That is brittle and will confuse future runs.

Fix:
- Add `component_fab/tools/prepare_colab_component_fab.py`.
- Reuse the source-bundling logic from `research/tools/prepare_colab_probe_backfill.py`.
- Generate `run_component_fab_colab.py` and optionally `.ipynb`.
- Support modes: `smoke`, `autonomous`, `deep_probe`, `fidelity`, `surrogate`.
- Persist status JSON, logs, and output reports back to Drive.

### P1 — `run_invention.py` records smoke_pass incorrectly

`run_autonomous.py` records smoke pass from forward/backward smoke fields. `run_invention.py` records smoke pass from `solo.promoted`, which includes more than smoke. That pollutes ledger semantics.

Impact: invention ledger entries can undercount smoke-passing candidates if they failed property/cross-check promotion. This affects promotion evidence and historical analysis.

Fix:
```python
solo_payload = result.get("solo") or {}
smoke = solo_payload.get("smoke") or {}
smoke_pass = bool(smoke.get("forward_passed") and smoke.get("backward_passed"))
```
Then pass `smoke_pass=smoke_pass` into `ledger.record_grade`.

### P1 — Default autonomous promotion behavior is confusing

`run_autonomous.py` defaults `--paired-seeds` to `0`, but promotion policy is fail-closed for complete promotion evidence. That is scientifically safer, but operationally confusing: a default run can appear to discover good candidates while never promoting them because paired evidence is absent.

Fix options:
- Make default `--paired-seeds 3` for promotion-capable runs.
- Or add `--promotion-mode {screen,promote}` where:
  - `screen`: cheap run, no promotion expected.
  - `promote`: requires paired seeds, range/niche evidence if enabled, and clear reports.
- At minimum, print a warning when `paired_seeds == 0` and promotion evidence is required.

### P1 — CLI contract is inconsistent

Different tools use different names for the same concepts:

- `run_deep_probe.py`: `--ledger-path`, `--output`
- `run_fidelity.py`: `--ledger`, `--store`, `--out`
- `run_surrogate.py`: `--ledger`, `--out`
- `run_lm_probe.py`: shared `--ledger`, `--output`
- `run_invention.py`: shared `--ledger`, `--output`

Impact: Colab automation and shell scripts become annoying and error-prone.

Fix:
- Standardize on shared `component_fab.tools._cli.add_common_args`.
- Preserve old flags as aliases for backward compatibility.

### P1 — Deep-probe default seed count is too weak

`run_deep_probe.py` defaults `--seed-count 1`. For frontier claims, that is too weak. The module itself correctly says nano evidence is noisy, but the CLI still defaults to a single seed.

Fix:
- Default `--seed-count 3` for `run_deep_probe.py`.
- Add `--fast` if single-seed runs are wanted.
- Promotion should require multi-seed evidence.

### P2 — Invention run cycle semantics are odd

`run_invention.py` records each active spec using `cycle=index`. That makes one invocation look like multiple cycles. It is not catastrophic because invention uses `promote_min_streak_cycles=1`, but it is semantically dirty.

Fix:
- Add `--cycle` argument defaulting to `1`, record all specs under that cycle.
- Or use a timestamp/run_id field in metadata and keep cycle stable.

### P2 — Surrogate acceptance is report-only

`run_surrogate.py` computes acceptance and prints it, but the autonomous loop depends on the user manually choosing `--selection surrogate`.

Fix:
- Add a small gate file like `component_fab/catalog/surrogate_policy.json`.
- `run_autonomous --selection auto` should use surrogate only when the latest report passes acceptance.

### P2 — Fidelity ladder is useful but too manually staged

`run_fidelity.py` correctly appends R0/R1 scores and reports rank correlation, but it does not feed weight-demotion policy back into autonomous ranking.

Fix:
- Emit a machine-readable `fidelity_policy.json` with per-metric trust weights.
- Load it from ranking/autonomous selection.

### P2 — Colab source bundle should be split by mode

Current cheap-probe tarball includes enough repo source, but a first-class fab Colab runner should avoid copying unnecessary local artifacts and DB files.

Fix:
- Bundle source-only plus specific input artifacts.
- Exclude generated catalog reports unless explicitly needed.
- Include a manifest with git commit SHA, branch, and generated timestamp.

## Recommended work order

1. Fix `run_invention.py` smoke-pass ledger semantics.
2. Add first-class `prepare_colab_component_fab.py`.
3. Standardize CLI flags through `_cli.py`.
4. Make deep-probe promotion multi-seed by default or add explicit promotion mode.
5. Add autonomous warning when `paired_seeds=0` with fail-closed evidence.
6. Add `--selection auto` backed by latest accepted surrogate report.
7. Make fidelity report produce ranking weights.
8. Add smoke tests for each runner parser and dry-run path.

## Minimal Colab worker contract

A correct Colab worker should:

1. Mount Drive.
2. Install only required dependencies.
3. Extract source bundle.
4. Set `PYTHONPATH` to extracted source root.
5. Validate imports:
   - `component_fab`
   - `component_fab.tools.run_autonomous`
   - `component_fab.tools.run_deep_probe`
6. Validate expected artifacts:
   - ledger path exists or can be created
   - catalog directory is writable
7. Run selected mode.
8. Stream stdout/stderr to Drive log.
9. Write status JSON every few seconds.
10. Write final report JSON to Drive.

## Bottom line

The system is directionally strong. The biggest frontier-research risk is not bad component code; it is bad evidence discipline. The project should bias toward fewer promotions with higher-quality evidence: multi-seed, fidelity-correlated, deep-probed, and transplant-tested. That is the path from toy discovery to credible frontier architecture research.

# Eval Working Board

Live coordination board for `research/eval`.

Historical claim logs and benchmark notes were moved to
`research/eval/WORKING.archive.md` on 2026-04-04 because the prior file had
stale claims, duplicate Codex identities, and mixed active/completed state.

## Rules

1. Claim work before editing a file.
2. Do not edit files claimed by another Codex unless the claim is cleared here.
3. Keep this file short. Active claims and blockers only.
4. Put completed work and benchmark history in `WORKING.archive.md`, not here.
5. For hot-path changes, run the required perf commands before and after.
6. If work crosses into `research/scientist`, record the claim here and use the
   same benchmark surfaces listed below. Do not invent separate perf numbers.

## Required Performance Validation

- `cd /home/tim/Projects/LLM && make profile-hotpaths`
- `python /home/tim/Projects/LLM/research/training/profiling.py`
- `cd /home/tim/Projects/LLM && python -m research.eval.benchmark_reference_runner`
- `cd /home/tim/Projects/LLM && python - <<'PY'`
- `import json`
- `from research.eval.benchmark_reference_runner import benchmark_reference_runner`
- `print(json.dumps(benchmark_reference_runner(n_steps=128, repeats=3), indent=2))`
- `PY`

## Active Claims

- none recorded

## Blocked / Coordination Notes

- Active work has moved into `research/scientist` because the remaining
  bottlenecks are outside `research/eval`. Check these claims before touching
  the runner or benchmark harness.
- If a claim is completed, move the details to `WORKING.archive.md` and remove it
  from this file.

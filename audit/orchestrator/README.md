# Autonomous multi-model audit orchestrator

Audits every directory under `LLM/`, fixes what it finds, and **loops until the
deterministic violation count stops dropping** — i.e. until there is no remaining ROI and
the codebase is as lean as the tools can prove. Standards come from `CLAUDE.md` and
`GLOBAL_DEV_PROMPT.md`. Results land in `audit/`.

## The model contract (why it's cheap and honest)

| Phase | Who | Why |
|---|---|---|
| **Audit** (read-only, parallel) | **all** of `claude`, `codex`, `agy`, `minimax` | Different models, different blind spots → diverse findings. Cheap models carry the broad sweeps. |
| **Triage** (merge → ordered plan) | `claude` | One brain dedupes every model's findings into an ROI-ordered, gated fix plan. |
| **Fix** (sequential, gated) | `codex` (easy/med), `claude` (hard) | Only these two ever mutate code. **The auditor is never the fixer** — keeps token cost down and avoids a model rubber-stamping its own report. |

## The safety contract (why autonomy is OK here)

A green test suite does **not** prove the scoring numbers are unchanged on this NAS
pipeline. So every fix runs alone on its own git branch behind a **two-part gate**:

1. **smoke tests** stay green, and
2. a **fixed-seed scoring run reproduces a committed baseline bit-for-bit**.

Gate pass → auto-merged to `master`. Gate fail → branch deleted, fix discarded. A fix that
quietly reconverges on a softmax-shaped path moves the scoring numbers and is **rejected
here** — the mission is enforced by the gate, not by trust.

## The loop / "no remaining ROI"

`detectors.py` computes a deterministic violation vector (god files >1250 LOC, god
functions >100 LOC, duplicates, dead code, lint) every round. The loop continues while that
total falls and stops when a round fails to reduce it. This is why "you ran the audit but
the god files were never split" **cannot** happen — the detector re-checks and re-queues
every round until they're actually gone.

## Use (no input needed beyond these)

```bash
cd audit/orchestrator

python orchestrate.py doctor            # validate config + CLIs + gate (run first)
python orchestrate.py capture-baseline  # freeze the fixed-seed scoring baseline (once)
python orchestrate.py loop              # autonomous: audit → triage → fix → gate → repeat
```

Other commands: `measure` (print the current violation vector), `audit --round N` (one
read-only fan-out only). The loop also auto-captures a baseline on first run if missing.

## Files

- `orchestrate.py` — the loop + subcommands.
- `adapters.py` — model-agnostic CLI launch/parse (`claude`/`codex`/`agy`/`minimax`).
- `detectors.py` — the deterministic ROI oracle.
- `gate.py` — smoke + fixed-seed scoring gate.
- `playbook.json` — audit passes (from the two prompt files) + triage/fix templates. Edit freely.
- `config.toml` — models, gate commands, loop budget, targets.
- `../findings/round-NN/` — per-model audit reports + the triage plan.
- `state/ledger.jsonl` — full audit→fix→gate trail; `state/reference_baseline.json` — frozen scoring.

## Tuning

Everything is in `config.toml`: `loop.max_rounds`, `loop.roi_min_improvement`,
`loop.targets`, the audit `models` pool, `fix.hard_severity` routing, `merge.mode`
(`auto`/`branch`/`accumulate`), and the two `gate` commands.

## Relationship to `conductor/` and `research/tools/full_repo_audit.py`

Those remain the canonical **deterministic** gates (pre-commit + CI weekly). This
orchestrator is the **agentic layer** on top: it uses the same thresholds as its ROI
oracle, then dispatches LLMs to actually *fix* what those gates only *report*.

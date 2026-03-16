# aria-commands

Five lean persona commands for Claude Code, tailored to the Aria workspace.
Drop into `.claude/commands/` — no build step, no dependencies, no setup script.

## Install

```bash
cp aria-commands/*.md /home/tim/Projects/LLM/.claude/commands/
```

Or globally:

```bash
cp aria-commands/*.md ~/.claude/commands/
```

## Commands

| Command | Mode | Use when |
|---------|------|----------|
| `/aria-architect` | System design | Evaluating a structural change, new coupling, boundary decision |
| `/aria-scientist` | Research integrity | Designing or reviewing an experiment, eval metric, or search change |
| `/aria-kernel` | Native/compute | Touching `aria_core`, CUDA kernels, Triton, pybind11, `libaria_runtime.so` |
| `/aria-bridge` | Integration seams | Touching `bridge.py`, `native_runner_adapter`, `shared_api`, `component_registry` |
| `/aria-review` | Pre-commit review | Final check before committing anything across the workspace |

## Philosophy

Each command narrows Claude to one cognitive mode. The modes don't overlap:

- **Architect** thinks in boundaries and long-term structure.
- **Scientist** thinks in experimental validity and eval integrity.
- **Kernel** thinks in correctness, parity, and performance.
- **Bridge** thinks in contracts, failure modes, and coupling debt.
- **Review** thinks in bugs, silent failures, and test coverage.

Running all five on a significant change takes ~10 minutes and catches different classes of problems.

# component_fab.runner

The autonomous runner is split by responsibility:

- `cycle.py`: one-cycle orchestration and human-readable cycle summaries.
- `selection.py`: terminal filtering, measured/NAS screening, quality ordering, and surrogate acquisition selection.
- `grading.py`: per-spec grading, grade metadata assembly, survivor buffering, niche metadata, and ledger writes.
- `promotion.py`: ARIA registration side effects for newly promoted components.

`component_fab.tools.run_autonomous` should stay CLI-only: argument parsing, signal handling, rotation, loop driving, and report emission.
